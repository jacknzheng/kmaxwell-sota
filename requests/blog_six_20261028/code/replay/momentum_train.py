"""Pinned eight-rank training with an explicit post-restore momentum intervention.
Only Muon's scheduler changes. Historical rings, Adam state, RNG, and rates survive.
Health observations and disk backpressure never select scientific outcomes.
"""
import json, math, pathlib, re, shutil, time
import torch
import torch.distributed as dist
from records.track_3_optimization import run
from records.track_3_optimization.harness import hooks, optimizers, loop
from records.track_3_optimization.harness.checkpoint import save_checkpoint
from momentum_schedule import TiedStepNesterov

_BUILD = optimizers.build_optimizer
_RESTORE = hooks.restore_checkpoint
_STATE = None
_CONFIG = None
OLD = dict(kind='ema', decay=.95, nesterov_mix=.95)

def new_identity(spec):
    return optimizers.scheduler_config_sha256({'momentum': spec})

def build_optimizer(name, params, *, momentum=None, **kw):
    if isinstance(momentum, dict) and momentum.get('kind') == 'tied_step_nesterov':
        assert name == 'muon'
        expected = {'kind','beta_after','beta_before','start_step'}
        if set(momentum) != expected: raise ValueError('Unexpected intervention fields')
        built = _BUILD(name, params, momentum=OLD, **kw)
        built.momentum_scheduler = TiedStepNesterov(**{k:v for k,v in momentum.items() if k!='kind'})
        for group in built.param_groups: group['scheduler_config_sha256'] = new_identity(momentum)
        return built
    return _BUILD(name, params, momentum=momentum, **kw)

def emit(config, name, row):
    if dist.get_rank() == 0:
        p=pathlib.Path(config['output_dir']).parent/name;p.parent.mkdir(parents=True,exist_ok=True)
        with p.open('a') as f:f.write(json.dumps(dict(observed_at=time.time(),**row),allow_nan=False)+'\n')

def restore_checkpoint():
    original = _RESTORE()
    def hook(config,state):
        global _STATE,_CONFIG
        receipt=json.loads((pathlib.Path(config['resume_checkpoint'])/'COMPLETE.json').read_text())
        selected=[]
        for row,(_,optimizer) in zip(config['optimizer_groups'],state['optimizer'].groups):
            spec=row.get('hyperparams',{}).get('momentum',{})
            if spec.get('kind')=='tied_step_nesterov':
                selected.append((spec,optimizer))
                # The initial shared checkpoint was created under the old scheduler.
                # Check/load its original identity rather than bypassing the loader.
                if receipt['boundary']==spec['start_step']:
                    for group in optimizer.param_groups:group['scheduler_config_sha256']=new_identity(OLD)
        if len(selected)!=1:raise ValueError('Expected exactly one selected Muon group')
        state=original(config,state)
        for spec,optimizer in selected:
            optimizer.momentum_scheduler=TiedStepNesterov(**{k:v for k,v in spec.items() if k!='kind'})
            for group in optimizer.param_groups:
                if group['iteration']!=receipt['boundary']:raise ValueError('Restored optimizer clock differs')
                group['scheduler_config_sha256']=new_identity(spec)
        _STATE,_CONFIG=state,config
        original_log=state['print_log']
        def log(message,*args,**kwargs):
            original_log(message,*args,**kwargs)
            match=re.match(r'step:(\d+)/\d+ val_loss:([^ ]+)',str(message))
            if match:emit(config,'validation.jsonl',dict(boundary=int(match[1]),val_loss=float(match[2]),tokens=config['val_tokens'],subset='fixed validation prefix',logged_precision_digits=5))
        state['print_log']=log
        state['momentum_step_wall']=time.perf_counter()
        emit(config,'intervention_receipt.jsonl',dict(parent=receipt,boundary=config['start_step'],spec=selected[0][0],history='all saved FP32 rings retained; scheduler changed after exact restore',rates=[dict(pattern=row['pattern'],lr=opt.param_groups[0]['lr'],initial_lr=opt.param_groups[0]['initial_lr'],weight_decay=opt.param_groups[0].get('weight_decay')) for row,(_,opt) in zip(config['optimizer_groups'],state['optimizer'].groups)]))
        return state
    return hook

def wait_for_space(config,needed):
    while True:
        free=shutil.disk_usage(config['output_dir']).free if dist.get_rank()==0 else 0
        okay=torch.tensor([int(free>=needed)],device='cuda',dtype=torch.int32)
        dist.broadcast(okay,0)
        if okay.item():return
        emit(config,'disk_backpressure.jsonl',dict(free_bytes=free,required_bytes=needed,action='wait for verified durable capture reclamation'))
        time.sleep(5)

def capture_space():
    def hook(config,state):
        t=state['step']
        if not state.get('momentum_capture_bank_space_checked'):
            count=len({u for lo,hi in config.get('capture',{}).get('windows',[]) for u in range(lo,hi+1)})
            wait_for_space(config,count*2_300_000_000+10_000_000_000)
            state['momentum_capture_bank_space_checked']=True
        if any(lo<=t<=hi for lo,hi in config.get('capture',{}).get('windows',[])):
            wait_for_space(config,config.get('capture_reserve_bytes',10_000_000_000))
        return state
    return hook

def checkpoint_state():
    def hook(config,state):
        boundary=state['step']+1
        if boundary in config.get('checkpoint_boundaries',[]):
            folder=pathlib.Path(config['checkpoint_dir'])/f'boundary_{boundary}'
            if folder.exists():raise RuntimeError('Preserve existing checkpoint; explicit resume required')
            wait_for_space(config,config.get('checkpoint_reserve_bytes',105_000_000_000))
            save_checkpoint(config,state,boundary)
        return state
    return hook

def accumulate_global_gradient(model,batch,microbatch_sequences,local_gradient_callback=None):
    # Same forward/backward and parameter all-reduce order as the pinned harness.
    inputs,targets=batch;assert len(inputs)%microbatch_sequences==0
    step=_STATE['step'];observe=(step%25==0 or step==_CONFIG['start_step'])
    observed_loss=torch.zeros((),device=inputs.device,dtype=torch.float32) if observe else None
    for i in range(len(inputs)//microbatch_sequences):
        loss=model(inputs[i*microbatch_sequences:(i+1)*microbatch_sequences],targets[i*microbatch_sequences:(i+1)*microbatch_sequences])
        if not torch.isfinite(loss):raise RuntimeError('non-finite training loss')
        loss.backward()
        if observe:observed_loss.add_(loss.detach().float())
    if local_gradient_callback is not None:local_gradient_callback()
    for name,p in model.named_parameters():
        assert p.grad is not None,name
        dist.all_reduce(p.grad,op=dist.ReduceOp.SUM)
    if observe:
        dist.all_reduce(observed_loss,op=dist.ReduceOp.SUM)
        norm_sq=sum(p.detach().float().square().sum() for p in model.parameters())
        emit(_CONFIG,'health.jsonl',dict(update=step,train_loss=float(observed_loss)/_CONFIG['batch_tokens'],train_tokens=_CONFIG['batch_tokens'],parameter_l2=float(norm_sq.sqrt())))

def print_training_progress(every=1):
    def hook(config,state):
        if every and state['step']%every==0:
            torch.cuda.synchronize();now=time.perf_counter()
            emit(config,'training_progress.jsonl',dict(update=state['step'],boundary=state['step']+1,wall_step_seconds=now-state['momentum_step_wall']))
            state['momentum_step_wall']=now
        return state
    return hook

_FACTORIES=hooks.hook_factories

def factories():
    result=_FACTORIES();result['capture_space']=capture_space;return result
optimizers.build_optimizer=build_optimizer
hooks.restore_checkpoint=restore_checkpoint
hooks.checkpoint_state=checkpoint_state
hooks.print_training_progress=print_training_progress
hooks.hook_factories=factories
loop.accumulate_global_gradient=accumulate_global_gradient
if __name__=='__main__':run.main()
