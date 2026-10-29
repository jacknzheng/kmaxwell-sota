"""One explicit momentum branch from the shared original full checkpoint."""
import json
from pathlib import Path
import yaml
from momentum_schedule import BETA_KAPPA_HALF,BETA_KAPPA_DOUBLE
FIRST_WINDOWS=[[2010,2013],[2016,2019],[2109,2112],[2359,2362],[2609,2612]]

def build(home,arm,parent=None,last=2612,windows=None,checkpoint=True):
    home=Path(home);parent=Path(parent) if parent else home/'restart/shared_2009'
    if arm not in ('lower_zeta','higher_zeta'):raise ValueError(arm)
    receipt=json.loads((parent/'COMPLETE.json').read_text())
    if receipt['world_size']!=8:raise ValueError('Expected original eight-rank checkpoint')
    for rank in range(8):
        if not (parent/f'rank{rank}.pt').is_file():raise FileNotFoundError(rank)
    output=home/'continuation'/f'seed0_{arm}_to_{last}'
    cfg=yaml.safe_load((home/'parent.yaml').read_text())
    cfg.update(run_id=output.name,output_dir=str(output/'training'),checkpoint_dir=str(output/'restart'),checkpoint_every=999999,
               checkpoint_boundaries=[last] if checkpoint else [],resume_checkpoint=str(parent),resume_protocol_sha256=receipt['protocol_sha256'],
               resume_anchor=None,stop_step=last+1,require_world_size=8,microbatch_sequences=8,val_tokens=1048576,
               train_data='data/fineweb10B/fineweb_train_*.bin',val_data='data/fineweb10B/fineweb_val_*.bin',
               momentum_arm=arm,capture_reserve_bytes=10_000_000_000,checkpoint_reserve_bytes=105_000_000_000)
    cfg.pop('start_step',None)
    cfg['capture']=dict(root=str(output/'captures'),windows=windows or FIRST_WINDOWS,anchors=[],window_gradient_only=True)
    cfg.setdefault('pre_forward',[]).append(dict(name='capture_space'))
    beta=BETA_KAPPA_HALF if arm=='lower_zeta' else BETA_KAPPA_DOUBLE
    for group in cfg['optimizer_groups']:
        group['lr_schedule']=dict(kind='piecewise_constant',boundaries=[],factors=[1.])
        if group['optimizer']=='muon':
            group['hyperparams']['momentum']=dict(kind='tied_step_nesterov',beta_after=beta,beta_before=.95,start_step=2009)
    for hook in cfg['post_optimizer']:
        if hook['name']=='validate_at_step_boundaries':hook['hyperparams']=dict(every=25,final_tenth_every=25,dense_window=[last+1,last+1],dense_every=1)
    return cfg
