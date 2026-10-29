"""Sparse, read-only captures around the unchanged native MaxEnt trainer."""
import os,pathlib,json,hashlib,sys
import torch
import torch.distributed as dist
from records.track_3_optimization.analysis.tuned_dense_capture import instrument,base_model
from records.track_3_optimization.harness.capture_store import CaptureWriter

class SparseRecorder:
 def __init__(self,model,optimizer,batch_tokens,start,stop):
  assert start == json.loads((pathlib.Path(os.environ['GAP_ROOT'])/'plan.json').read_text())['maxentslop']['anchor'] and dist.get_world_size()==8
  self.model=base_model(model);self.optimizer=optimizer;self.tokens=batch_tokens
  self.steps=set(json.loads(os.environ['GAP_CAPTURE_STEPS']));self.rank=dist.get_rank()
  assert all(start<=t<stop for t in self.steps)
  self.root=pathlib.Path(os.environ['DENSE_CAPTURE_ROOT']);self.writer=CaptureWriter(self.root,self.rank)
  self.names={id(p):n for n,p in self.model.named_parameters()}
  self.owned=[p for i,p in enumerate(optimizer.param_groups[0]['params']) if i%8==self.rank]
  if self.rank==0:
   (self.root/'capture_config.json').write_text(json.dumps(dict(windows=[[t,t] for t in sorted(self.steps)],anchors=[],ring_length=256,storage='sparse boundary captures; no raw streams')))
  dist.barrier()
  if self.rank==0:
   (pathlib.Path(os.environ['GAP_ROOT'])/'RESTORE_COMPLETE.json').write_text(json.dumps(dict(boundary=start,world_size=8,all_rank_optimizer_model_rng_restored=True)))
  dist.barrier()
 def begin(self,step):
  assert self.optimizer._step==step
  self.active=step in self.steps;self.momenta={};self.directions={};self.taps=None
  self.before={n:p.detach().cpu().clone() for n,p in self.model.named_parameters()} if self.active and self.rank==0 else None
  self.settings=[dict(pattern=r'^blocks\..*\.weight$',optimizer='muon',groups=[{k:float(g[k]) for k in ('lr','weight_decay')}]) for g in self.optimizer.param_groups]
 def observe(self,p,g,m,q,taps,step,group,polar):
  if not self.active:return
  n=self.names[id(p)];self.momenta[n]=m.detach().cpu().clone();self.directions[n]=q.detach().cpu().clone()
  self.taps=taps.detach().cpu().clone()
 def end(self,step):
  if self.active:
   assert len(self.momenta)==len(self.owned)
   payload=dict(update=step,rank=self.rank,format=2,post_insertion_prepolar_momentum=self.momenta,polar_output=self.directions,kernel_weights_used=self.taps,normalize_available=False,batch_receipts={'used':{'batch_tokens':self.tokens}},settings=self.settings)
   if self.rank==0:payload.update(model_before=self.before,buffers={n:b.detach().cpu().clone() for n,b in self.model.named_buffers()})
   r=self.writer.save('boundary',step,payload)
   rows=[None]*8;dist.all_gather_object(rows,dict(update=step,kind='boundary',file=f'rank{self.rank}/{r["file"]}',bytes=r['bytes'],sha256=r['sha256']))
   if self.rank==0:
    with (self.root/'index.jsonl').open('a') as f:
     for row in rows:f.write(json.dumps(row)+'\n')
  if self.rank==0:
   p=self.root/'training_progress.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(dict(update=step)));tmp.replace(p)
  self.before=None

source=pathlib.Path(os.environ['TUNED_TRAINER_SOURCE'])
assert hashlib.sha256(source.read_bytes()).hexdigest()==os.environ['TUNED_TRAINER_SHA256']
sys.path.insert(0,str(source.parent));sys.argv[0]=str(source)
exec(compile(instrument(source.read_text()),str(source),'exec'),dict(__name__='__main__',__file__=str(source),DenseRecorder=SparseRecorder))
