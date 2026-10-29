"""Only storage backpressure changes; production optimizer/forward/checkpoint unchanged."""
import os,time,pathlib,shutil,json
import torch
import torch.distributed as dist
if os.environ['BLOG_FAMILY']=='momentum':
 import momentum_train
from records.track_3_optimization.harness import hooks
from records.track_3_optimization import run
if os.environ['BLOG_FAMILY']=='lr':
 # Same capture-only checkpoint policy as continuous20_train: no implicit stop-step full save.
 def checkpoint_state():
  def hook(config,state):return state
  return hook
 hooks.checkpoint_state=checkpoint_state
old=hooks.hook_factories

def bounded_capture_space():
 def hook(config,state):
  t=state['step']
  if not any(lo<=t<=hi for lo,hi in config['capture']['windows']):return state
  root=pathlib.Path(config['capture']['root'])
  while True:
   free=shutil.disk_usage(root.parent).free if dist.get_rank()==0 else 0
   used=sum(x.stat().st_size for x in root.rglob('*.pt')) if dist.get_rank()==0 and root.exists()else 0
   okay=torch.tensor([int(free>=int(os.environ.get('BLOG_MIN_FREE_BYTES','80000000000')) and used<=int(os.environ.get('BLOG_CAPTURE_RESERVOIR_BYTES','28000000000')))],device='cuda',dtype=torch.int32);dist.broadcast(okay,0)
   if okay.item():return state
   if dist.get_rank()==0:(root.parent/'disk_backpressure.json').write_text(json.dumps(dict(step=t,free_bytes=free,capture_bytes=used,required_free=80000000000,action='wait for R2-verified capture archive; preserve in-memory optimizer state')))
   time.sleep(5)
 return hook

def factories():
 d=old();d['capture_space']=bounded_capture_space;return d
hooks.hook_factories=factories
run.main()
