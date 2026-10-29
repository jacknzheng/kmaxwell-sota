"""Solve additional already-captured targets for an original-run replay (no training, no new sidecars).
Usage: venv/bin/python solve_more.py 2725 2675 ...   (targets must be in plan captures and not historical/measured)"""
import os,sys,pathlib,json,subprocess,time,fcntl,hashlib,math
R=pathlib.Path(os.environ.get('GAP_ROOT',pathlib.Path(__file__).resolve().parent)).resolve();os.chdir(R);os.environ['GAP_ROOT']=str(R);sys.path[:0]=[str(R),str(R/'source')]
from common import write
P=json.load(open(R/'plan.json'));F=P['family'];SPEC=P[F];O=R/'blog_outputs'/P['run_id'];C=O/'captures'
targets=[int(t) for t in sys.argv[1:]];assert targets
hist=set(P['historical_done'][F]);assert not set(targets)&hist,'historical target'
assert set(targets)<=set(P['captures']),'target not captured'
assert (O/'CAPTURE_COMPLETE.json').exists()
import re
lock=open(R/'.blog_gpu.lock','a');fcntl.flock(lock,fcntl.LOCK_EX)   # same runtime-relative lock as the patched run_gaps.py
ENV={**os.environ,'PYTHONPATH':str(R/'source')+':'+str(R),'GAP_ROOT':str(R),'OMP_NUM_THREADS':'2','TORCHINDUCTOR_COMPILE_THREADS':'1','WANDB_MODE':'offline','PYTHONUNBUFFERED':'1','NCCL_CUMEM_ENABLE':'0','NCCL_CUMEM_HOST_ENABLE':'0','DENSE_PREFIX_WINDOW':'9','DENSE_INDEPENDENT':'0'}
TR=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=8'];remote=P['r2_destination']
flags=['--disable-http2','--multi-thread-streams','0','--transfers','2','--checkers','16']
def status(stage,**kw):write(O/'status.json',dict(stage=stage,time=time.time(),family=F,**kw))
def child(argv,log,env=ENV,cwd=R):
 with open(log,'ab') as f:
  p=subprocess.Popen(argv,stdout=f,stderr=subprocess.STDOUT,env=env,cwd=cwd);write(O/'active_process.json',dict(pid=p.pid,argv=argv));rc=p.wait()
 if rc:raise RuntimeError(f'Child exit{rc}: {log}')
def restore_capture(t):
 lk=open(C/f'.point_{t}.lock','a');fcntl.flock(lk,fcntl.LOCK_EX)
 pins=[json.loads(l) for l in (C/'index.jsonl').read_text().splitlines() if json.loads(l).get('kind')=='boundary' and json.loads(l).get('update')==t];assert len(pins)==8
 if not all((C/d['file']).exists() for d in pins):
  assert (C/f'R2_VERIFIED_{t}.json').exists();f=O/f'restore_{t}.txt';f.write_text('\n'.join(str((C/d['file']).relative_to(R)) for d in pins)+'\n');subprocess.run(['rclone','copy',remote,str(R),'--files-from',str(f),*flags],check=True)
 for d in pins:
  with (C/d['file']).open('rb') as fh:assert hashlib.file_digest(fh,'sha256').hexdigest()==d['sha256']
 return lk,pins
import torch
torch.set_num_threads(2)
import run_queue as q
from frozen_prediction import predict_kernel
q.ROOT=R;q.OUT=O/'results';q.OUT.mkdir(exist_ok=True);q.PLAN.update(P);q.ENV=ENV
def nearest_warm(t):
 done=[]
 for f in q.OUT.glob('point_*.json'):
  d=json.load(open(f));out=pathlib.Path(d['output'])
  if all((out/f'current_vector_rank{r}.pt').exists() for r in range(8)):done.append((abs(d['update']-t),out))
 for name in sorted((R/'warm').glob(f'{F}*')) if (R/'warm').exists() else []:
  m=re.search(r'(\d+)$',name.name)
  if m and all((name/f'current_vector_rank{r}.pt').exists() for r in range(8)):done.append((abs(int(m.group(1))-t),name))
 return min(done)[1] if done else None
for t in targets:
 result=q.OUT/f'point_{t:06d}.json'
 if result.exists():continue
 guard,pins=restore_capture(t);payload=torch.load(C/f'rank0/boundary_{t:06d}.pt',map_location='cpu',mmap=True,weights_only=False)
 settings=[g for s in payload['settings'] if s['optimizer']=='muon' for g in s['groups']];assert settings
 lr,wd=settings[0]['lr'],settings[0]['weight_decay'];pred=predict_kernel(payload['kernel_weights_used'].numpy(),learning_rate=lr,weight_decay=wd);q.PLAN['prediction']=pred['prediction'];del payload
 out=q.OUT/'measure'/f't{t:06d}_16m';out.mkdir(parents=True,exist_ok=True)
 write(out/'input_manifest.json',dict(pins={d['file']:d['sha256'] for d in pins},validation=P['validation_sha256'],lineage=P['lineage'],update=t,tokens=16777216))
 warm=nearest_warm(t)
 argv=TR+[str(R/'single_solve.py'),'--capture',str(C),'--validation',str(R/'data/fineweb10B/fineweb_val_000000.bin'),'--output',str(out),'--update',str(t),'--token-sizes','16777216','--dimensions','8,16,24,32,48,64,96,128','--microbatches','16','--fixed-probes','--hvp-mode','compile','--head-backend','cached','--matmul-precision','high','--gemm-precision','bf16_fp32']
 if warm:argv+=['--warm-start',str(warm)]
 status('measuring',update=t,warm=str(warm));child(argv,q.OUT/f'solve_{t}.log')
 progress=json.load(open(out/'progress.json'));contract=json.load(open(R/'production_contract.json'))
 differences={k:dict(actual=progress.get(k),expected=v) for k,v in contract.items() if progress.get(k)!=v};assert not differences,differences
 assert all(abs(v-wd)<1e-15 for v in progress['weight_decay_by_matrix'].values())
 row=q.read_result(out,'16m');row.update(run=F,lineage=P['lineage'],seed=0,learning_rate=lr,weight_decay=wd,qualified=row['dimension_qualified'],control_audit=pred,timestamp=t,kappa_star=pred['kappa_star'],crossing_angle=pred['crossing_angle'],kappa_measured=(row['zeta']-wd/524288)*lr*524288,semantic_contract_verified=True,source_hashes_match=True,public_view_visible=True,publication_role='primary',training_microbatch_sequences=64,global_batch_tokens=524288,warm_start=str(warm),adaptive_followup=True)
 write(result,row);guard.close();status('point_complete',update=t)
 P2=json.load(open(R/'plan.json'));P2.setdefault('adaptive_measurements',[]);P2['adaptive_measurements'].append(t);json.dump(P2,open(R/'plan.json','w'),indent=2)
status('adaptive_complete',targets=targets)
