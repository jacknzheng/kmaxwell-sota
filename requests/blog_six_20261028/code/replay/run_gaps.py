"""Eight-GPU native replay, immutable capture archive, genuinely new 16M solves."""
import os,sys,pathlib,json,subprocess,time,threading,fcntl,hashlib
import yaml
R=pathlib.Path(os.environ.get('GAP_ROOT',pathlib.Path(__file__).resolve().parent)).resolve();os.chdir(R);sys.path[:0]=[str(R),str(R/'source')];os.environ['GAP_ROOT']=str(R)
ARCHIVE=os.environ.get('GAP_ARCHIVE','0')=='1'  # 1 only if an rclone remote is configured; otherwise everything stays local
from common import write
P=json.load(open(R/'plan.json'));F=P['family'];SPEC=P[F];O=R/'blog_outputs'/P['run_id'];O.mkdir(parents=True,exist_ok=True);C=O/'captures'
assert not set(P['new_measurements'])&set(P['historical_done'][F])
lock=open(R/'.blog_gpu.lock','a');fcntl.flock(lock,fcntl.LOCK_EX)
ENV={**os.environ,'PYTHONPATH':str(R/'source')+':'+str(R),'GAP_ROOT':str(R),'OMP_NUM_THREADS':'2','TORCHINDUCTOR_COMPILE_THREADS':'1','WANDB_MODE':'offline','PYTHONUNBUFFERED':'1','NCCL_CUMEM_ENABLE':'0','NCCL_CUMEM_HOST_ENABLE':'0','DENSE_PREFIX_WINDOW':'9','DENSE_INDEPENDENT':'0'}
TR=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=8']
remote=P['r2_destination'];flags=['--disable-http2','--multi-thread-streams','0','--transfers','2','--checkers','16','--s3-upload-cutoff','64Mi','--s3-chunk-size','64Mi','--s3-upload-concurrency','4']
def status(stage,**kw):write(O/'status.json',dict(stage=stage,time=time.time(),family=F,**kw))
def child(argv,log,env=ENV,cwd=R):
 with open(log,'ab') as f:
  p=subprocess.Popen(argv,stdout=f,stderr=subprocess.STDOUT,env=env,cwd=cwd);write(O/'active_process.json',dict(pid=p.pid,argv=argv));rc=p.wait()
 if rc:raise RuntimeError(f'Child exit{rc}: {log}')
assert (R/'DIRECT_DOWNLOAD_COMPLETE').exists()
for name in (['archive_captures.py','archive_gradients.py','backup_loop.py'] if ARCHIVE else []):
 subprocess.Popen([sys.executable,str(R/name),'--runtime',str(R),'--remote',remote],env=ENV,stdout=open(R/(name+'.log'),'ab'),stderr=subprocess.STDOUT)
def evict_inputs():
 while not (R/'RESTORE_COMPLETE.json').exists():time.sleep(2)
 proof=json.load(open(R/'RESTORE_COMPLETE.json'));assert proof['all_rank_optimizer_model_rng_restored']
 removed=[]
 for p in (R/'input').rglob('*.pt'):
  removed.append(dict(file=str(p),bytes=p.stat().st_size));p.unlink()
 write(R/'INPUT_CACHE_RELEASED.json',dict(files=removed,time=time.time(),immutable_source=SPEC['source']))
if not (O/'CAPTURE_COMPLETE.json').exists():
 import math
 need=math.ceil(SPEC['stop']*524288/100_000_000)+1;have=len(list((R/'data/fineweb10B').glob('fineweb_train_*.bin')))
 assert have>=need,f'need {need} fineweb train shards for stop {SPEC["stop"]} (one global batch of 524288 tokens per step, ~100M tokens per shard); found {have}'
 if os.environ.get('GAP_EVICT_INPUTS','0')=='1':threading.Thread(target=evict_inputs,daemon=True).start()
 status('training',start=SPEC['anchor'],stop=SPEC['stop'])
 if F=='pr323':
  cfg=yaml.safe_load((R/'original_pr323.yaml').read_text())
  cfg.update(run_id=P['run_id'],output_dir=str(O/'training'),checkpoint_dir=str(O/'restart'),checkpoint_every=SPEC['stop'],stop_step=SPEC['stop'],start_step=SPEC['anchor'],resume_checkpoint=None,resume_anchor=str(R/'input/pr323/rank0/anchor_000975.pt'),microbatch_sequences=64,train_data='data/fineweb10B/fineweb_train_*.bin',val_data='data/fineweb10B/fineweb_val_*.bin',capture=dict(root=str(C),windows=[[t,t] for t in P['captures']],anchors=[],window_gradient_only=True))
  assert cfg['train_steps']==3250 and all(g['lr_schedule']==dict(kind='cooldown',fraction=.7) for g in cfg['optimizer_groups'])
  (O/'training.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False));child(TR+[str(R/'pr_streaming_train.py'),str(O/'training.yaml')],O/'training.log')
 else:
  native={**ENV,'DENSE_CAPTURE_ROOT':str(C),'GAP_CAPTURE_STEPS':json.dumps(P['captures']),'TUNED_TRAINER_SOURCE':str(R/'train_gpt_muon_maxentslop.py'),'TUNED_TRAINER_SHA256':SPEC['trainer_sha256'],'STOP_STEP':str(SPEC['stop']),'CHECKPOINT_ROOT':str(O/'restart'),'CHECKPOINT_STEPS':str(SPEC['stop'])}
  if SPEC['anchor']>0: native['RESUME_CHECKPOINT']=str(R/'input/maxentslop')
  else: native.pop('RESUME_CHECKPOINT',None)  # scratch: anchor 0 -> trainer starts from step 0 (seed 0)
  for k in ['WAYPOINT_SCHEDULE_PATH','CONSTRAINED_SCHEDULE_PATH','TRIAL_BATCH_PATH','COMPARE_CHECKPOINT']:native.pop(k,None)
  child(TR+[str(R/'maxent_sparse_capture.py'),'--seed','0','--train_steps',str(SPEC['train_steps'])],O/'training.log',env=native)
 write(O/'CAPTURE_COMPLETE.json',dict(captures=P['captures'],lineage=P['lineage']))
full=O/'restart'/(f"boundary_{SPEC['stop']}" if F=='pr323' else f"step_{SPEC['stop']:05d}");assert (full/('COMPLETE.json' if F=='pr323' else 'manifest.json')).exists()
def archive_full():
 dest=remote+'/'+str(full.relative_to(R))
 while True:
  try:
   subprocess.run(['rclone','copy',str(full),dest,'--size-only','--disable-http2','--multi-thread-streams','0','--transfers','2','--s3-chunk-size','64Mi','--s3-upload-concurrency','4'],check=True,stdout=open(O/'restart_archive.log','ab'),stderr=subprocess.STDOUT)
   sizes={x['Path']:x['Size'] for x in json.loads(subprocess.check_output(['rclone','lsjson',dest,'--disable-http2']))}
   assert all(sizes.get(p.name)==p.stat().st_size for p in full.iterdir() if p.is_file())
   write(O/'FULL_RESTART_R2_VERIFIED.json',dict(remote=dest,files=sizes,time=time.time()));return
  except Exception as e:write(O/'archive_error.json',dict(error=str(e)));time.sleep(15)
archive=threading.Thread(target=archive_full,daemon=True)
if ARCHIVE:archive.start()
def restore_capture(t):
 lock=open(C/f'.point_{t}.lock','a');fcntl.flock(lock,fcntl.LOCK_EX)
 pins=[json.loads(l) for l in (C/'index.jsonl').read_text().splitlines() if json.loads(l).get('kind')=='boundary' and json.loads(l).get('update')==t];assert len(pins)==8
 if not all((C/d['file']).exists() for d in pins):
  assert (C/f'R2_VERIFIED_{t}.json').exists();f=O/f'restore_{t}.txt';f.write_text('\n'.join(str((C/d['file']).relative_to(R)) for d in pins)+'\n');subprocess.run(['rclone','copy',remote,str(R),'--files-from',str(f),*flags],check=True)
 return lock,pins
import torch,math
torch.set_num_threads(2)
overlap=None;comparisons={}
if 1250 in P['captures']:overlap,pins=restore_capture(1250)
for rank in (range(8) if overlap else []):
 a=torch.load(R/f'reference1250/rank{rank}/boundary_001250.pt',map_location='cpu',mmap=True,weights_only=False);b=torch.load(C/f'rank{rank}/boundary_001250.pt',map_location='cpu',mmap=True,weights_only=False)
 for key in ['model_before','post_insertion_prepolar_momentum','polar_output']:
  if key not in a:continue
  assert set(a[key])==set(b[key]);n=dn=0.
  for name,x in a[key].items():
   x=x.double();y=b[key][name].double();n+=float(x.square().sum());dn+=float((x-y).square().sum())
  comparisons[f'rank{rank}/{key}']=dict(relative_l2=math.sqrt(dn/n) if n else None)
 if rank==0:
  def muon(x):return [(g['lr'],g['weight_decay']) for s in x['settings'] if s['optimizer']=='muon' for g in s['groups']]
  assert muon(a)==muon(b)
 del a,b
write(O/'OVERLAP_1250.json',dict(comparisons=comparisons,compatible_controls=True,strict_veto=False,lineage=P['lineage'],skipped=(overlap is None)));overlap and overlap.close()
import run_queue as q
from frozen_prediction import predict_kernel
q.ROOT=R;q.OUT=O/'results';q.OUT.mkdir(exist_ok=True);q.PLAN.update(P);q.ENV=ENV
import re as _re
warm=None
for cand in sorted((R/'warm').glob(f'{F}*'),key=lambda d:abs(int(_re.search(r'(\d+)$',d.name).group(1))-P['new_measurements'][0])) if (R/'warm').exists() else []:
 if all((cand/f'current_vector_rank{r}.pt').exists() for r in range(8)):warm=cand;break
for t in P['new_measurements']:
 result=q.OUT/f'point_{t:06d}.json'
 if result.exists():warm=pathlib.Path(json.load(open(result))['output']);continue
 guard,pins=restore_capture(t);payload=torch.load(C/f'rank0/boundary_{t:06d}.pt',map_location='cpu',mmap=True,weights_only=False)
 settings=[g for s in payload['settings'] if s['optimizer']=='muon' for g in s['groups']];assert settings
 lr,wd=settings[0]['lr'],settings[0]['weight_decay'];pred=predict_kernel(payload['kernel_weights_used'].numpy(),learning_rate=lr,weight_decay=wd);q.PLAN['prediction']=pred['prediction'];del payload
 out=q.OUT/'measure'/f't{t:06d}_16m';out.mkdir(parents=True,exist_ok=True)
 write(out/'input_manifest.json',dict(pins={d['file']:d['sha256'] for d in pins},validation=P['validation_sha256'],lineage=P['lineage'],update=t,tokens=16777216))
 argv=TR+[str(R/'single_solve.py'),'--capture',str(C),'--validation',str(R/'data/fineweb10B/fineweb_val_000000.bin'),'--output',str(out),'--update',str(t),'--token-sizes','16777216','--dimensions','8,16,24,32,48,64,96,128','--microbatches','16','--fixed-probes','--hvp-mode','compile','--head-backend','cached','--matmul-precision','high','--gemm-precision','bf16_fp32']
 if warm:argv+=['--warm-start',str(warm)]
 status('measuring',update=t);child(argv,q.OUT/f'solve_{t}.log')
 progress=json.load(open(out/'progress.json'));contract=json.load(open(R/'production_contract.json'))
 differences={k:dict(actual=progress.get(k),expected=v) for k,v in contract.items() if progress.get(k)!=v};assert not differences,differences
 assert all(abs(v-wd)<1e-15 for v in progress['weight_decay_by_matrix'].values())
 row=q.read_result(out,'16m');row.update(run=F,lineage=P['lineage'],seed=0,learning_rate=lr,weight_decay=wd,qualified=row['dimension_qualified'],control_audit=pred,timestamp=t,kappa_star=pred['kappa_star'],crossing_angle=pred['crossing_angle'],kappa_measured=(row['zeta']-wd/524288)*lr*524288,semantic_contract_verified=True,source_hashes_match=True,public_view_visible=True,publication_role='primary',training_microbatch_sequences=64,global_batch_tokens=524288)
 write(result,row);warm=out;guard.close();status('point_complete',update=t)
status('science_complete_archiving');lock.close()
if ARCHIVE:archive.join()
status('first_wave_complete',completed=len(P['new_measurements']),further_gap_selection_required=True)
