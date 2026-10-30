"""One immutable seed-0 intervention queue; production training/operator unchanged."""
import argparse,json,os,pathlib,subprocess,sys,time,fcntl,hashlib,shutil
import yaml
P=argparse.ArgumentParser();P.add_argument('--queue',required=True);P.add_argument('--phase',choices=['broad','all','capture','adaptive_one'],default='all');P.add_argument('--timestamp',type=int);P.add_argument('--runtime',default='/workspace/pr323_blog');a=P.parse_args()
Q=json.load(open(a.queue));R=pathlib.Path(a.runtime).resolve();sys.path.insert(0,str(R));sys.path.insert(0,str(R/'source'));os.chdir(R)
from common import write
import run_queue as q
OUT=R/'blog_outputs'/Q['run_id'];OUT.mkdir(parents=True,exist_ok=True)
lock=(R/'.blog_gpu.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
def status(stage,**kw):write(OUT/'status.json',dict(stage=stage,time=time.time(),arm=Q['arm'],**kw))
def child(argv,log,**kw):
 with open(log,'ab')as f:
  p=subprocess.Popen(argv,stdout=f,stderr=subprocess.STDOUT,**kw);status('running',pid=p.pid,argv=argv);rc=p.wait()
 if rc:raise RuntimeError(f'Child rc{rc}; preserved {log}')
q.ENV['BLOG_CAPTURE_RESERVOIR_BYTES']=str(min(8_000_000_000_000,max(28_000_000_000,shutil.disk_usage(R).total-180_000_000_000)))
q.ROOT=R;q.OUT=OUT/'results';q.OUT.mkdir(exist_ok=True);q.ENV.update(PYTHONPATH=str(R/'source')+':'+str(R),DENSE_PREFIX_WINDOW='9',DENSE_INDEPENDENT='0',NCCL_CUMEM_ENABLE='0',NCCL_CUMEM_HOST_ENABLE='0');q.PLAN.update(deadline=4102444800,analysis_microbatch=int(os.environ.get("BLOG_MICROBATCH","16")),lineage=Q['run_id'],validation_sha256='5b95c8e0966f0861685b307b23dc5ae42b228ef74b28cb499784ae021f201640');q.child=child
parent=R/'restart/shared_2009'
if not Q.get('analysis_only') and not (OUT/'CAPTURE_COMPLETE.json').exists():
 receipt=json.load(open(parent/'COMPLETE.json'));assert receipt['boundary']==2009 and receipt['world_size']==8
 for rank in range(8):assert(parent/f'rank{rank}.pt').stat().st_size==11908782730
warm=R/Q.get('warm_path','warm/shared_2009');capture=OUT/'captures';done=OUT/'CAPTURE_COMPLETE.json'
if Q.get('analysis_only'):assert done.exists(),'Analysis-only requires original completed capture receipt'
if not done.exists():
 if (capture/'index.jsonl').exists():raise RuntimeError('Partial capture preserved; explicit resume/attempt directory required')
 if Q['family']=='momentum':
  from momentum_config import build
  cfg=build(R,Q['arm'],parent,Q['last_capture'],[[t,t]for t in Q['capture_updates']],Q.get('final_full_restart',False));trainer='momentum_train.py'
 else:
  from extension_config import build
  cfg=build(R,parent,OUT,Q['last_capture'],Q['lr_factor'],2010);cfg['capture']['windows']=[[t,t]for t in Q['capture_updates']];cfg['checkpoint_every']=999999;trainer='continuous20_train.py'
 cfg.update(microbatch_sequences=Q.get('training_microbatch',int(os.environ.get('BLOG_TRAIN_MICROBATCH','8'))),run_id=Q['run_id'],output_dir=str(OUT/'training'),checkpoint_dir=str(OUT/'restart'),train_data_seed=0)
 cfg['capture']['root']=str(capture)
 if Q.get('final_full_restart'):cfg['checkpoint_boundaries']=[Q['last_capture']+1]
 cfg['pre_forward']=[h for h in cfg.get('pre_forward',[])if h.get('name')!='capture_space']+[dict(name='capture_space')]
 q.ENV['BLOG_FAMILY']=Q['family'];trainer='bounded_train.py'
 (OUT/'training.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
 child(q.TORCHRUN+[str(R/trainer),str(OUT/'training.yaml')],OUT/'training.log',cwd=R,env=q.ENV)
 counts={}
 for line in (capture/'index.jsonl').read_text().splitlines():
  d=json.loads(line)
  if d.get('kind')=='boundary':counts.setdefault(d['update'],set()).add(d['file'])
 assert all(len(counts.get(t,[]))==8 for t in Q['capture_updates'])
 write(done,dict(run_id=Q['run_id'],source_boundary=2009,captured=Q['capture_updates'],cohort='fresh_replay; overlap comparison pending; not pooled with historical observations'))
# Preserve ten overlap captures for CPU tensor/digest comparison; no bitwise scientific veto.
refs=json.load(open(R/f"overlap_{Q['arm']}.json"));actual={}
for line in (capture/'index.jsonl').read_text().splitlines():
 d=json.loads(line)
 if d.get('kind')=='boundary':actual[(d['update'],d['file'])]=d['sha256']
comparisons=[]
for ref in refs:
 pins=ref['pins'];comparisons.append(dict(reference=ref['index_origin'],compared=len(pins),identical_files=sum(actual.get((x['update'],x['file']))==x['sha256'] for x in pins),interpretation='File equality certifies exact archivedbytes; inequality alone does not quantify tensor differences. Keep newrealization separate; no measurement veto.'))
write(OUT/'OVERLAP_COMPARISON.json',dict(comparisons=comparisons,strict_veto=False,default_publication='distinct_replay_realization'))
write(OUT/'COHORT_POLICY.json',dict(overlap=list(range(2010,2020)),policy=Q['cohort_policy']))
if Q.get('compare_endpoint') and not (OUT/'OVERLAP_COMPARISON_TENSORS_2039.json').exists():
 child([sys.executable,str(R/'compare_endpoint.py'),'--runtime',str(R),'--arm',Q['arm'],'--output',str(OUT)],OUT/'endpoint_comparison.log')
if a.phase=='adaptive_one':
 assert a.timestamp in set(Q['capture_updates'])|{j['timestamp']for j in Q['saved_measurements']},'Selectedpoint not in savedcapturebank; explicit capturegeneration required'
 prior=[]
 for f in q.OUT.glob('point_*.json'):
  d=json.load(open(f))
  if d.get('run_id')==Q['run_id']:prior.append(d)
 if prior:warm=pathlib.Path(min(prior,key=lambda d:abs(d['update']-a.timestamp))['output'])
if a.phase=='capture':
 status('capture_complete_waiting_adaptive_jobs');sys.exit(0)
for job in [{'timestamp':t,'capture_root':str(capture),'lineage':Q['run_id']}for t in (sorted(set(Q['replay_measurements'])|set(Q['capture_updates'])) if a.phase=='adaptive_one' else Q['replay_measurements'])]+[dict(j,capture_root=str(R/'saved_captures'/f"{Q['arm']}_{j['timestamp']}"),lineage=j['source_lineage'])for j in Q['saved_measurements']]:
 t=job['timestamp']
 if a.phase=='adaptive_one' and t!=a.timestamp:continue
 if a.phase=='broad' and Q['family']=='lr' and t%100 and t!=Q['plot_window'][1] and t not in Q.get('priority_burst',[]):continue
 result=q.OUT/f'point_{t:06d}.json'
 if result.exists():warm=pathlib.Path(json.load(open(result))['output']);continue
 q.CAPTURE=pathlib.Path(job['capture_root']);q.PLAN['lineage']=job['lineage'];q.PLAN['prediction']=None
 while shutil.disk_usage(R).free < 20_000_000_000:
  status('waiting_for_verified_archive_space',timestamp=t,free_bytes=shutil.disk_usage(R).free,reserve_bytes=20000000000);time.sleep(5)
 point_lock=(q.CAPTURE/f'.point_{t}.lock').open('a');fcntl.flock(point_lock,fcntl.LOCK_EX)
 pins=[]
 for line in (q.CAPTURE/'index.jsonl').read_text().splitlines():
  d=json.loads(line)
  if d.get('kind')=='boundary' and d.get('update')==t:pins.append(d)
 assert len(pins)==8
 if not all((q.CAPTURE/d['file']).exists()for d in pins):
  receipt=json.load(open(q.CAPTURE/f'R2_VERIFIED_{t}.json'));assert receipt['update']==t
  listing=OUT/f'restore_{t}.txt';listing.write_text('\n'.join(str((q.CAPTURE/d['file']).relative_to(R))for d in pins)+'\n')
  subprocess.run(['rclone','copy',receipt['remote'],str(R),'--files-from',str(listing),'--transfers','8','--disable-http2'],check=True)
 for d in pins:
  with (q.CAPTURE/d['file']).open('rb')as f:assert hashlib.file_digest(f,'sha256').hexdigest()==d['sha256']
 status('measuring',timestamp=t)
 import torch
 from frozen_prediction import predict_kernel
 weights=None
 for rank in range(8):
  payload=torch.load(q.CAPTURE/f'rank{rank}/boundary_{t:06d}.pt',map_location='cpu',weights_only=False,mmap=True)
  w=payload['kernel_weights_used'].detach().cpu()
  if weights is None:weights=w.clone()
  else:assert torch.equal(weights,w),'Rank kernel mismatch'
  del payload
 lr=.025*(Q['lr_factor'] or 1.)
 native=predict_kernel(weights.numpy(),learning_rate=lr,weight_decay=.05)
 audit={str(t):dict(native,kernel_sha256=hashlib.sha256(weights.numpy().tobytes()).hexdigest(),all8_kernel_equal=True,rate_source='original2009 branch config; all-group LRfactor atstep2009')}
 write(q.CAPTURE/f'control_audit_{t}.json',audit[str(t)])
 q.PLAN['prediction']=audit[str(t)]['prediction'];row=q.measure(t,'16m',warm);row.update(arm=Q['arm'],branch=Q['arm'],seed=0,timestamp=t,boundary=t,tau=t-2009,learning_rate=lr,weight_decay=.05,run_id=job['lineage'],lineage=job['lineage'],source_boundary=2009,control_audit=audit[str(t)])
 
 if Q['family']=='momentum':
  from momentum_schedule import TiedStepNesterov
  beta=.9750961447125188 if Q['arm']=='higher_zeta' else .8990666066208468
  stationary=predict_kernel(TiedStepNesterov(beta)(10000).weights.numpy(),learning_rate=.025,weight_decay=.05)
  row.update(beta=beta,stationary_prediction=stationary['prediction'],frozen_kernel_prediction=native['prediction'],prediction_convention='actual frozen-kernel first contact; stationary prediction separately recorded')
 write(result,row);warm=pathlib.Path(row['output']);status('point_complete',timestamp=t);point_lock.close()
status('broad_complete' if a.phase=='broad' else 'complete')
