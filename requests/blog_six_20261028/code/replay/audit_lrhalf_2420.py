"""Protocol audit for lrhalf 2420: one fresh Gaussian start at 16M (audit tier, DENSE_INDEPENDENT=1).
Diagnostic only; the published warm-start point is retained regardless of outcome."""
import json,os,pathlib,sys,time,fcntl,hashlib,subprocess
R=pathlib.Path('/workspace/pr323_blog');sys.path[:0]=[str(R),str(R/'source')];os.chdir(R)
from common import write
import run_queue as q
t=2420;Q=json.load(open(R/'queue_lrhalf.json'));OUT=R/'blog_outputs'/Q['run_id'];RES=OUT/'results';CAP=OUT/'captures'
LOG=R/'audit_lrhalf_2420.log';ST=R/'audit_lrhalf_2420_status.json'
def st(stage,**kw):write(ST,dict(stage=stage,time=time.time(),**kw))
def lr_idle():
 try:plan=json.load(open(R/'adaptive_lr_jobs.json'));state=json.load(open(R/'node_status.json'))
 except (OSError,ValueError):return False
 attempted={(x['queue'][6:-5],x['timestamp']) for x in state.get('arms',[]) if x.get('phase')=='adaptive_one'}
 return plan.get('stop') or (not [j for j in plan['jobs'] if (j['arm'],j['timestamp']) not in attempted] and state.get('stage')!='running_arm')
if (RES/f'audit_{t:06d}.json').exists():raise SystemExit('audit already present')
st('waiting_for_lr_queue_drain')
while not lr_idle():time.sleep(20)
lock=(R/'.blog_gpu.lock').open('a');st('waiting_for_gpu_lock');fcntl.flock(lock,fcntl.LOCK_EX)
def child(argv,log,**kw):
 with open(log,'ab')as f:
  p=subprocess.Popen(argv,stdout=f,stderr=subprocess.STDOUT,**kw);st('running',pid=p.pid);rc=p.wait()
 if rc:raise RuntimeError(f'Child rc{rc}; preserved {log}')
q.ROOT=R;q.OUT=RES;q.child=child;q.ENV.update(PYTHONPATH=str(R/'source')+':'+str(R),DENSE_PREFIX_WINDOW='9',NCCL_CUMEM_ENABLE='0',NCCL_CUMEM_HOST_ENABLE='0');q.PLAN.update(deadline=4102444800,analysis_microbatch=16,lineage=Q['run_id'],validation_sha256='5b95c8e0966f0861685b307b23dc5ae42b228ef74b28cb499784ae021f201640')
pins=[d for d in map(json.loads,(CAP/'index.jsonl').read_text().splitlines()) if d.get('kind')=='boundary' and d.get('update')==t];assert len(pins)==8
if not all((CAP/d['file']).exists() for d in pins):
 receipt=json.load(open(CAP/f'R2_VERIFIED_{t}.json'));assert receipt['update']==t
 listing=OUT/f'restore_audit_{t}.txt';listing.write_text('\n'.join(str((CAP/d['file']).relative_to(R)) for d in pins)+'\n')
 st('restoring_shards');subprocess.run(['rclone','copy',receipt['remote'],str(R),'--files-from',str(listing),'--transfers','8','--disable-http2'],check=True)
for d in pins:
 with (CAP/d['file']).open('rb')as f:assert hashlib.file_digest(f,'sha256').hexdigest()==d['sha256']
point=json.load(open(RES/f'point_{t:06d}.json'));q.CAPTURE=CAP;q.PLAN['prediction']=point['prediction']
st('measuring_independent_start');began=time.time()
row=q.measure(t,'16m_audit',pathlib.Path(point['output']))
row.update(arm='lrhalf',branch='lrhalf',seed=0,timestamp=t,boundary=t,run_id=Q['run_id'],lineage=Q['run_id'],publication_role='diagnostic',audit_of_point=dict(zeta=point['zeta'],dimension=point['dimension'],residual=point['residual'],output=point['output']),audit_relative_difference=row['zeta']/point['zeta']-1,wall_seconds=time.time()-began,reason='Protocol audit (plan token_protocol audit_reason: extreme adjacent-checkpoint jump): warm point +405% vs prediction, 5x neighbours 2240/2580, residual 8x neighbours; fresh Gaussian start 77193 at 16M. No veto; published point retained.')
write(RES/f'audit_{t:06d}.json',row);st('complete',zeta=row['zeta'],audit_relative_difference=row['audit_relative_difference'])
