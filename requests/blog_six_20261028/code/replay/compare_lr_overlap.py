"""CPU-only original/October27 LR state comparison; no new zeta measurements."""
import concurrent.futures, hashlib, json, math, pathlib, subprocess, time
import torch
R=pathlib.Path('/workspace/pr323_blog');D=R/'lr_overlap';torch.set_num_threads(2)
remote='muoff-r2:muoff-research/runs/blog_six_figures_20261027/node_57963909'
def fetch(pair):
 arm,t=pair;folder=D/'new'/arm;folder.mkdir(parents=True,exist_ok=True)
 base=f'blog_outputs/blog_seed0_{arm}_replay2009_20261027/captures'
 for name,dest in [(f'rank0/boundary_{t:06d}.pt',folder/f'boundary_{t:06d}.pt'),(f'R2_VERIFIED_{t}.json',folder/f'R2_VERIFIED_{t}.json')]:
  subprocess.run(['rclone','copyto',remote+'/'+base+'/'+name,str(dest),'--disable-http2'],check=True)
pairs=[(arm,t) for arm in ['lr2x','lrhalf'] for t in [2010,2019]]
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(fetch,pairs))
while not (D/'DIRECT_DOWNLOAD_COMPLETE').exists():time.sleep(5)
def scalars(obj,depth=0):
 if isinstance(obj,(str,int,float,bool)) or obj is None:return obj
 if torch.is_tensor(obj):return dict(tensor_shape=list(obj.shape),dtype=str(obj.dtype),value=obj.tolist() if obj.numel()<=512 else None)
 if depth>6:return str(type(obj))
 if isinstance(obj,dict):return {str(k):scalars(v,depth+1) for k,v in obj.items()}
 if isinstance(obj,(list,tuple)):return [scalars(x,depth+1) for x in obj] if len(obj)<40 else dict(length=len(obj))
 return str(type(obj))
reports=[]
for arm,t in pairs:
 op=D/'old'/arm/f'rank0/boundary_{t:06d}.pt';np=D/'new'/arm/f'boundary_{t:06d}.pt'
 pin=next(x for x in map(json.loads,(D/'old'/arm/'index.jsonl').read_text().splitlines()) if x.get('kind')=='boundary' and x.get('update')==t and x['file'].startswith('rank0/'))
 newpin=next(x for x in json.loads((D/'new'/arm/f'R2_VERIFIED_{t}.json').read_text())['files'] if '/rank0/' in x['file'])
 for path,p in [(op,pin),(np,newpin)]:
  with path.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==p['sha256']
 old=torch.load(op,map_location='cpu',weights_only=False,mmap=True);new=torch.load(np,map_location='cpu',weights_only=False,mmap=True)
 comparisons={}
 tensor_keys=['model_before','post_insertion_prepolar_momentum','polar_output','executed_update']
 for key in tensor_keys:
  if key not in old and key not in new:continue
  x,y=old[key],new[key];assert set(x)==set(y)
  n=dn=mx=identical=0
  for name in x:
   u,v=x[name].double(),y[name].double();delta=v-u;n+=float(u.square().sum());dn+=float(delta.square().sum());mx=max(mx,float(delta.abs().max()));identical+=int(torch.equal(x[name],y[name]))
  comparisons[key]=dict(relative_l2=math.sqrt(dn/n) if n else None,difference_l2=math.sqrt(dn),max_abs=mx,identical_tensors=identical,tensors=len(x))
 metadata={label:{k:scalars(v) for k,v in obj.items() if k not in tensor_keys} for label,obj in [('original',old),('oct27',new)]}
 row=dict(arm=arm,update=t,comparisons=comparisons,metadata=metadata,original_source=f'checkpoints/critroll/pr323_lr_snap_portable_20261019/continuous20/{arm}/captures',oct27_source=remote,scope='Global model and rank0-owned Muon actions',strict_veto=False)
 (D/f'{arm}_{t}_comparison.json').write_text(json.dumps(row,indent=2));reports.append({k:v for k,v in row.items() if k!='metadata'})
 print(json.dumps(reports[-1]),flush=True)
 del old,new
(D/'SUMMARY.json').write_text(json.dumps(dict(time=time.time(),comparisons=reports),indent=2))
subprocess.run(['rclone','copy',str(D),'muoff-r2:muoff-research/runs/blog_six_20261027/lr_overlap_audit','--include','*_comparison.json','--include','SUMMARY.json','--disable-http2'],check=True)
