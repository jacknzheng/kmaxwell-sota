import torch,json,pathlib,hashlib
R=pathlib.Path('/workspace/pr323_blog/lr_overlap');torch.set_num_threads(2);rows=[]
for arm in ['lr2x','lrhalf']:
 for t in [2010,2019]:
  old=torch.load(R/'old'/arm/f'rank0/boundary_{t:06d}.pt',map_location='cpu',weights_only=False,mmap=True);new=torch.load(R/'new'/arm/f'boundary_{t:06d}.pt',map_location='cpu',weights_only=False,mmap=True)
  a,b=old['kernel_weights_used'],new['kernel_weights_used'];equal=torch.equal(a,b);assert equal
  row=dict(arm=arm,update=t,kernel_equal=equal,kernel_sha256=hashlib.sha256(a.contiguous().numpy().tobytes()).hexdigest(),maximum_difference=float((a-b).abs().max()))
  rows.append(row)
  p=R/f'{arm}_{t}_comparison.json';d=json.load(open(p))
  for side,k in [('original',a),('oct27',b)]:d['metadata'][side]['kernel_weights_used']['value']=k.tolist()
  p.write_text(json.dumps(d,indent=2))
(R/'KERNEL_EXACT_EQUALITY.json').write_text(json.dumps(rows,indent=2));print(json.dumps(rows))
