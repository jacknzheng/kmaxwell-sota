"""Stage archived receipts, then atomically append root-approved broad gap singles."""
import json,pathlib,subprocess,time
R=pathlib.Path('/workspace/pr323_blog');jobs=[('lr2x',2540),('lrhalf',2320),('lr2x',2620),('lrhalf',2380),('lr2x',2700),('lrhalf',2660),('lrhalf',2700)]
files=[f'blog_outputs/blog_seed0_{arm}_replay2009_20261027/captures/R2_VERIFIED_{t}.json'for arm,t in jobs]
p=R/'followup_receipt_restore.txt';p.write_text('\n'.join(files)+'\n')
subprocess.run(['rclone','copy','muoff-r2:muoff-research/runs/blog_six_figures_20261027/node_57963909',str(R),'--files-from',str(p),'--transfers','8','--disable-http2'],check=True)
for rel,(_,t) in zip(files,jobs):assert json.load(open(R/rel))['update']==t
p=R/'adaptive_lr_jobs.json';d=json.load(open(p));existing={(j['arm'],j['timestamp']) for j in d['jobs']}
for arm,t in jobs:
 if (arm,t)not in existing:d['jobs'].append(dict(arm=arm,timestamp=t))
d.update(stop=False,fallback_allowed=False,policy='Root-approved broad gap coverage; reassess actual results before further choices; wait and notify when queue empty',last_authorized_change=dict(time=time.time(),note='Root approves7additional singles after initial8; noLR2x>2740.'))
temp=p.with_suffix('.next');temp.write_text(json.dumps(d,indent=2));temp.replace(p)
print(json.dumps(dict(jobs=len(d['jobs']),added=jobs)))
