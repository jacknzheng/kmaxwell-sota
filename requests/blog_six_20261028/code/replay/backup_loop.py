"""DirectR2backup: frozen small-file snapshots; only completed solver tensors."""
import argparse,json,pathlib,subprocess,time,os
p=argparse.ArgumentParser();p.add_argument('--runtime',default='/workspace/pr323_blog');p.add_argument('--remote',required=True);a=p.parse_args();R=pathlib.Path(a.runtime);S=R/'.backup_snapshot';S.mkdir(exist_ok=True)
flags=['--transfers','8','--checkers','16','--disable-http2','--s3-upload-cutoff','8Mi','--s3-chunk-size','8Mi','--s3-upload-concurrency','32','--contimeout','15s','--timeout','60s','--stats','30s']
while True:
 small=[];tensors=[]
 paths=[x for x in R.iterdir()if x.is_file()and not x.name.startswith('.')and x.suffix in ('.py','.json','.yaml','.tgz')and x.name!='download_manifest.json']+list((R/'blog_outputs').rglob('*'))
 for x in paths:
  if not x.is_file()or x.is_symlink()or any(part.startswith('.')for part in x.relative_to(R).parts)or 'wandb'in x.parts:continue
  rel=x.relative_to(R).as_posix()
  if x.suffix=='.pt':
   if 'captures'in x.parts:continue
   progress=x.parent/'progress.json'
   try:complete=json.load(open(progress)).get('stage')=='fixed_probes_complete'
   except (OSError,ValueError):complete=False
   if complete:tensors.append(rel)
   continue
  dest=S/rel;dest.parent.mkdir(parents=True,exist_ok=True)
  try:
   size=x.stat().st_size
   with x.open('rb')as f:data=f.read(size)
  except OSError:continue
  dest.write_bytes(data);small.append(rel)
 codes=[]
 for base,files,tag in [(S,small,'small'),(R,tensors,'tensors')]:
  if not files:continue
  listing=R/f'backup_{tag}_files.txt';listing.write_text('\n'.join(files)+'\n')
  try:rc=subprocess.run(['rclone','copy',str(base),a.remote,'--files-from',str(listing),*flags]).returncode
  except OSError as e:rc=127
  codes.append(rc)
 (R/'backup_status.json').write_text(json.dumps(dict(time=time.time(),returncode=max(codes,default=0),small_files=len(small),completed_solver_tensors=len(tensors),mutable_files_snapshotted=True)))
 time.sleep(20)
