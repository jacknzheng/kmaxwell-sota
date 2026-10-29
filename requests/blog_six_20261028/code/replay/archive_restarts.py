"""Preserve closed full restarts in R2 and reclaim verified local shards."""
import argparse, hashlib, json, pathlib, subprocess, time
p=argparse.ArgumentParser();p.add_argument('--runtime',required=True);p.add_argument('--remote',required=True);a=p.parse_args();R=pathlib.Path(a.runtime)
while True:
 for receipt in (R/'blog_outputs').glob('*/restart/boundary_*/COMPLETE.json'):
  folder=receipt.parent;proof=folder/'R2_VERIFIED.json'
  if proof.exists():continue
  try:
   metadata=json.loads(receipt.read_text());assert metadata['world_size']==8
   paths=[folder/f'rank{i}.pt' for i in range(8)]+[receipt]
   pins=[]
   for path in paths:
    with path.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
    pins.append(dict(file=str(path.relative_to(R)),bytes=path.stat().st_size,sha256=digest))
   listing=folder/'.archive_files.txt';listing.write_text('\n'.join(x['file'] for x in pins)+'\n')
   subprocess.run(['rclone','copy',str(R),a.remote,'--files-from',str(listing),'--transfers','8','--disable-http2','--s3-chunk-size','64Mi','--s3-upload-concurrency','4'],check=True)
   for pin in pins:
    remote=json.loads(subprocess.check_output(['rclone','lsjson',a.remote+'/'+pin['file'],'--stat','--disable-http2']))
    assert remote['Size']==pin['bytes']
   proof.write_text(json.dumps(dict(time=time.time(),remote=a.remote,files=pins,source_sha256_recorded=True),indent=2))
   subprocess.run(['rclone','copyto',str(proof),a.remote+'/'+str(proof.relative_to(R)),'--disable-http2'],check=True)
   for path in paths[:-1]:path.unlink()
  except Exception as e:
   (R/'restart_archive_error.json').write_text(json.dumps(dict(time=time.time(),error=str(e))))
 time.sleep(10)
