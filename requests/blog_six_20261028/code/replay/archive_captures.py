"""Archive closed8rank snapshots; evict only SHA-pinned local files after remote size proof."""
import argparse,pathlib,json,subprocess,time,hashlib,fcntl,concurrent.futures,os
P=argparse.ArgumentParser();P.add_argument('--runtime',default='/workspace/pr323_blog');P.add_argument('--remote',required=True);a=P.parse_args();R=pathlib.Path(a.runtime)
flags=['--disable-http2','--s3-upload-cutoff','8Mi','--s3-chunk-size','8Mi','--s3-upload-concurrency','32']
while True:
 for index in (R/'blog_outputs').glob('*/captures/index.jsonl'):
  cap=index.parent;groups={}
  for line in index.read_text().splitlines():
   try:d=json.loads(line)
   except ValueError:continue
   if d.get('kind')=='boundary':groups.setdefault(d['update'],{})[d['file']]=d
  for t,pins in sorted(groups.items()):
   if len(pins)!=8:continue
   lock=(cap/f'.point_{t}.lock').open('a')
   try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
   except BlockingIOError:lock.close();continue
   try:
    paths=[cap/k for k in pins]
    if not all(p.exists()for p in paths):continue
    for p in paths:
     pin=pins[p.relative_to(cap).as_posix()]
     assert p.stat().st_size==pin['bytes']
     with p.open('rb')as f:assert hashlib.file_digest(f,'sha256').hexdigest()==pin['sha256']
    listing=cap/f'.upload_{t}.txt';listing.write_text('\n'.join(str(p.relative_to(R))for p in paths)+'\n')
    subprocess.run(['rclone','copy',str(R),a.remote,'--files-from',str(listing),'--transfers','8',*flags],check=True)
    def head(p):
     rel=str(p.relative_to(R));d=json.loads(subprocess.check_output(['rclone','lsjson',a.remote.rstrip('/')+'/'+rel,'--stat','--disable-http2']));assert d['Size']==p.stat().st_size;return dict(file=rel,bytes=d['Size'],sha256=pins[p.relative_to(cap).as_posix()]['sha256'])
    with concurrent.futures.ThreadPoolExecutor(max_workers=8)as pool:proof=list(pool.map(head,paths))
    receipt=cap/f'R2_VERIFIED_{t}.json';receipt.write_text(json.dumps(dict(update=t,remote=a.remote,verified_at=time.time(),files=proof),indent=2))
    subprocess.run(['rclone','copyto',str(receipt),a.remote.rstrip('/')+'/'+str(receipt.relative_to(R)),'--disable-http2'],check=True)
    for p in paths:p.unlink()
   except Exception as e:
    (R/'capture_archive_error.json').write_text(json.dumps(dict(time=time.time(),update=t,error=str(e))))
   finally:lock.close()
 time.sleep(3)
