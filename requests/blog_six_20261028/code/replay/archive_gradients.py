"""Preserve extra native gradient captures with the same verified-before-evict policy."""
import pathlib,json,subprocess,sys,time,fcntl,hashlib
R=pathlib.Path('/workspace/pr323_blog');remote=sys.argv[1]
while True:
 for index in (R/'blog_outputs').glob('*/captures/index.jsonl'):
  cap=index.parent;rows=[];closed={}
  for line in index.read_text().splitlines():
   try:d=json.loads(line)
   except ValueError:continue
   if d.get('kind')=='gradient':rows.append(d)
   if d.get('kind')=='boundary':closed.setdefault(d['update'],set()).add(d['file'])
  for d in rows:
   t=d['update'];p=cap/d['file']
   if not p.exists()or len(closed.get(t,[]))!=8:continue
   lock=(cap/f'.point_{t}.lock').open('a')
   try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
   except BlockingIOError:lock.close();continue
   try:
    assert p.stat().st_size==d['bytes']
    with p.open('rb')as f:assert hashlib.file_digest(f,'sha256').hexdigest()==d['sha256']
    dest=remote.rstrip('/')+'/'+str(p.relative_to(R))
    subprocess.run(['rclone','copyto',str(p),dest,'--disable-http2','--s3-upload-cutoff','8Mi','--s3-chunk-size','8Mi','--s3-upload-concurrency','32'],check=True)
    head=json.loads(subprocess.check_output(['rclone','lsjson',dest,'--stat','--disable-http2']));assert head['Size']==d['bytes']
    receipt=cap/f'R2_GRADIENT_VERIFIED_{t}.json';receipt.write_text(json.dumps(dict(time=time.time(),file=d['file'],bytes=d['bytes'],sha256=d['sha256'],remote=dest)))
    subprocess.run(['rclone','copyto',str(receipt),remote.rstrip('/')+'/'+str(receipt.relative_to(R)),'--disable-http2'],check=True)
    p.unlink()
   except Exception as e:(R/'gradient_archive_error.json').write_text(json.dumps(dict(time=time.time(),update=t,error=str(e))))
   finally:lock.close()
 time.sleep(3)
