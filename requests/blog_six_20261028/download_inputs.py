#!/usr/bin/env python3
"""Download selected starter captures using scoped, expiring GET links; no cloud credentials."""
import argparse,concurrent.futures,hashlib,json,os,pathlib,urllib.request
P=argparse.ArgumentParser(description=__doc__);P.add_argument('--manifest',default='access/downloads.json');P.add_argument('--update',type=int,action='append',help='1300, 1325, or 1425; repeat to select several. Default: all.');P.add_argument('--workers',type=int,default=4);P.add_argument('--list',action='store_true');a=P.parse_args()
root=pathlib.Path(__file__).resolve().parent;manifest=json.loads((root/a.manifest).read_text());items=[x for x in manifest['files'] if a.update is None or x.get('update') is None or x['update'] in a.update]
def digest(p):
 h=hashlib.sha256()
 with p.open('rb')as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
 return h.hexdigest()
def fetch(x):
 p=(root/x['path']).resolve()
 if not p.is_relative_to(root):raise ValueError('Manifest path escapes package')
 def valid(path):return path.exists() and path.stat().st_size==x['bytes'] and (not x.get('sha256') or digest(path)==x['sha256'])
 if valid(p):print('Already present:',x['path'],flush=True);return
 p.parent.mkdir(parents=True,exist_ok=True);part=p.with_name(p.name+'.part');offset=part.stat().st_size if part.exists()else 0
 if offset>x['bytes']:part.unlink();offset=0
 if offset<x['bytes']:
  req=urllib.request.Request(x['url'],headers={'Range':f'bytes={offset}-'} if offset else {})
  with urllib.request.urlopen(req,timeout=180)as response:
   append=offset>0 and response.status==206
   with part.open('ab' if append else 'wb')as f:
    while chunk:=response.read(8*1024*1024):f.write(chunk)
 if not valid(part):
  part.unlink(missing_ok=True);raise RuntimeError('Size or SHA256 mismatch: '+x['path']+'; rerun to download again')
 part.replace(p);print('Ready:',x['path'],flush=True)
print('Links expire:',manifest['expires_utc']);print('Selected bytes:',sum(x['bytes']for x in items))
if a.list:
 for x in items:print(x['path'],x['bytes'])
else:
 with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers)as pool:list(pool.map(fetch,items))
