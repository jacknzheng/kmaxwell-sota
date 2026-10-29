#!/usr/bin/env python3
"""Resumable block-stream downloader for a Strange Loop URL plan. Stdlib only.

    python3 download_plan.py PLAN.jsonl.gz --dest DIR [--workers 16] [--only PREFIX ...] [--list]

PLAN rows: {"path", "size", "block_size", "urls": [...]} (one row per file; urls in block
order). Files are written to DEST/<path> through a .part file plus a .blocks state file
(one byte per block), so a killed run resumes with only the missing blocks. Memory is
bounded by workers x block_size (8 MiB per block). A block URL that the store refuses
(HTTP 403/404/410 = expired) is recorded, the file is left resumable, and the run exits 3
with a JSON summary listing the files that need a fresh plan (re-mint on the owning
account; the plan file is never modified). URLs are never printed or logged.
"""
import argparse, gzip, hashlib, http.client, json, os, sys, threading, time, urllib.parse
from concurrent.futures import ThreadPoolExecutor

UA = "strangeloop-cli"  # the block store answers 403 to other user agents


class Expired(Exception):
    pass


def open_plan(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


local = threading.local()


def fetch(url, expected):
    parts = urllib.parse.urlsplit(url)
    target = parts.path + ("?" + parts.query if parts.query else "")
    failure = None
    for attempt in range(6):
        if attempt:
            time.sleep(min(8.0, 0.5 * 2 ** (attempt - 1)))
        conns = local.__dict__.setdefault("conns", {})
        conn = conns.get(parts.netloc)
        try:
            if conn is None:
                conn = conns[parts.netloc] = http.client.HTTPSConnection(parts.netloc, timeout=120)
            conn.request("GET", target, headers={"User-Agent": UA})
            resp = conn.getresponse()
            data = resp.read()
            if resp.will_close:
                conns.pop(parts.netloc, None); conn.close()
            if resp.status in (403, 404, 410):
                raise Expired()
            if resp.status != 200 or len(data) != expected:
                failure = "HTTP %d / %d bytes" % (resp.status, len(data)); continue
            return data
        except Expired:
            raise
        except (OSError, http.client.HTTPException) as exc:
            failure = type(exc).__name__
            conns.pop(parts.netloc, None)
            try: conn.close()
            except Exception: pass
    raise RuntimeError("block failed 6 times (%s)" % failure)


def download(row, dest_root, workers, verify):
    rel = row["path"].lstrip("/")
    if ".." in rel.split("/"):
        raise ValueError("unsafe path in plan: %r" % rel)
    dest = os.path.join(dest_root, rel)
    size, bs = int(row["size"]), int(row["block_size"])
    count = -(-size // bs) if size else 0
    if len(row["urls"]) != count:
        raise ValueError("%s: plan has %d urls for %d blocks" % (rel, len(row["urls"]), count))
    if os.path.exists(dest) and os.path.getsize(dest) == size:
        return dict(path=rel, status="present", bytes=size)
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    part, state = dest + ".part", dest + ".blocks"
    done = bytearray(count)
    if os.path.exists(state) and os.path.getsize(state) == count and os.path.exists(part) and os.path.getsize(part) == size:
        done[:] = open(state, "rb").read()
    fd = os.open(part, os.O_CREAT | os.O_RDWR, 0o600)
    os.ftruncate(fd, size)
    sfd = os.open(state, os.O_CREAT | os.O_RDWR, 0o600)
    os.ftruncate(sfd, count)
    lock = threading.Lock(); expired = []; errors = []; moved = [0]

    def one(i):
        if done[i] or expired or errors:
            return
        expected = min(bs, size - i * bs)
        try:
            data = fetch(row["urls"][i], expected)
        except Expired:
            with lock: expired.append(i)
            return
        except Exception as exc:
            with lock: errors.append("block %d: %s" % (i, exc))
            return
        off = 0
        while off < len(data):
            off += os.pwrite(fd, data[off:], i * bs + off)
        os.pwrite(sfd, b"\x01", i)
        with lock:
            done[i] = 1; moved[0] += expected

    pending = [i for i in range(count) if not done[i]]
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(one, pending))
    os.fsync(fd); os.close(fd); os.fsync(sfd); os.close(sfd)
    if errors:
        return dict(path=rel, status="error", detail=errors[:3], blocks_remaining=count - sum(done))
    if expired:
        return dict(path=rel, status="expired", blocks_remaining=count - sum(done), first_expired_block=min(expired))
    if verify and row.get("sha256"):
        h = hashlib.sha256()
        with open(part, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 23), b""):
                h.update(chunk)
        if h.hexdigest() != row["sha256"]:
            return dict(path=rel, status="sha256_mismatch")
    os.replace(part, dest); os.unlink(state)
    return dict(path=rel, status="done", bytes=size, seconds=round(time.time() - t0, 1), moved=moved[0])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("plan")
    ap.add_argument("--dest", required=True)
    ap.add_argument("--workers", type=int, default=16, help="blocks in flight (memory = workers x 8 MiB)")
    ap.add_argument("--only", nargs="*", default=[], help="path prefixes to download (default: all rows)")
    ap.add_argument("--list", action="store_true", help="print the plan's paths and sizes, download nothing")
    ap.add_argument("--no-verify", action="store_true")
    a = ap.parse_args()
    rows = [r for r in open_plan(a.plan) if not a.only or any(r["path"].startswith(p) for p in a.only)]
    if a.list:
        for r in rows:
            print(r["size"], r["path"])
        print(len(rows), "files", sum(r["size"] for r in rows), "bytes", file=sys.stderr)
        return 0
    summary = dict(done=0, present=0, expired=[], errors=[], bytes=0)
    for r in rows:
        out = download(r, a.dest, a.workers, not a.no_verify)
        print(json.dumps(out), flush=True)
        if out["status"] in ("done", "present"):
            summary[out["status"]] += 1; summary["bytes"] += out["bytes"]
        elif out["status"] == "expired":
            summary["expired"].append(out["path"])
        else:
            summary["errors"].append(out)
    print(json.dumps(dict(summary=summary)), flush=True)
    if summary["expired"]:
        print("URLs expired for %d files; ask the plan owner for a fresh plan (re-mint), then rerun the same command: only missing blocks are fetched." % len(summary["expired"]), file=sys.stderr)
        return 3
    return 1 if summary["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
