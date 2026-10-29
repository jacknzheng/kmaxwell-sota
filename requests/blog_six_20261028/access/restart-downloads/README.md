# Requested restart downloads — supplement to the complete takeover ZIP

This supplement supplies the missing training restart access. Keep your existing full takeover package. No owner account credentials or Strange Loop login are needed.

Run with Python 3 (standard library only):

    python3 download_plan.py downloads.jsonl.gz --dest /your/data/root --workers 16

List before downloading:

    python3 download_plan.py downloads.jsonl.gz --dest /your/data/root --list

Select one restart using --only and its full path prefix. The downloader preserves the catalog paths beneath --dest. It resumes partial files block-by-block. These are bearer URLs: keep this package private. The service may expire links; an expiry leaves partial files resumable. Ask us for refreshed links if that happens. The outer ZIP link expiry does NOT establish the internal block-link lifetime.

Contents: shared boundary_2009 full restart (8 rank files + markers), native seed0 MaxEnt restarts step_01250, step_01750, step_02250 (8 rank files + manifest each), and training shards fineweb_train_000001.bin through 000015.bin. Total 52 files, 373,084,368,545 bytes (~373 GB decimal). Fixed validation shard and source remain in the prior handoff.

Downloads are direct from research storage to your machine, not via a GPU rented by us. Downloader sets the required User-Agent and reads blocks in their original order. DOWNLOAD_TEST.json records successful retrieval of metadata and representative first blocks from all four restart groups plus shard15; it is not a full-checkpoint verification claim.

All our Vast instances were destroyed at user request on October28 around04:29UTC. Florida is not running; no future output will arrive from the old GPU queues. Training and measurement work now belongs to your compute. Do not trust old node status files as live state.

For MaxEnt density, latest requested range is1000–2500. Reuse existing measurements below1250. Capture every25 updates during short replays and measure missing coarse points first, then adaptively fill visual gaps. 2600 was a stale queue extension and is not required. Preserve separate replay provenance per native anchor. Replay drivers in the original package are reference orchestration with machine-specific paths, locks, and upload destinations; they must not be launched as-is.
