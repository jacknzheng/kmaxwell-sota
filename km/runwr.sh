#!/bin/bash
# Serial WR-run driver: one kmwr.py per specfile, one torchrun at a time.
# Launch detached with:  nohup bash /root/km/runwr.sh <spec1.json> <spec2.json> ... &
# (plain nohup — NOT (setsid nohup ... &); see WRITEUP.md §9)
for f in "$@"; do
  echo "==== $(date -u +%FT%TZ) START $f ====" >> /root/km/wr_driver.log
  python3 /root/km/kmwr.py --specfile "$f" >> /root/km/wr_driver.log 2>&1
  echo "==== $(date -u +%FT%TZ) END $f rc=$? ====" >> /root/km/wr_driver.log
done
touch /root/km/WR_DONE
