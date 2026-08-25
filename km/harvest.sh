#!/bin/bash
# Pull all campaign artifacts from the 6 boxes (serial ssh — parallel ssh to
# different jobs races on the shared baseten cert file).
#   ledger/<run>/            -> ~/jackzhengretardruns/ledger/<run>/
#   logs/kmaxwell/<fleet>/   -> ~/modded-nanogpt/logs/kmaxwell/<fleet>/
set -uo pipefail
LEDGER=~/jackzhengretardruns/ledger
LOGROOT=~/modded-nanogpt/logs/kmaxwell
mkdir -p "$LEDGER" "$LOGROOT"
for id in wlmz82q 3y51enw wxgj90q q974r03 qrgxkr3 qz7dvew; do
  H=training-job-$id-0.ssh.baseten.co
  echo "== $id =="
  names=$(ssh -o ConnectTimeout=10 $H 'ls -d /root/km/ledger/*/ 2>/dev/null' | xargs -n1 basename 2>/dev/null)
  for n in $names; do
    mkdir -p "$LEDGER/$n"
    scp -q $H:"/root/km/ledger/$n/{spec.json,verdict.json,cmd.txt,train.log}" "$LEDGER/$n/" 2>/dev/null \
      || for fn in spec.json verdict.json cmd.txt train.log; do scp -q $H:/root/km/ledger/$n/$fn "$LEDGER/$n/" 2>/dev/null; done
    echo "  pulled $n"
  done
  fleets=$(ssh -o ConnectTimeout=10 $H 'ls -d /root/modded-nanogpt/logs/kmaxwell/*/ 2>/dev/null' | xargs -n1 basename 2>/dev/null)
  for f in $fleets; do
    mkdir -p "$LOGROOT/$f"
    scp -q $H:"/root/modded-nanogpt/logs/kmaxwell/$f/*" "$LOGROOT/$f/" 2>/dev/null
    echo "  pulled logs/kmaxwell/$f"
  done
done
echo HARVEST-DONE
