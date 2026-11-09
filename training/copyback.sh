#!/bin/bash
# Run on your machine while the pod trains:
#
#   SSH_KEY=~/.ssh/my_pod_key bash copyback.sh HOST PORT [LOCAL_DIR]
#
# Every 60 s: touch the pod's KEEPALIVE (so guard.sh leaves it running), pull
# STATUS and the logs, and pull each checkpoint that has its SAVED marker
# (later epochs reuse unchanged files from epoch 0 with --copy-dest). Exits once
# the pod has written DONE or FAILED and every saved checkpoint is local.
# LOCAL_DIR defaults to ./ft-output.
set -u
H=$1; P=$2; D=${3:-./ft-output}; KEY=${SSH_KEY:-$HOME/.ssh/id_ed25519}
S="ssh -i $KEY -p $P -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 -o ServerAliveInterval=15"
R=/workspace/ft
mkdir -p "$D/output" "$D/logs"
while true; do
  $S "root@$H" "touch $R/KEEPALIVE" || echo "$(date -u +%TZ) ssh failed"
  rsync -a -e "$S" "root@$H:$R/{STATUS,run.log,guard.log,pip-freeze.txt}" "$D/logs/" 2>/dev/null
  for e in $($S "root@$H" "cd $R/output 2>/dev/null && for d in checkpoint-epoch-*; do [ -e \$d/SAVED ] && echo \$d; done"); do
    [ -e "$D/output/$e/.copied" ] && continue
    extra=""
    [ "$e" != checkpoint-epoch-0 ] && [ -e "$D/output/checkpoint-epoch-0/.copied" ] \
      && extra="--copy-dest=$(cd "$D/output/checkpoint-epoch-0" && pwd)/"
    if rsync -a --partial $extra -e "$S" "root@$H:$R/output/$e/" "$D/output/$e/"; then
      touch "$D/output/$e/.copied"; echo "$(date -u +%TZ) copied $e ($(du -sh "$D/output/$e" | cut -f1))"
    fi
  done
  if $S "root@$H" "test -e $R/DONE -o -e $R/FAILED"; then
    rsync -a -e "$S" "root@$H:$R/output/sanity.json" "$D/output/" 2>/dev/null
    rsync -a -e "$S" "root@$H:$R/{STATUS,run.log}" "$D/logs/"
    have=$(ls -d "$D"/output/checkpoint-epoch-*/.copied 2>/dev/null | wc -l | tr -d ' ')
    saved=$($S "root@$H" "ls -d $R/output/checkpoint-epoch-*/SAVED 2>/dev/null | wc -l")
    if [ "$have" -ge "$saved" ]; then
      echo "$(date -u +%TZ) finished: $(tail -1 "$D/logs/STATUS"); $have checkpoints copied"; exit 0
    fi
  fi
  sleep 60
done
