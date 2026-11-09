#!/bin/bash
# Pod-side guard, started by run_pod.sh: stops the pod (its disk is kept) when
# nobody is watching, so a forgotten pod cannot bill for long.
#
# copyback.sh on your machine touches KEEPALIVE every minute. The guard stops
# the pod when KEEPALIVE is over 45 min old and either the run has ended (DONE
# or FAILED) or 3 h have passed since the guard started.
#
# Uses the RUNPOD_POD_ID and RUNPOD_API_KEY that Runpod sets in the pod's
# environment, and Runpod's REST API (POST /v2/pods/<id>/action {"action":"stop"}).
W=${W:-/workspace/ft}; T0=$(date +%s)
eval "$(tr '\0' '\n' < /proc/1/environ | grep -E '^RUNPOD_(POD_ID|API_KEY)=' | sed 's/^/export /')"
API=https://api.runpod.io/v2/pods/$RUNPOD_POD_ID
code=$(curl -s -o /dev/null -w "%{http_code}" -H "Authorization: Bearer $RUNPOD_API_KEY" "$API")
echo "$(date -u +%TZ) guard armed for $RUNPOD_POD_ID (API read: HTTP $code)" >> "$W/guard.log"
while sleep 60; do
  now=$(date +%s)
  ka=$(stat -c %Y "$W/KEEPALIVE" 2>/dev/null || echo "$T0")
  idle=$(( now - ka ))
  ended=0; { [ -e "$W/DONE" ] || [ -e "$W/FAILED" ]; } && ended=1
  if [ "$idle" -gt 2700 ] && { [ "$ended" = 1 ] || [ $(( now - T0 )) -gt 10800 ]; }; then
    echo "$(date -u +%FT%TZ) guard: stopping pod (ended=$ended, no keepalive for ${idle}s)" >> "$W/STATUS"
    curl -s -X POST -H "Authorization: Bearer $RUNPOD_API_KEY" -H "Content-Type: application/json" \
      -d '{"action":"stop"}' "$API/action" >> "$W/guard.log" && exit 0
  fi
done
