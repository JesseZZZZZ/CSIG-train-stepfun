#!/bin/bash
# rjob.sh — the ONLY way this experiment touches the cluster. Login-side control plane.
#
#   ./rjob.sh submit <body-script-path-in-pod> [env KEY=VAL ...]
#   ./rjob.sh status
#   ./rjob.sh logs [n]
#   ./rjob.sh stop
#
# Owner constraints encoded here rather than left to discipline:
#   1. 🔴 rjob name is EXACTLY `test` — no suffix, no alias (owner, 2026-09-01).
#   2. 🔴 Total GPU usage must never exceed 8. Since the name is a single fixed string, only
#      one rjob of ours can exist at a time, which makes the 8-card ceiling structural rather
#      than something to remember. `submit` refuses if `test` is already alive.
#   3. `stop` only ever stops `test`, and only if THIS script launched it (registry file).
#      `test` is a maximally generic name in a SHARED namespace — 20 other jobs from
#      colleagues are visible in `brainctl get rjob`. A name check alone would be no guard
#      at all, so the registry is the real gate.
#   4. `--predict-only` is NEVER used as go/no-go. MEASURED 2026-09-01: it reported
#      "gpu 353/352" for coding and "152/144" for sessionrouter, i.e. refusal for both,
#      which contradicts rsi_2608's measurement that a real submit is admitted anyway.
#      The predictor and the admission path disagree; trust the submit.
#   5. `--detach`: MEASURED in rsi_2608 that an attached multi-GPU submit dies at
#      `timeout waiting for pod after 30m0s` — that is the CLIENT giving up, not the
#      scheduler refusing. Detached, the job waits server-side and the pod writes to JFS.
set -uo pipefail

RJOB_NAME="${RJOB_NAME:-test}"   # 🔴 defaults to `test` (owner: no alias for the MAIN run). Owner
                                 # 2026-09-11 sanctioned "再开一组H200", so a SECOND concurrent group
                                 # may override this (e.g. RJOB_NAME=glm53sft) — its own registry row.
NS="${NS:-shai-core}"
GROUP="${GROUP:-coding}"
TAGS="${TAGS:-h200}"
GPUS="${GPUS:-1}"
CPUS="${CPUS:-16}"
MEM_MB="${MEM_MB:-131072}"
PRIORITY="${PRIORITY:-Medium}"

ROOT_LOCAL=/mnt/ws-jfs/posttrain/i-xuyiming/paper/data_select_2609
LOGDIR="${ROOT_LOCAL}/logs"
LAUNCHED="${LOGDIR}/launched_rjobs.tsv"
mkdir -p "$LOGDIR"

if [ "${GPUS}" -gt 8 ]; then
  echo "REFUSING: GPUS=${GPUS} exceeds the owner's 8-card ceiling." >&2; exit 9
fi

phase () { /kubebrain/brainctl get rjob "$RJOB_NAME" -n "$NS" 2>/dev/null | awk 'NR==2{print $2}'; }

case "${1:-}" in
  submit)
    BODY="${2:?need pod-side body script path}"; shift 2
    ph=$(phase)
    case "$ph" in
      Running|Starting|Pending|Scheduled|Queueing)
        echo "REFUSING: rjob '$RJOB_NAME' already exists in phase=$ph." >&2
        echo "Only one '$RJOB_NAME' can exist, which is what keeps total GPUs <= 8." >&2
        echo "Run '$0 stop' first if that job is ours and finished with." >&2
        exit 9 ;;
      Succeeded|Failed|Stopped)
        # 🔴 MEASURED 2026-09-01: a finished rjob keeps OWNING its name —
        # `failed to create rjob: rjobs.rjob.brainpp.cn "test" already exists` — and
        # `brainctl stop` refuses it ("can not stop this rjob, current phase is Succeeded").
        # Only `delete` frees the name. With the name pinned to the single string `test`,
        # every run must reap its predecessor, so do it here instead of by hand.
        # Guarded by the registry: we never delete a `test` we did not launch.
        if ! [ -s "$LAUNCHED" ] || ! cut -f1 "$LAUNCHED" | grep -Fxq "$RJOB_NAME"; then
          echo "REFUSING: '$RJOB_NAME' exists (phase=$ph) but is NOT in our registry." >&2
          echo "It may belong to someone else — '$RJOB_NAME' is a maximally generic name" >&2
          echo "in a namespace shared with ~20 other jobs. Not touching it." >&2
          exit 9
        fi
        echo "reaping our own finished rjob '$RJOB_NAME' (phase=$ph) to free the name"
        /kubebrain/brainctl delete "rjob/$RJOB_NAME" -n "$NS" 2>&1 | head -3
        for i in $(seq 1 30); do [ -z "$(phase)" ] && break; sleep 2; done
        [ -n "$(phase)" ] && { echo "REFUSING: '$RJOB_NAME' still present after delete." >&2; exit 9; }
        ;;
    esac
    # 🔴 MEASURED 2026-09-01: bash reads a script from disk LAZILY. Editing serve_eval_body.sh
    # while a pod was executing it shifted the byte offsets under the running interpreter and it
    # died mid-run with `syntax error near unexpected token \`do\'` — on a file that passes
    # `bash -n` perfectly. So every submission runs from an immutable snapshot, and later edits
    # can never reach into a live job.
    SNAP="${LOGDIR}/snap_$(date -u +%Y%m%dT%H%M%SZ)"
    mkdir -p "$SNAP"
    cp -r "$(dirname "$0")"/. "$SNAP"/
    BODY_SNAP="${SNAP#/mnt/ws-jfs}"                 # login view -> pod view
    BODY_SNAP="/mnt${BODY_SNAP}/$(basename "$BODY")"
    echo "snapshot: $SNAP"
    echo "pod will run: $BODY_SNAP"
    ENVFLAGS=(-e "DEPLOY_DIR=$(dirname "$BODY_SNAP")")
    for kv in "$@"; do ENVFLAGS+=(-e "$kv"); done
    TS=$(date -u +%Y%m%dT%H%M%SZ)
    SUBLOG="${LOGDIR}/submit_${TS}.log"
    printf '%s\t%s\tgpus=%s\tgroup=%s\ttags=%s\tbody=%s\n' \
        "$RJOB_NAME" "$TS" "$GPUS" "$GROUP" "$TAGS" "$BODY" >> "$LAUNCHED"
    echo "submitting name=$RJOB_NAME gpus=$GPUS group=$GROUP tags=$TAGS"
    echo "body=$BODY"
    echo "env=${*:-none}"
    nohup /kubebrain/rlaunch \
        --cpu "$CPUS" --memory "$MEM_MB" --gpu "$GPUS" \
        --charged-group="$GROUP" --priority="$PRIORITY" \
        --positive-tags="$TAGS" \
        --host-network --detach \
        --name "$RJOB_NAME" \
        --mount=juicefs+s3://aliyun-sh/posttrain:/mnt/posttrain \
        "${ENVFLAGS[@]}" \
        -- bash -lc "bash $BODY_SNAP" \
        > "$SUBLOG" 2>&1 &
    echo "submit pid=$! log=$SUBLOG"
    ;;
  status)
    echo "--- rjob $RJOB_NAME ---"
    /kubebrain/brainctl get rjob "$RJOB_NAME" -n "$NS" 2>&1 | head -6
    echo "--- last submit log ---"
    tail -12 "$(ls -t ${LOGDIR}/submit_*.log 2>/dev/null | head -1)" 2>/dev/null || echo "(none)"
    ;;
  logs)
    tail -"${2:-40}" "$(ls -t ${LOGDIR}/submit_*.log 2>/dev/null | head -1)" 2>/dev/null || echo "(none)"
    ;;
  stop)
    if ! [ -s "$LAUNCHED" ] || ! cut -f1 "$LAUNCHED" | grep -Fxq "$RJOB_NAME"; then
      echo "REFUSING to stop '$RJOB_NAME': not in our registry $LAUNCHED" >&2; exit 9
    fi
    echo "stopping rjob/$RJOB_NAME in $NS (phase was: $(phase))"
    /kubebrain/brainctl stop "rjob/$RJOB_NAME" -n "$NS" 2>&1 | head -5
    ;;
  *) sed -n '2,10p' "$0"; exit 1 ;;
esac
