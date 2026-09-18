#!/usr/bin/env bash
# run_ddp.sh -- launch the 8-rank data-parallel LoRA SFT inside the B300 pod.
#
# WHY A HAND-ROLLED LAUNCHER INSTEAD OF torchrun
#   The house rule is that EVERY python3 runs with `-I -B` (isolated mode, no bytecode) because of
#   the GRADER-HIJACK fixture that shadows stdlib module names. torchrun spawns its own children
#   and does not pass those flags down, so the guarantee would silently stop at rank 0. Here every
#   rank is an explicit `python3 -I -B`, so the invariant holds for all 8 processes.
#
# WHY PID-TRACKED CLEANUP AND NEVER `pkill -f`
#   A `pkill -f <pattern>` run from a script whose own command line contains that pattern kills the
#   script. That happened twice in this project (exit 144, twice, silently losing a run). So: record
#   each rank's PID, and only ever signal those exact PIDs.
#
# WHY THE FAILURE PATH MATTERS
#   Ranks synchronise on collectives (all_reduce/barrier). If one rank dies -- OOM, bad shard, CUDA
#   error -- the surviving 7 block inside NCCL until the timeout (default here 60 min) and the pod
#   looks "Running" at 0% GPU the whole time. That exact false-healthy state has already burned 20
#   minutes of a session in this project. So we poll, and the moment any rank exits non-zero we tear
#   the rest down and report which rank died first.
set -uo pipefail

MODEL="${MODEL:?set MODEL to the model dir}"
DATA="${DATA:?set DATA to the tokenised train.jsonl}"
WORLD="${WORLD:-8}"
STEPS="${STEPS:-200}"
ACCUM="${ACCUM:-1}"
LR="${LR:-1e-4}"
MAXLEN="${MAXLEN:-65536}"
OUT="${OUT:-/data/ckpt_ddp}"
SCRIPT="${SCRIPT:-/tmp/train_sft_lora_ddp.py}"
LOGDIR="${LOGDIR:-/data/ddp_logs}"
# LoRA capacity. Exposed because v7's held-out curve flattened (improvement per 50 steps decayed
# 10x) while its minimum was still the LAST step -- that is consistent with either a data ceiling or
# a CAPACITY ceiling, and rank is the single-variable way to tell them apart.
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
TARGETS="${TARGETS:-q_proj,k_proj,v_proj,o_proj}"
# 🔴 MUST be forwarded, for the same reason LORA_R had to be: a flag this launcher does not pass is a
# flag the run silently does not get. Before LORA_R was forwarded, v8 would have been a disguised v7
# rerun. CHUNKED_CE is the same trap with higher stakes -- an unforwarded 0 means the run quietly uses
# HF's labels= path, so it would be a v9 rerun wearing a v11 label, and the memory/speed claim would
# be measured on the wrong code path.
# 🔴 关卡 0 需要的两个种子。按本文件自己的规矩："a flag this launcher does not pass is a
# flag the run silently does not get" —— 之前 --bucket-seed 就没被传下去，所以历史上所有运行
# 的数据顺序其实是同一个（1234），只有 LoRA 初始化在变。两个都必须显式转发。
SEED="${SEED:-0}"                 # 0 = 不播种，与历史运行一致
BUCKET_SEED="${BUCKET_SEED:-1234}"
CHUNKED_CE="${CHUNKED_CE:-0}"
CHUNK_PROOF_LEN="${CHUNK_PROOF_LEN:-8192}"

export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29517}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"          # WARN, not INFO: INFO floods 8 ranks x MBs
export TOKENIZERS_PARALLELISM=false              # 8 ranks x tokenizer threads = pointless churn
# The single-card run was measured pinning only 91-94% of ONE core out of 1600% allowed, i.e. it is
# kernel-launch-bound. With 8 ranks on one node, letting each spawn 249/8 OMP threads would thrash;
# cap it so the ranks do not fight over the CPUs they each need for launching.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

mkdir -p "$LOGDIR" "$OUT"
echo "=== DDP LAUNCH world=$WORLD steps=$STEPS accum=$ACCUM (global batch $((WORLD*ACCUM))) ==="
echo "    model=$MODEL"
echo "    data=$DATA"
echo "    lora r=$LORA_R alpha=$LORA_ALPHA targets=$TARGETS"
echo "    seed=$SEED bucket_seed=$BUCKET_SEED  (seed=0 表示不播种，与历史运行一致)"
echo "    chunked_ce=$CHUNKED_CE (0 = HF labels= path)  chunk_proof_len=$CHUNK_PROOF_LEN"
echo "    out=$OUT   logs=$LOGDIR   master=$MASTER_ADDR:$MASTER_PORT"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits \
  | awk -F', ' '{printf "    gpu %s: %s / %s MiB used\n",$1,$2,$3}'

PIDS=()
for r in $(seq 0 $((WORLD-1))); do
  python3 -I -B "$SCRIPT" \
      --rank "$r" --world "$WORLD" \
      --model "$MODEL" --data "$DATA" \
      --steps "$STEPS" --accum "$ACCUM" --lr "$LR" --max-len "$MAXLEN" \
      --rank16 "$LORA_R" --alpha "$LORA_ALPHA" --targets "$TARGETS" \
      --chunked-ce "$CHUNKED_CE" --chunk-proof-len "$CHUNK_PROOF_LEN" \
      --seed "$SEED" --bucket-seed "$BUCKET_SEED" \
      --out "$OUT" \
      > "$LOGDIR/rank$r.log" 2>&1 &
  PIDS+=($!)
  echo "    rank $r -> pid ${PIDS[$r]}  log $LOGDIR/rank$r.log"
done

cleanup() {
  # Only ever touch the PIDs we started ourselves.
  for p in "${PIDS[@]}"; do
    kill -0 "$p" 2>/dev/null && kill -TERM "$p" 2>/dev/null
  done
  sleep 5
  for p in "${PIDS[@]}"; do
    kill -0 "$p" 2>/dev/null && kill -KILL "$p" 2>/dev/null
  done
}
trap cleanup EXIT INT TERM

FAILED_RANK=-1
FAILED_CODE=0
while :; do
  alive=0
  for r in $(seq 0 $((WORLD-1))); do
    p="${PIDS[$r]}"
    if kill -0 "$p" 2>/dev/null; then
      alive=$((alive+1))
    else
      wait "$p"; rc=$?
      if [ "$rc" -ne 0 ] && [ "$FAILED_RANK" -lt 0 ]; then
        FAILED_RANK="$r"; FAILED_CODE="$rc"
      fi
    fi
  done
  [ "$alive" -eq 0 ] && break
  if [ "$FAILED_RANK" -ge 0 ]; then
    echo "!!! rank $FAILED_RANK exited $FAILED_CODE while $alive rank(s) still running."
    echo "!!! the survivors would now block in NCCL until timeout -- tearing them down."
    echo "--- last 30 lines of the failing rank ($LOGDIR/rank$FAILED_RANK.log) ---"
    tail -30 "$LOGDIR/rank$FAILED_RANK.log"
    cleanup
    break
  fi
  sleep 10
done
trap - EXIT INT TERM

echo
echo "=== rank 0 log (the full training trace) ==="
cat "$LOGDIR/rank0.log"
echo
echo "=== per-rank tail (proves all ranks did work, not just rank 0) ==="
for r in $(seq 0 $((WORLD-1))); do
  echo "--- rank $r ---"
  tail -3 "$LOGDIR/rank$r.log"
done

if [ "$FAILED_RANK" -ge 0 ]; then
  echo "RESULT: SFT_DDP_LAUNCH_FAILED rank=$FAILED_RANK code=$FAILED_CODE"
  exit 1
fi
echo "=== train_result.json ==="
[ -f "$OUT/train_result.json" ] && python3 -I -B -c "
import json,sys
d=json.load(open('$OUT/train_result.json'))
for k in ('world','global_batch','steps','median_step_s','loss_first3','loss_last3','decreased',
          'peak_gib','shard_counts','eval_hist','eval_decreased','sync_proof'):
    print('  %-16s %s'%(k,d.get(k)))
" || echo "  (missing -- rank 0 did not reach the save)"
