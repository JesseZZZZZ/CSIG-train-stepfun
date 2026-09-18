#!/usr/bin/env bash
# launch.sh — reproduce the deepseek glm-5.3 full-data SFT on H200 (coding quota), ep1.
#
# This is the EXACT config that trained successfully (2026-09-17). Key hard-won settings:
#   - MAXLEN=131072: the largest single-card length that fits one H200 (262144 OOMs by ~7 GiB — measured).
#   - OFFLOAD_ACT=1 + PARAM_OFFLOAD=1: BOTH required at 131072 (else GPU ~166 GiB OOM).
#   - MEM_MB=700000: MUST be big enough to land on a >=683 GB node. A smaller request gets squeezed onto a
#     390 GB node and the host OOM-killer kills a rank (code 137). This was the #1 failure — see README.
#   - PROBE=0: skip the memory sweep (131072 fit is already confirmed).
#   - WORLD=4: 4 data-parallel ranks (<=6 coding budget with peer). GPU per-rank fits; host RAM is the limit.
set -euo pipefail
PD=/mnt/posttrain/i-xuyiming/paper
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"   # data_select_2609

# rjob.sh control vars MUST be exported BEFORE the command (not passed as body args — that silently
# requests 1 GPU). Body/training vars come after.
RJOB_NAME=test GPUS=4 MEM_MB=700000 CPUS=48 GROUP=coding TAGS=h200 \
bash "$HERE/rjob.sh" submit run_sft_fla.sh \
  MODEL=$PD/rsi_2608/qwen-3.8-27B/model/Qwen3.8-27B \
  SRC_DATA=$ROOT/sft/packed_glm53ds_131k/train.jsonl \
  STACK=$ROOT/sft/stack_glm53ds_offload \
  OUT=$ROOT/sft/ckpt_glm53ds_h200_131k \
  MAXLEN=131072 WORLD=4 STEPS=432 ACCUM=2 PROBE=0 \
  OFFLOAD_ACT=1 PARAM_OFFLOAD=1 OFFLOAD_PIN=0 PARAM_OFFLOAD_PIN=0 \
  LORA_R=8 LORA_ALPHA=16 CHUNKED_CE=2048 CKPT_EVERY=50 EVAL_EVERY=25

# STEPS=432 = 1 epoch (3455 train pieces / global_batch 8). For ep2 use STEPS=864.
# Monitor: brainctl get rjob test -n shai-core
#          brainctl exec replica/test-<hash> -n shai-core -- tail /dev/shm/xym_sft2609/logs/rank0.log
#   ⚠️ the rjob phase reads "Succeeded" even when a rank was OOM-killed mid-run — ALWAYS check
#      progress.jsonl line-count vs total_steps + grep the submit log for code=137/LAUNCH_FAILED.
