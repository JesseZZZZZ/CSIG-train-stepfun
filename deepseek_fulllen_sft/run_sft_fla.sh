#!/usr/bin/env bash
# run_sft.sh -- pod-side body script: 8-rank data-parallel LoRA SFT of Qwen3.8-27B on the
#               gpt-5.6 Python->JavaScript repo-translation trajectories, on 8xH200 (143 GB/card).
#
# Modelled on rsi_2608/qwen-3.8-27B/sft/{run_ddp.sh, sft_27b_ddp8_v10_24k.sh}. It REUSES that
# trainer and that launcher unchanged; everything here is environment, staging and the flags.
# Read docs/SFT_PLAN.md for why each number is what it is. DO NOT LAUNCH THIS FROM THE LOGIN HOST.
#
# ============================================================================================
# WHAT IS DIFFERENT FROM EVERY PRIOR RUN IN rsi_2608, and why -- all four are load-bearing
# ============================================================================================
# 1. THE CARD IS HALF THE SIZE. Every rsi_2608 number was measured on B300 at 267.7-268.6 GiB.
#    H200 is 143 GB. The prior runs peaked at 148.6-170.5 GiB *allocated* (v4..v10), i.e. they
#    DO NOT FIT here at all. Only the chunked-CE configuration (v11c/v12: 101.97 GiB allocated at
#    max_len 32768) is even a candidate, and the measured single-card ladder
#        peak_GiB = 53.35 + 1.0126 * (T / 1000)          [fit over T = 32768..163840, chunk 4096]
#    puts max_len 16384 at ~69.9 GiB and max_len 32768 at ~86.5 GiB allocated.
#    We take 16384. See §3 of SFT_PLAN.md; the data agrees independently (p90 = 20,088 tokens).
#    ⚠️ `peak_gib` printed by the trainer is torch's ALLOCATED, and the MEASURED
#    allocated-vs-nvidia-smi gap on B300 was +88.8..+109.5 GiB. That gap is allocator slack, and
#    on a 143 GB card it cannot exist -- but that is a prediction, not a measurement, which is
#    exactly why PROBE=1 below exists and why expandable_segments is on.
# 2. --chunked-ce 4096 IS MANDATORY, not an optimisation. HF's labels= path materialises
#    [B,T,vocab] in fp32 and vocab here is 248320, so T=16384 alone is 15.2 GiB and T=32768 is
#    30.3 GiB as ONE allocation -- the exact monolith that made v10's ranks 1-7 log
#    `failed to allocate 32,065,454,080 bytes` while peak was only 170.5/268. chunk 4096 is
#    3.79 GiB at ANY T. MEASURED bonus: it was also 1.12x FASTER (v11c 10.69 vs v9 11.93 s/step)
#    at byte-identical batches, and cut peak 169.03 -> 101.97 GiB.
# 3. LORA TARGETS ARE FIXED. 🔴 The trainer's default `q_proj,k_proj,v_proj,o_proj` matches on the
#    leaf attribute name, and this checkpoint has self_attn.{q,k,v,o}_proj on only 16 of its 64
#    layers -- the other 48 are `linear_attention` layers whose projections are named
#    in_proj_qkv / in_proj_z / in_proj_a / in_proj_b / out_proj. "o_proj" is NOT a substring of
#    "out_proj". So every prior run adapted 25% OF THE DEPTH and nothing else (this reconciles
#    the "injected 64 modules" line: 16 layers x 4). We target all 64 layers. This also means the
#    prior conclusion "capacity is exhausted, data is the lever" was measured on a crippled
#    adapter and should not be carried over as settled.
# 4. NO OSS FETCH. The weights are already on JFS, so the 136-1343 s parallel-fetch stage of
#    sft_27b_ddp8_v10_24k.sh is deleted, not reimplemented.
#
# WHAT IS DELIBERATELY UNCHANGED, so the comparison to v11c/v12 stays readable:
#   world 8, batch 1/rank, accum 2 (global 16), length bucketing, r=64 / alpha=128 (alpha/r=2),
#   lr 1e-4, AdamW(wd 0, betas 0.9/0.95), 10% linear warmup then FLAT, clip 1.0,
#   --early-stop-min-delta 0.0018 --early-stop-patience 3, --eval-max 64, --ckpt-every 100.
set -uo pipefail

# ---- proxy hygiene, house rule: do this before anything can try to phone out -----------------
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
export NO_PROXY='*' no_proxy='*'
export PYTHONUNBUFFERED=1

T00=$(date +%s)
stamp() { echo "  [t=$(( $(date +%s) - T00 ))s] $*"; }

# =============================================================================================
# 0. EVERY CACHE INTO /dev/shm, INCLUDING $HOME
# =============================================================================================
# 🔴 MEASURED (2026-08-31, gpu-h100-0299): the workspace root that carries $HOME filled to
# 100G/100G with 1.0M available, and triton's default `~/.triton` then died with
# `OSError: [Errno 28] No space left on device`. The stack trace surfaced AFTER the model config
# had been accepted, so it read as "model architectures failed to be inspected" -- a disk problem
# wearing a model problem's clothes, which cost a whole session to diagnose.
# $HOME is redirected too, because libraries invent new dot-directories faster than anyone can
# enumerate them.
CACHE="${CACHE:-/dev/shm/xym_sft2609}"
mkdir -p "$CACHE"/{home,triton,inductor,hf,xdg,torch,tmp,data,ckpt,logs,code}
export CACHE                                        # the probe in stage 3 reads it
export HOME="$CACHE/home"
export TRITON_CACHE_DIR="$CACHE/triton"
export TORCHINDUCTOR_CACHE_DIR="$CACHE/inductor"
export XDG_CACHE_HOME="$CACHE/xdg"
export HF_HOME="$CACHE/hf"
export TORCH_HOME="$CACHE/torch"
export TMPDIR="$CACHE/tmp"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1      # never phone home mid-load
# CPU activation offload pass-through (owner 2026-09-15, full-length 262144 training): exported here so
# the trainer subprocess inherits it via os.environ. OFFLOAD_ACT=1 turns on save_on_cpu in chunked_ce;
# OFFLOAD_PIN=1 uses pinned host mem (faster, but pins hundreds of GiB). Default off -> normal runs.
export OFFLOAD_ACT="${OFFLOAD_ACT:-0}" OFFLOAD_PIN="${OFFLOAD_PIN:-0}"
echo "############ 0. environment"
echo "  cache_root=$CACHE  HOME=$HOME  (workspace root has hit 100% before; ENOSPC masquerades"
echo "  as 'model architectures failed to be inspected')"
df -h /dev/shm "$CACHE" 2>/dev/null | sed 's/^/  df: /'

# ---- inputs (all overridable, house rule) ----------------------------------------------------
MODEL="${MODEL:-/mnt/ws-jfs/posttrain/i-xuyiming/paper/rsi_2608/qwen-3.8-27B/model/Qwen3.8-27B}"
SRC_DATA="${SRC_DATA:-/mnt/ws-jfs/posttrain/i-xuyiming/paper/data_select_2609/sft/packed/train_cap4_16k.jsonl}"
STACK="${STACK:-/mnt/ws-jfs/posttrain/i-xuyiming/paper/rsi_2608/qwen-3.8-27B/sft}"
DATA="$CACHE/data/train.jsonl"
OUT="${OUT:-$CACHE/ckpt}"
LOGDIR="${LOGDIR:-$CACHE/logs}"

# ---- the recipe (see SFT_PLAN.md §4 for the arithmetic behind each) --------------------------
WORLD="${WORLD:-8}"
STEPS="${STEPS:-1400}"          # 1400 x gb16 / 11036 train samples = 2.03 epochs
ACCUM="${ACCUM:-2}"             # global batch 16; MEASURED 0.88x the per-token cost of accum 1
LR="${LR:-1e-4}"
MAXLEN="${MAXLEN:-16384}"
LORA_R="${LORA_R:-64}"
LORA_ALPHA="${LORA_ALPHA:-128}"
CHUNKED_CE="${CHUNKED_CE:-4096}"
CHUNK_PROOF_LEN="${CHUNK_PROOF_LEN:-8192}"
EVAL_MAX="${EVAL_MAX:-64}"      # MUST equal the packer's --holdout-head or eval leaks by prompt
EVAL_EVERY="${EVAL_EVERY:-25}"
CKPT_EVERY="${CKPT_EVERY:-100}"
PROBE="${PROBE:-1}"             # 1 = run the single-card memory probe before committing 8 cards
# 🔴 All 64 layers, not just the 16 full-attention ones. See header note 3.
# Safe against the vision tower on purpose: its leaf names are `qkv`, `proj`, `linear_fc1/2`, and
# no target below is a substring of any of those, so `model.visual.*` receives no adapter.
# ⚠️ in_proj_a and in_proj_b are DELIBERATELY EXCLUDED. MEASURED shapes from the safetensors
# index: they are [48, 5120], i.e. the gated-delta-net's decay/gate projections with an output
# width of 48. A rank-64 adapter on a 5120->48 map cannot exceed rank 48, so 25% of its parameters
# are structurally dead, and these two tensors control the linear-attention forget gate -- the one
# place where a rank-64 perturbation at lr 1e-4 can destabilise a 48-of-64-layer recurrence rather
# than merely fail to help. Everything else keeps full-rank headroom:
#   q_proj [12288,5120]  k_proj [1024,5120]  v_proj [1024,5120]  o_proj [5120,6144]
#   in_proj_qkv [10240,5120]  in_proj_z [6144,5120]  out_proj [5120,6144]
# DERIVED trainable params at r=64: 16*2,621,440 + 48*2,424,832 = 158,334,976 (0.579% of 27.36e9),
# against 41,943,040 (MEASURED) in every prior run -- 3.78x the capacity over 4x the layer coverage.
TARGETS="${TARGETS:-q_proj,k_proj,v_proj,o_proj,in_proj_qkv,in_proj_z,out_proj}"

# 🔴 39.5 GiB of allocator slack was MEASURED on B300 (232.43 reserved vs 192.97 allocated).
# On a 143 GB card that slack is the difference between running and an OOM that reports plenty of
# free memory, so expandable_segments moves from "nice to have" to "on by default".
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"     # WARN not INFO: INFO floods 8 ranks x MBs
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader | sed 's/^/  gpu /'
echo "  nproc=$(nproc)"

# =============================================================================================
# 1. torch MUST be a cu12.8 build. MEASURED: a cu13 wheel cannot initialise on this driver.
# =============================================================================================
# 🔴 MEASURED on the H100/H200 fleet (driver 570.124.06 = CUDA 12.8): torch 2.11.0 / 2.12.1 /
# 2.13.0 are cu13 wheels; they pip-install fine and `nvidia-smi` looks healthy, then
# `torch.cuda.is_available()` returns False and the real error is
#     RuntimeError: The NVIDIA driver on your system is too old (found version 12080)
# The cu12.8 family is torch 2.10.0 / 2.9.1. Every rsi_2608 SFT number was produced under torch
# 2.13.0+cu130 on B300 -- a runtime that CANNOT be reproduced here, which is one more reason the
# throughput in SFT_PLAN.md is an estimate with a range and not a promise.
TORCH_SPEC="${TORCH_SPEC:-torch==2.8.0}"
PIP_INDEX="${PIP_INDEX:-http://mirrors.i.basemind.com/pypi/simple/}"
echo "############ 1. python stack"
# ---- fla enablement (1/3): triton<->libcuda preflight -- the reason run_sft_fla.sh exists ------
# run_sft.sh runs the 48 gated-delta-net (linear_attention) layers on the SLOW fp32 torch fallback.
# To get the FUSED GDN path (fla), triton must JIT-compile its cuda_utils, and for that its `-lcuda`
# needs a version-LESS libcuda.so on the link path. 🔴 The default /lib/.../libcuda.so.1 can be a
# 0-byte stub (or only .so.1 is shipped, no .so): triton then builds a .so that throws
# `undefined symbol: cuModuleGetFunction`, `import fla` raises, and transformers SILENTLY reverts to
# fp32 GDN (is_flash_linear_attention_available() only find_spec's, so it returns True and LIES that
# the kernels run). Fix (adapted from deploy/h200_qwen35_body.sh, proven ~7.6x on Qwen3.5): locate a
# REAL libcuda.so.1 (skip *stubs*), symlink libcuda.so (+ .so.1) into /dev/shm/cudalink, point
# TRITON_LIBCUDA_PATH/LD_LIBRARY_PATH/LIBRARY_PATH at it, ldconfig it, and WIPE the triton cache
# (a poisoned cuda_utils.so gets cached and reused). Must run BEFORE any torch/triton import.
echo "  -- fla enablement (1/3): triton<->libcuda preflight --"
REAL_LIBCUDA=""
for d in /usr/local/nvidia/lib64 /usr/local/nvidia/lib /usr/lib/x86_64-linux-gnu /lib/x86_64-linux-gnu \
         /usr/local/cuda/compat /usr/local/cuda/lib64 /run/nvidia/driver/usr/lib/x86_64-linux-gnu; do
  case "$d" in *stubs*) continue;; esac
  if [ -e "$d/libcuda.so.1" ] && [ -s "$d/libcuda.so.1" ]; then REAL_LIBCUDA="$d"; break; fi
done
echo "  real libcuda dir = ${REAL_LIBCUDA:-NOT-FOUND}  (ldconfig libcuda=$(ldconfig -p 2>/dev/null | grep -c libcuda))"
if [ -n "$REAL_LIBCUDA" ]; then
  mkdir -p /dev/shm/cudalink
  ln -sf "$REAL_LIBCUDA/libcuda.so.1" /dev/shm/cudalink/libcuda.so
  ln -sf "$REAL_LIBCUDA/libcuda.so.1" /dev/shm/cudalink/libcuda.so.1
  export TRITON_LIBCUDA_PATH=/dev/shm/cudalink
  export LD_LIBRARY_PATH="/dev/shm/cudalink:$REAL_LIBCUDA:${LD_LIBRARY_PATH:-}"
  export LIBRARY_PATH="/dev/shm/cudalink:${LIBRARY_PATH:-}"
  echo "/dev/shm/cudalink" > /etc/ld.so.conf.d/zz-cudalink.conf 2>/dev/null && ldconfig 2>/dev/null && echo "  ldconfig registered /dev/shm/cudalink"
else
  echo "  WARNING: no real libcuda found -- triton import (and thus the fused GDN path) will likely fail"
fi
export FLA_TILELANG=0
# 🔴 wipe run_sft.sh's OWN triton cache ($CACHE/triton on /dev/shm), NOT /tmp/tri: a broken
# cuda_utils.so compiled before this preflight would otherwise be reused and re-poison the run.
rm -rf "$TRITON_CACHE_DIR" && mkdir -p "$TRITON_CACHE_DIR"
# Pod python is PEP 668 externally-managed, so pip refuses to touch it without this flag.
# --break-system-packages puts the wheels in the interpreter's REAL site-packages, which matters
# for step 1b below: a `--target` + PYTHONPATH install would be invisible to `python3 -I`.
if ! python3 -c "import torch" 2>/dev/null; then
  timeout 3000 python3 -m pip install --break-system-packages --quiet \
    --index-url "$PIP_INDEX" "$TORCH_SPEC" "transformers>=5.15,<6" \
    > "$CACHE/pipinstall.log" 2>&1
  rc=$?; echo "  pip rc=$rc (log $CACHE/pipinstall.log)"; tail -3 "$CACHE/pipinstall.log"
fi
python3 -c "
import torch, transformers
print('  torch', torch.__version__, '| transformers', transformers.__version__)
print('  torch cuda build:', torch.version.cuda, '| arch_list:', torch.cuda.get_arch_list()[-4:])
assert torch.version.cuda and torch.version.cuda.startswith('12.8'), \
    'torch is not a cu12.8 build; a cu13 wheel cannot initialise on driver 570.124.06'
" || { echo "RESULT: TORCH_WRONG_CUDA_BUILD"; exit 2; }

# ---- fla enablement (2/3): install fla, then upgrade torch 2.8.0's bundled triton -------------
# 🔴 torch 2.8.0 bundles triton 3.4, which is INSIDE fla's Hopper correctness guard [3.4, 3.7.1):
# fla REFUSES the gated-GDN backward (chunk_bwd_dqkwg) there and transformers silently reverts to the
# fp32 fallback -- the ~26 h slow path this file exists to avoid. Upgrading triton to 3.7.1 (its #640
# regression fixed) is fla's OWN blessed fix. Pin <3.8 to the peer's proven 3.7.x, not untested 3.8.
# Reuse run_sft.sh's $PIP_INDEX; --trusted-host because that mirror is plain http. Non-fatal (no
# set -e): a mirror hiccup then surfaces truthfully at the (3/3) FLA_STATUS gate below.
PIP_FLA="python3 -m pip install --break-system-packages --index-url $PIP_INDEX --trusted-host mirrors.i.basemind.com"
timeout 1800 $PIP_FLA "flash-linear-attention==0.5.2" "fla-core==0.5.2" einops 2>&1 | tail -6
echo "  fla install rc=${PIPESTATUS[0]}"
timeout 1200 $PIP_FLA -U "triton>=3.7.1,<3.8" 2>&1 | tail -6
echo "  triton upgrade rc=${PIPESTATUS[0]}"
python3 -c "import triton; print('  triton now', triton.__version__)" 2>&1

# ---- 1a. CUDA preflight: make it THROW, do not ask it for a boolean ---------------------------
# `torch.cuda.is_available()` swallows the reason. 5 s here beats discovering it after a 52 GiB load.
WORLD="$WORLD" python3 - <<'PY' || { echo "RESULT: CUDA_PREFLIGHT_FAILED"; exit 2; }
import torch
try:
    torch.cuda.init()
    n = torch.cuda.device_count()
    print("  cuda OK: %d cards, %s, capability %s"
          % (n, torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0)))
    free, tot = torch.cuda.mem_get_info(0)
    print("  card 0: %.1f GiB free of %.1f GiB" % (free / 2**30, tot / 2**30))
    import os; _W = int(os.environ.get("WORLD", "8"))
    assert n >= _W, "expected %d cards, saw %d" % (_W, n)
except Exception as e:
    raise SystemExit("  CUDA PREFLIGHT EXCEPTION: %r" % (e,))
PY

# ---- fla enablement (3/3): truthful FLA_STATUS -- import fla.ops, do NOT trust find_spec --------
# is_flash_linear_attention_available() returns True whenever fla is merely installed. The REAL test
# transformers' kernel dispatcher performs is `import fla.ops` + a callable gated-delta-rule op. This
# MUST print ON_FUSED_GDN, or the 8-card run is silently on the ~26 h fp32 fallback. CUDA is already
# validated by 1a, so torch.zeros(1,'cuda') here just forces the driver/libcuda path fla will use.
# Non-fatal by design (matches deploy/h200_qwen35_body.sh): the stage-3 PROBE's measured per-step
# estimate is the quantitative backstop -- fp32 GDN would show up there as a far larger s/step.
python3 - <<'PY' 2>&1
import triton, torch
print("  triton", triton.__version__, "| torch", torch.__version__)
torch.zeros(1, device="cuda")
real = False; fla_err = ""
try:
    import fla, fla.ops
    real = callable(getattr(fla.ops, "chunk_gated_delta_rule", None))
except Exception as e:
    fla_err = "%s: %s" % (type(e).__name__, str(e)[:140])
print("  FLA_STATUS: " + ("ON_FUSED_GDN" if real else "OFF_FP32_FALLBACK (fla_err=%s)" % fla_err))
PY

# ---- 1b. does `python3 -I -B` (what run_ddp.sh launches every rank with) still see torch? -----
# 🔴 `-I` implies `-E` and `-s`: it discards PYTHONPATH and the user site-packages. If pip fell
# back to a user-site install -- and $HOME is now on /dev/shm, so that would ALSO be volatile --
# then all 8 ranks would start against a different (or absent) torch than the one just verified.
# Rated HIGH risk in RUNTIME_MATRIX.md (R11). Detect it; do not assume it.
if python3 -I -B -c "import torch, transformers" 2>/dev/null; then
  PYFLAGS="-I -B"; echo "  launcher python flags: -I -B (torch visible in isolated mode)"
else
  PYFLAGS="-B"
  export PYTHONPATH="$(python3 -c 'import site,sys; print(site.getusersitepackages())')${PYTHONPATH:+:$PYTHONPATH}"
  echo "  🔴 torch NOT visible under -I; falling back to -B and PYTHONPATH=$PYTHONPATH"
fi
stamp "stage 1 done"

# =============================================================================================
# 2. stage code + data onto /dev/shm
# =============================================================================================
echo "############ 2. stage"
[ -d "$MODEL" ] || { echo "RESULT: MODEL_DIR_NOT_FOUND $MODEL"; exit 2; }
[ -f "$SRC_DATA" ] || { echo "RESULT: DATA_NOT_FOUND $SRC_DATA"; exit 2; }
for f in train_sft_lora_ddp.py run_ddp.sh; do
  cp "$STACK/$f" "$CACHE/code/$f" || { echo "RESULT: STACK_FILE_MISSING $f"; exit 2; }
done
# run_ddp.sh hardcodes `python3 -I -B` for all 8 ranks. Patch it only if 1b said we must, and say so.
if [ "$PYFLAGS" != "-I -B" ]; then
  sed -i "s|python3 -I -B \"\$SCRIPT\"|python3 $PYFLAGS \"\$SCRIPT\"|" "$CACHE/code/run_ddp.sh"
  echo "  patched run_ddp.sh rank launch to: python3 $PYFLAGS"
fi
# 🔴 run_ddp.sh forwards --steps/--accum/--lr/--max-len/--rank16/--alpha/--targets/--chunked-ce
# and NOTHING ELSE (MEASURED: `grep -c EXTRA run_ddp.sh` == 0). So --eval-max, --eval-every and
# --ckpt-every would silently fall back to the trainer's defaults. Those defaults happen to equal
# what we want (64 / 25 / 100), and that is precisely the trap: the run would be correct by
# coincidence and would stop being correct the day a default changes. Teach the launcher to append
# $EXTRA, then ASSERT the patch landed rather than trusting sed.
sed -i 's|--out "$OUT" \\|--out "$OUT" ${EXTRA:-} \\|' "$CACHE/code/run_ddp.sh"
# grep -F, not plain grep: MEASURED, `grep -q -- '${EXTRA:-}'` returns 1 even when the string IS
# present, because BRE eats the `$`/`{}`. A fail-closed assertion built on a broken pattern aborts
# every healthy run instead, which is worse than having no assertion at all.
grep -qF -- '${EXTRA:-}' "$CACHE/code/run_ddp.sh" \
  || { echo "RESULT: LAUNCHER_PATCH_FAILED (EXTRA not forwarded; eval/ckpt flags would be lost)"; exit 2; }
bash -n "$CACHE/code/run_ddp.sh" \
  || { echo "RESULT: LAUNCHER_PATCH_BROKE_SYNTAX"; exit 2; }
echo "  patched run_ddp.sh to forward \$EXTRA to every rank (syntax re-checked)"
# Each of the 8 ranks reads the whole jsonl into python lists. Serving that from JFS 8x is silly,
# and the RAM cost is real: 73.68M tokens x 2 arrays x ~40 B/int ~ 5.9 GiB PER RANK (~47 GiB total).
# Pod must be launched with >= 96Gi of memory. Stage to tmpfs so the read is local.
cp "$SRC_DATA" "$DATA"
echo "  data: $(wc -l < "$DATA") samples, $(du -h "$DATA" | cut -f1)"
MAN="$SRC_DATA.manifest.json"
if [ -f "$MAN" ]; then
  python3 - "$MAN" "$EVAL_MAX" <<'PY'
import json, sys
m = json.load(open(sys.argv[1])); ev = int(sys.argv[2])
es = m["eval_split"]
print("  manifest: src_sha256=%s" % m["src_sha256"][:16])
print("  manifest: %d samples, %.2fM tokens, %.2fM scored (%.1f%%), max_len=%d, thinking=%s"
      % (m["out_lines"], m["tokens_total"] / 1e6, m["tokens_scored"] / 1e6,
         100 * m["scored_fraction"], m["config"]["max_len"], m["config"]["thinking"]))
v = m["verification"]
print("  manifest: mask verified V1 %d/%d render, V3 %d/%d gen-prompt turns, V4 %d/%d decode"
      % (v["v1_ok"], v["checked"], v["v3_turns_ok"], v["v3_turns"], v["v4_turns_ok"], v["v3_turns"]))
print("  manifest: holdout_head=%d, head/tail prompt overlap=%d"
      % (es["holdout_head_records"], es["head_tail_prompt_overlap"]))
assert es["head_tail_prompt_overlap"] == 0, "eval/train share prompt groups"
# 🔴 The trainer's eval set is the FIRST min(int(0.15*N), --eval-max) lines. If --eval-max exceeds
# holdout_head, eval reaches past the reserved block into training prompts and the held-out loss
# silently stops measuring generalisation on this corpus (2,454 prompts / 53,808 records).
assert ev <= es["holdout_head_records"], \
    "EVAL_MAX %d > holdout_head %d: eval would leak into train prompts" % (ev, es["holdout_head_records"])
print("  OK: EVAL_MAX %d <= holdout_head %d" % (ev, es["holdout_head_records"]))
PY
  [ $? -ne 0 ] && { echo "RESULT: DATA_MANIFEST_CHECK_FAILED"; exit 2; }
else
  echo "  ⚠️ no manifest beside the data; eval/train prompt disjointness UNVERIFIED"
fi
stamp "stage 2 done"

# =============================================================================================
# 3. single-card memory probe -- 143 GB is untested territory, and this costs ~3 min
# =============================================================================================
# The trainer is data-parallel with a manual all-reduce of the adapters only; NOTHING is sharded.
# Per-rank memory at batch 1 is therefore IDENTICAL on 1 card and on 8, so one card answers
# "does max_len fit" before 8 cards are committed for hours.
if [ "$PROBE" = "1" ]; then
  echo "############ 3. single-card fit probe at max_len $MAXLEN"
  CUDA_VISIBLE_DEVICES=0 python3 $PYFLAGS - "$MODEL" "$DATA" "$MAXLEN" "$CHUNKED_CE" "$LORA_R" "$LORA_ALPHA" "$TARGETS" <<'PY'
import json, sys, time, torch, transformers
sys.argv[0] = "probe"
model_dir, data, maxlen, chunk, r, alpha, targets = sys.argv[1:8]
maxlen, chunk, r, alpha = int(maxlen), int(chunk), int(r), int(alpha)
sys.path.insert(0, __import__("os").environ["CACHE"] + "/code")
from train_sft_lora_ddp import inject_lora, chunked_ce, batch, enable_param_offload, resolve_decoder_layers
t0 = time.perf_counter()
m = transformers.Qwen3_5ForConditionalGeneration.from_pretrained(
    model_dir, dtype=torch.bfloat16, low_cpu_mem_usage=True).to("cuda:0")
print("  loaded %.1fs  %.2f GiB" % (time.perf_counter() - t0, torch.cuda.memory_allocated() / 2**30))
n_inj, n_tr = inject_lora(m, targets.split(","), r, alpha)
print("  LoRA: %d modules, %d trainable params (%.2f%% of 27.36e9)" % (n_inj, n_tr, 100 * n_tr / 27.36e9))
if n_inj == 0:
    raise SystemExit("RESULT: NO_LORA_TARGETS_MATCHED")
# how many of the 64 decoder layers actually got an adapter -- the whole point of note 3
import re
layers = set()
for name, mod in m.named_modules():
    if type(mod).__name__ == "LoRALinear":
        g = re.search(r"layers\.(\d+)\.", name)
        if g:
            layers.add(int(g.group(1)))
print("  adapters cover %d distinct decoder layers (must be 64, not 16)" % len(layers))
m.gradient_checkpointing_enable(); m.config.use_cache = False; m.train()
def _gpu(): return torch.cuda.memory_allocated() / 2**30, torch.cuda.memory_reserved() / 2**30
print("  [mem] after grad-ckpt/train: alloc %.2f resv %.2f GiB" % _gpu())
if __import__("os").environ.get("PARAM_OFFLOAD", "0") == "1":
    nl, nb = enable_param_offload(m, "cuda:0")
    print("  [mem] after enable_param_offload: alloc %.2f resv %.2f GiB (offload claims %.2f GiB over %d layers)"
          % (_gpu() + (nb / 2**30, nl)))
    torch.cuda.empty_cache()
    print("  [mem] after empty_cache: alloc %.2f resv %.2f GiB  <-- if alloc still ~48+, offload did NOT free GPU weights"
          % _gpu())
    # what is still resident? embeddings / vision / norm / lm_head (outside the decoder layers)
    res = {}
    for nm, p in m.named_parameters():
        if p.device.type == "cuda" and not p.requires_grad:
            top = nm.split(".")[0] if "." in nm else nm
            # bucket by the module just under the top
            parts = nm.split(".")
            key = parts[0] + ("." + parts[1] if len(parts) > 1 else "")
            res[key] = res.get(key, 0) + p.numel() * p.element_size()
    for k, v in sorted(res.items(), key=lambda kv: -kv[1])[:8]:
        print("      resident frozen: %-40s %.2f GiB" % (k, v / 2**30))
# longest real sample, padded up to max_len -- the worst case the run will actually see
best = None
for line in open(data):
    d = json.loads(line)
    if len(d["input_ids"]) <= maxlen and (best is None or len(d["input_ids"]) > len(best[0])):
        best = (d["input_ids"], d["labels"])
print("  worst-case sample: %d tokens" % len(best[0]))
# per-decoder-layer memory trace hooks (fire during forward; dumped on OOM too)
_dl, _ = resolve_decoder_layers(m)
_trace = []
def _mk(i):
    def _h(mod, inp, out):
        _trace.append((i, torch.cuda.memory_allocated() / 2**30))
    return _h
_hs = [_dl[i].register_forward_hook(_mk(i)) for i in range(len(_dl))]
def _dump_trace(tag):
    for h in _hs: h.remove()
    if _trace:
        mx = max(a for _, a in _trace)
        print("  [mem %s] per-layer alloc at 0/8/.../%d, max single-layer=%.2f GiB:"
              % (tag, len(_dl) - 1, mx))
        for i, a in _trace:
            if i % 8 == 0 or i == len(_dl) - 1 or a == mx:
                print("      layer %2d: %.2f%s" % (i, a, "  <== PEAK" if a == mx else ""))
# SWEEP max_len on the SAME offloaded model: find the largest ctx that fits one H200.
free0, tot = torch.cuda.mem_get_info(0)
print("  [sweep] card %.1f GiB, weights resident ~%.2f GiB; trying lengths:" % (tot / 2**30, torch.cuda.memory_allocated() / 2**30))
sweep = [32768, 65536, 131072, 163840, 196608, 229376, maxlen]
sweep = sorted(set(min(L, len(best[0])) for L in sweep))
fits = None
for L in sweep:
    if L > len(best[0]): continue
    _trace.clear()
    ids, lab, att = batch([(best[0][:L], best[1][:L])], [0], 0, "cuda:0")
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    try:
        loss = chunked_ce(m, ids, lab, att, chunk, do_backward=True)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        peak = torch.cuda.max_memory_allocated() / 2**30
        resv = torch.cuda.max_memory_reserved() / 2**30
        print("  [sweep] L=%7d  OK   fwd+bwd %6.1fs  loss %.4f  peak_alloc %6.2f  peak_resv %6.2f GiB"
              % (L, dt, loss, peak, resv))
        fits = (L, peak, resv, dt)
        del ids, lab, att, loss
    except torch.OutOfMemoryError as e:
        print("  [sweep] L=%7d  OOM  (peak ~%.2f GiB before fail)  <-- ceiling" %
              (L, torch.cuda.max_memory_allocated() / 2**30))
        _dump_trace("OOM@%d" % L)
        break
    torch.cuda.empty_cache()
print("  SWEEP_RESULT largest_fitting_max_len=%s" % (fits[0] if fits else "none"))
if fits is None:
    print("RESULT: NO_LENGTH_FITS")
    sys.exit(0)
L, peak, resv, dt = fits
free, tot = torch.cuda.mem_get_info(0)
print("  fwd+bwd %.2fs  loss-set  peak_alloc %.2f GiB  peak_reserved %.2f GiB  card %.1f GiB"
      % (dt, peak, resv, tot / 2**30))
print("  DERIVED per-step estimate at accum 2: %.1f s/step, %.0f aggregate tok/s over 8 ranks"
      % (2 * dt, 8 * 2 * L / (2 * dt)))
if resv > 0.85 * tot / 2**30:
    print("  🔴 peak_reserved is >85%% of the card: lower MAXLEN or CHUNKED_CE before the 8-card run")
print("RESULT: PROBE_OK")
PY
  rc=$?; [ $rc -ne 0 ] && { echo "RESULT: PROBE_FAILED rc=$rc"; exit 3; }
  stamp "stage 3 done"
fi

# =============================================================================================
# 4. 8-RANK DATA-PARALLEL LoRA SFT
# =============================================================================================
# ARITHMETIC (MEASURED data, DERIVED schedule -- full version in SFT_PLAN.md §4):
#   11,100 packed samples; 64 reserved as held-out => 11,036 train.
#   length bucketing: 11,036 // 8 = 1,379 blocks of 8, so 1,379 samples/rank (4 dropped).
#   steps/epoch = 1,379 / accum 2 = 690.  STEPS 1400 => 2.03 epochs.
#   warmup = 10% of STEPS = 140 steps (the trainer derives it from --steps; changing STEPS
#   silently changes the LR schedule, which is why it is written down here).
#   mean 6,637 tok/sample => 16 x 6,637 = 106,192 ctx tok/step => 148.7M tokens for the run.
# 🔴 Every flag the launcher does not forward is a flag the run does not get: CHUNKED_CE and
# LORA_R were both silently dropped once in this project's history, which turned one labelled
# experiment into a rerun of the previous one. run_ddp.sh forwards these two; --eval-max,
# --eval-every and --ckpt-every it does NOT, so they are appended via EXTRA below.
echo "############ 4. 8-rank data-parallel LoRA SFT"
echo "  targets=$TARGETS"
echo "  steps=$STEPS accum=$ACCUM (global batch $((WORLD*ACCUM))) max_len=$MAXLEN chunked_ce=$CHUNKED_CE"
t0=$(date +%s)
MODEL="$MODEL" DATA="$DATA" WORLD="$WORLD" STEPS="$STEPS" ACCUM="$ACCUM" LR="$LR" \
  LORA_R="$LORA_R" LORA_ALPHA="$LORA_ALPHA" TARGETS="$TARGETS" \
  MAXLEN="$MAXLEN" CHUNKED_CE="$CHUNKED_CE" CHUNK_PROOF_LEN="$CHUNK_PROOF_LEN" \
  OUT="$OUT" SCRIPT="$CACHE/code/train_sft_lora_ddp.py" LOGDIR="$LOGDIR" \
  EXTRA="--eval-max $EVAL_MAX --eval-every $EVAL_EVERY --ckpt-every $CKPT_EVERY" \
  bash "$CACHE/code/run_ddp.sh" 2>&1 | tail -1200
echo "  DDP_SECONDS=$(( $(date +%s) - t0 ))"
stamp "stage 4 done"

# =============================================================================================
# 5. artefacts -- 🔴 14 prior runs in this project lost every adapter they trained
# =============================================================================================
# `find` over the whole rsi_2608 tree returns ZERO .pt files: all 14 completed or partial runs
# lost their weights, so no evaluation defined later can ever be run against any of them. The
# trainer now writes lora_adapters_latest.pt every --ckpt-every steps AND keeps the eval-minimum
# checkpoint. Copying them OFF tmpfs is this script's job: $CACHE is /dev/shm and dies with the pod.
echo "############ 5. artefacts"
ls -la "$OUT" 2>/dev/null
PERSIST="${PERSIST:-/mnt/ws-jfs/posttrain/i-xuyiming/paper/data_select_2609/sft/ckpt/$(date +%Y%m%d-%H%M%S)}"
mkdir -p "$PERSIST" && cp -v "$OUT"/*.pt "$OUT"/*.json "$OUT"/progress.jsonl "$PERSIST"/ 2>/dev/null
cp -v "$LOGDIR"/rank0.log "$PERSIST"/ 2>/dev/null
echo "  persisted -> $PERSIST"
ls -la "$PERSIST" 2>/dev/null
echo "--- per-rank GPU state (proves all 8 cards did work) ---"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits
echo "--- progress.jsonl (one line per step; the loss curve lives here, not in stdout) ---"
wc -l "$OUT/progress.jsonl" 2>/dev/null
echo "############ DONE total=$(( $(date +%s) - T00 ))s"
