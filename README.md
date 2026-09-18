# deepseek_fulllen_sft — Qwen3.8-27B × glm-5.3-deepseek FULL-DATA LoRA SFT on H200

Self-contained training code for the owner-directed experiment: **SFT Qwen3.8-27B on the FULL
glm-5.3-deepseek trajectory corpus (no filtering), at the longest context that fits one H200.**

---

## What this trains

| | |
|---|---|
| **Model** | Qwen3.8-27B (`rsi_2608/qwen-3.8-27B/model/Qwen3.8-27B`), GDN hybrid (16 full-attn + 48 linear-attn layers) |
| **Data** | `glm-5.3-deepseek` FULL corpus — 3,104 raw trajectories → 3,069 after dropping 35 exotic-Unicode records (owner: "这种 exotic 的直接删掉") → packed to **3,519 pieces** at MAXLEN=131072, `max_per_prompt=0` (NOTHING filtered), `overlength=split` (no truncation, all content kept) |
| **Method** | LoRA r=8 / α=16, targets incl GDN `in_proj_qkv,in_proj_z,out_proj` (covers all 64 layers) |
| **Length** | **MAXLEN=131072** — the largest single-card length on H200 (see "Why not 262144" below) |
| **Schedule** | ep1 = 432 steps (WORLD=4 × ACCUM=2 = global_batch 8; 3455 train / 8) |
| **Precision / kernels** | bf16 + **fla fused GDN** (libcuda→/dev/shm/cudalink + triton≥3.7.1 + torch 2.8.0) |
| **Offload** | `PARAM_OFFLOAD=1` (frozen base weights → CPU, streamed per layer) + `OFFLOAD_ACT=1` (save_on_cpu) — BOTH required at 131072 |
| **Hardware** | 4×H200 (coding quota, name=`test`), **MEM_MB=700000 (must land on a ≥683 GB node)** |

---

## Files

```
run_sft_fla.sh              # the rjob BODY: install torch2.8/fla/triton, libcuda preflight, fit-probe, launch DDP
rjob.sh                     # rjob submitter (name=test, coding, GPUS/MEM_MB/CPUS as ENV before the command)
trainer/train_sft_lora_ddp.py   # THE trainer actually used (LoRA DDP + PARAM_OFFLOAD + mask/chunk proofs)
trainer/run_ddp.sh          # launches N explicit `python3 -B` ranks (no torchrun)
data_prep/pack_traj.py      # chat.jsonl → token-level SFT pieces (assistant-only mask, split, holdout)
data_prep/filter_exotic_unicode.py   # drops records that trip pack_traj's strict V4 decode-back (exotic Unicode)
configs/launch.sh           # ready-to-run: the exact successful launch command
configs/packed_131k.manifest.json  # the packed-data manifest (counts, tokenizer sha, verification)
```

Data (1.7 GB, NOT bundled — already on JFS): `sft/packed_glm53ds_131k/train.jsonl` (and the clean source
`sft/raw/glm-5.3-deepseek/chat_clean.jsonl`). Repack with:
```
python3 packenv/bin/python data_prep/pack_traj.py \
  --src sft/raw/glm-5.3-deepseek/chat_clean.jsonl \
  --out sft/packed_glm53ds_131k/train.jsonl \
  --tokenizer <qwen3.8-27B> --max-len 131072 --thinking off --overlength split \
  --max-per-prompt 0 --holdout-head 64 --shuffle-seed 1234 --workers 16 --verify 32
```

---

## Run it

```bash
bash configs/launch.sh          # submits rjob name=test, 4×H200, 700GB, ep1
brainctl get rjob test -n shai-core
brainctl exec replica/test-<hash> -n shai-core -- tail -6 /dev/shm/xym_sft2609/logs/rank0.log
```
Output/persist dir: `sft/ckpt_glm53ds_h200_131k/` → `lora_adapters_best.pt` (eval-min = non-overfit ckpt),
`lora_adapters.pt` (final), `train_result.json`, `rank0.log`, `progress.jsonl`.

---

## ⚠️ Why NOT full-length 262144 (measured, not assumed)

Owner wanted single-sequence 262144. It is **physically impossible on one H200**. A multi-length sweep on
the offloaded model:

| max_len | peak_alloc | peak_resv | result |
|---|---|---|---|
| 32768 | 68.5 GiB | 71.5 | OK |
| 65536 | 85.9 | 91.9 | OK |
| **131072** | **120.6** | **132.5** | **OK ← ceiling** |
| 163840 | — | — | OOM (~7 GiB over) |
| 262144 | 138.4 (+8.5 wanted ≈ 147) | — | OOM |

CPU offload frees the ~45 GiB of frozen weights (GPU weights drop to 5.67 GiB), but the **live per-layer
transient activation at 262144 (~147 GiB) exceeds the 139.8 GiB card** — offload can only move *saved*
tensors, not the ones being computed right now. To truly fit 262144 you need **sequence parallelism across
cards** (each card handles 1/N of the sequence) or a bigger card (B300 275 GB — but owner said leave B300
for others). 131072 is the largest single-card length; records >131072 are split into ≤131072 windows
(task-header re-stated per window, no content loss).

---

## 🔧 Fixes baked into this code (each was a real failure, 2026-09-16/17)

1. **MASK-PROOF OOM at 131072** — the mask proof ran `model(labels=lab)` on the FULL-length sample → HF
   materialises `[B,T,vocab]` fp32 logits ≈130 GiB → OOM. **Fix**: slice the mask-proof inputs to
   `min(len, chunk_proof_len=8192)` (the proof is length-independent). In `trainer/train_sft_lora_ddp.py`
   PROOF 1. (The chunk proof already did this.)
2. **host-RAM OOM (code 137), the big one** — 4 ranks × (86 GiB activation-offload + 48 GiB weight-offload)
   ≈ 536 GiB of CPU RAM needed, but the pod **landed on a 390 GB node** (I'd requested only 488 GB) →
   kernel OOM-killed a rank after ~3 steps. GPU was half-empty (104/140) the whole time. **Fix: request
   MEM_MB=700000 so the scheduler MUST place it on a ≥683 GB node.** H200 nodes vary (saw 390 GB and 683 GB);
   the memory request is also a node-size filter. (An 8-GPU 1.5 TB request Pending'd — few nodes have that
   much; 700 GB is in the schedulable range.)
3. **rjob.sh GPUS is an ENV var, not a body arg** — passing `GPUS=8` after the command silently requests
   1 GPU. Control vars (GPUS/MEM_MB/CPUS/GROUP/TAGS) go BEFORE `bash rjob.sh submit`; training vars after.
4. **run_sft_fla copies the trainer from `$STACK`, not deploy/** — the offload + proof fixes must live in the
   `$STACK` trainer (here `sft/stack_glm53ds_offload/`), which is what `trainer/` in this package mirrors.
5. **"Succeeded" is a lie on crash** — run_ddp.sh exits and the body persists even when a rank is OOM-killed,
   so the rjob reads `Succeeded` after 3 steps. **Always check `progress.jsonl` line-count vs total_steps +
   grep the submit log for `code=137`/`LAUNCH_FAILED`.**

---

## Status (2026-09-17)

Training **stable** on a 683 GB node: ~245/432 steps, held-out eval loss falling 0.261→0.234 (learning, not
overfit), peak GPU ~124 GiB, host RAM ~300/683 GiB, zero errors. ep1 ETA ~8h from step 245. best.pt = the
eval-minimum (non-overfit) checkpoint, per the owner's "挑一个不过拟合的 ckpt" directive.

Full decision log: `../docs/qwen38_gpt56/RESEARCH_LOG.md`.
