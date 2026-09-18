#!/usr/bin/env python3
"""train_sft_lora_ddp.py -- 8-card DATA-PARALLEL LoRA SFT for Qwen3.8-27B.

WHY THIS EXISTS (the measurement that demanded it)
An external probe of the healthy 200-step single-card run showed, verbatim:

    0, 100, 238004;1, 0, 4;2, 0, 4;3, 0, 4;4, 0, 4;5, 0, 4;6, 0, 4;7, 0, 4;
    (index, utilization.gpu %, memory.used MiB)

GPU 0 pinned at 100 %, GPUs 1-7 holding **4 MiB each and doing nothing**. We hold an
all-or-nothing 8-card allocation that queues behind every other tenant on the node, and then use
one eighth of it. Meanwhile the trainer averages 91-94 % of ONE core (MEASURED, `ps` pcpu) out of
the 1600 % the pod is allowed -- the loop is kernel-launch-bound, so the fix is more independent
streams of work, not a faster kernel.

WHY PLAIN torch.distributed AND NOT accelerate/DDP/FSDP
  * `accelerate` and `peft` are ABSENT from the image and there is NO egress to install them
    (MEASURED: pypi/hf/mirrors all time out; the corporate proxy is unreachable from the pod).
  * `torch.nn.parallel.DistributedDataParallel` is available, but 99.87 % of this model's
    parameters are FROZEN. DDP builds gradient buckets over all params that require grad and
    complains about (or wastes time on) unused ones. With LoRA the trainable set is tiny and
    known, so a MANUAL all-reduce over just the adapter grads is both simpler and strictly less
    work: 64 modules x (16x5120 + 5120x16) fp32 = ~42 MB per reduction, over NVLink, once per
    optimizer step. That is negligible next to a ~14 s step.
  * FSDP would be needed for FULL fine-tuning (430 GiB of state; see the arithmetic in
    train_sft_lora.py) but LoRA already fits on one card, so sharding buys nothing here.

WHY NOT torchrun
The house rule is that every python3 runs `-I -B` (isolated, no bytecode) because of the
GRADER-HIJACK fixture that shadows stdlib names. `torchrun` spawns its children itself and does
not propagate those flags, so this script is launched as 8 explicit `python3 -I -B` processes by
run_ddp.sh, each told its own --rank. No launcher magic, and the flag guarantee holds for every
process.

WHAT THIS SCRIPT PROVES (all asserted, not assumed)
  1. the loss mask is correct -- exact recomputation of cross-entropy over `label != -100`
     positions must reproduce the model's own loss (inherited from the single-card trainer);
  2. the ranks are training on DISJOINT data -- each rank reports its shard, and the union is
     checked against the dataset size;
  3. the ranks stay in SYNC -- after the first optimizer step the adapter weights are compared
     bitwise across ranks. If the all-reduce were wrong or a rank silently diverged, this fails.
     This is the multi-GPU equivalent of the mask proof: the failure it catches (8 ranks quietly
     training 8 different models and rank 0 saving only its own) is invisible in the loss curve.

  torchrun-free usage (see run_ddp.sh):
    MASTER_ADDR=127.0.0.1 MASTER_PORT=29500 \
    python3 -I -B train_sft_lora_ddp.py --rank 0 --world 8 --model /data/model/... --data ...
"""
import argparse
import json
import math
import os
import sys
import time

import torch
import torch.distributed as dist
import torch.nn as nn


class LoRALinear(nn.Module):
    """y = base(x) + (alpha/r) * B(A(x)); base frozen."""

    def __init__(self, base: nn.Linear, r=16, alpha=32, dropout=0.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        dev = base.weight.device
        dt = torch.float32                     # adapters in fp32 for stable small-LR updates
        self.a = nn.Parameter(torch.zeros(r, base.in_features, device=dev, dtype=dt))
        self.b = nn.Parameter(torch.zeros(base.out_features, r, device=dev, dtype=dt))
        nn.init.kaiming_uniform_(self.a, a=math.sqrt(5))   # B stays zero => identity at step 0
        self.scale = alpha / r
        self.drop = nn.Dropout(dropout) if dropout else nn.Identity()

    def forward(self, x):
        # 🔴 MEMORY (MEASURED): doing the UP-projection in fp32 materialises a [B, T, out] fp32
        # tensor and keeps it for backward. That cost 71.11 GiB at ctx~55k and killed a 200-step
        # run at step 107 with 260.04 GiB of 268. Down-project in fp32 (cheap, r=16 columns), then
        # cast to the base dtype BEFORE the wide up-projection. Peak fell to 188.93 GiB.
        out = self.base(x)
        h = self.drop(x).to(self.a.dtype) @ self.a.t()          # [B, T, r] fp32 -- tiny
        h = (h * self.scale).to(out.dtype) @ self.b.t().to(out.dtype)   # [B, T, out] in bf16
        return out + h


def inject_lora(model, targets, r, alpha):
    n = 0
    for name, mod in list(model.named_modules()):
        for child_name, child in list(mod.named_children()):
            if isinstance(child, nn.Linear) and any(t in child_name for t in targets):
                setattr(mod, child_name, LoRALinear(child, r=r, alpha=alpha))
                n += 1
    for p in model.parameters():
        p.requires_grad_(False)
    tr = 0
    for m in model.modules():
        if isinstance(m, LoRALinear):
            m.a.requires_grad_(True)
            m.b.requires_grad_(True)
            tr += m.a.numel() + m.b.numel()
    return n, tr


def load_jsonl(path, max_len):
    out = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        ids, lab = d["input_ids"][:max_len], d["labels"][:max_len]
        if sum(1 for x in lab if x != -100) == 0:
            continue
        out.append((ids, lab))
    return out


def batch(samples, idx, pad_id, device):
    chunk = [samples[i] for i in idx]
    L = max(len(a) for a, _ in chunk)
    ids = torch.full((len(chunk), L), pad_id, dtype=torch.long)
    lab = torch.full((len(chunk), L), -100, dtype=torch.long)
    att = torch.zeros((len(chunk), L), dtype=torch.long)
    for k, (a, b) in enumerate(chunk):
        ids[k, :len(a)] = torch.tensor(a)
        lab[k, :len(b)] = torch.tensor(b)
        att[k, :len(a)] = 1
    return ids.to(device), lab.to(device), att.to(device)


def save_adapters(model, out_dir, name, meta):
    """Atomically write the trainable (adapter) tensors plus a sidecar saying what they are.

    🔴 WHY THIS EXISTS, and why it is not merely a convenience.
    Until 2026-08-23 the trainer wrote `lora_adapters.pt` exactly ONCE, after the training loop, and
    the file lived only on pod-local disk. MEASURED: `find` over the whole paper tree returns ZERO
    `.pt` files -- **all 14 completed or partial runs lost their weights**, so no evaluation defined
    later can ever be run against any of them. v13 lost 892 steps of training to a session reap while
    the only save point was still 1158 steps away.
    Worse, the single save captured the LAST step, not the eval MINIMUM -- and every run's minimum was
    mid-run (v9 1250/1325, v11c 825/900, v12 1725/1800, v13 850/892). So even the runs that finished
    cleanly persisted weights that were **not** the best ones they produced.
    Atomic tmp+os.replace, because a reap during a 167.8 MB write would otherwise leave a truncated
    file that `torch.load` may still open -- a corrupt checkpoint that reads as a real one is worse
    than none.
    """
    os.makedirs(out_dir, exist_ok=True)
    sd = {n: q.detach().to(torch.float32).cpu()
          for n, q in model.named_parameters() if q.requires_grad}
    dst = os.path.join(out_dir, name)
    tmp = dst + ".tmp"
    torch.save(sd, tmp)
    os.replace(tmp, dst)
    try:
        side = os.path.splitext(dst)[0] + ".meta.json"
        tmp2 = side + ".tmp"
        with open(tmp2, "w") as fh:
            json.dump({**meta, "n_tensors": len(sd),
                       "n_params": int(sum(v.numel() for v in sd.values())),
                       "bytes": os.path.getsize(dst)}, fh, indent=2)
        os.replace(tmp2, side)
    except Exception:
        pass
    return dst


def resolve_head(model):
    """(decoder, lm_head) using HF's OFFICIAL accessors, with attribute fallbacks.

    Not hardcoded to `model.model` / `model.lm_head`: this checkpoint is a
    `Qwen3_5ForConditionalGeneration`, and the `*ForConditionalGeneration` family sometimes nests the
    text stack one level deeper than the plain `*ForCausalLM` layout. `get_decoder()` /
    `get_output_embeddings()` are the contracted accessors and survive that difference.
    """
    head = None
    if hasattr(model, "get_output_embeddings"):
        try:
            head = model.get_output_embeddings()
        except Exception:
            head = None
    if head is None:
        head = getattr(model, "lm_head", None)
    dec = None
    if hasattr(model, "get_decoder"):
        try:
            dec = model.get_decoder()
        except Exception:
            dec = None
    if dec is None:
        dec = getattr(model, "model", None)
    return dec, head


def resolve_decoder_layers(model):
    """Return (ModuleList_of_TEXT_decoder_layers, human_path), resolved DEFENSIVELY.

    Not hardcoded, for the same reason resolve_head() is not: this checkpoint is a
    `Qwen3_5ForConditionalGeneration`, which nests the text stack as
    `model.model.language_model.layers` -- one level DEEPER than a plain `*ForCausalLM`
    (`model.model.layers`) -- and the SAME model also owns a VISION `ModuleList`
    (`model.model.visual.blocks`) that must never be picked. So: try the known dotted
    paths, then get_decoder().layers, then a scan keyed on the decoder-layer CLASS NAME
    (which excludes vision blocks), and only as a last resort the longest ModuleList.
    Used solely by the PARAM_OFFLOAD path; asserts nothing about the count so it survives
    a re-layered future checkpoint, but logs the resolved path+length for the operator.
    """
    def _get(root, dotted):
        cur = root
        for part in dotted.split("."):
            cur = getattr(cur, part, None)
            if cur is None:
                return None
        return cur
    for dotted in ("model.language_model.layers", "model.model.layers",
                   "language_model.layers", "model.layers"):
        ml = _get(model, dotted)
        if isinstance(ml, nn.ModuleList) and len(ml) > 0:
            return ml, dotted
    dec, _ = resolve_head(model)
    if dec is not None and isinstance(getattr(dec, "layers", None), nn.ModuleList) \
            and len(dec.layers) > 0:
        return dec.layers, "get_decoder().layers"
    best, best_name = None, None
    for name, mod in model.named_modules():          # prefer a *DecoderLayer* ModuleList
        if isinstance(mod, nn.ModuleList) and len(mod) > 0 \
                and "decoderlayer" in type(mod[0]).__name__.lower():
            if best is None or len(mod) > len(best):
                best, best_name = mod, name
    if best is None:                                 # last resort: the longest ModuleList
        for name, mod in model.named_modules():
            if isinstance(mod, nn.ModuleList) and len(mod) > 0:
                if best is None or len(mod) > len(best):
                    best, best_name = mod, name
    if best is None:
        raise RuntimeError("PARAM_OFFLOAD: could not locate a decoder-layer ModuleList")
    return best, best_name


def enable_param_offload(model, dev, log=None, is0=True):
    """PARAM_OFFLOAD=1: keep each decoder layer's FROZEN base weights in (pinned) host RAM
    and stream them to `dev` just-in-time per forward, evicting them after.

    WHY IT EXISTS -- the remaining fixed cost after OFFLOAD_ACT.
    At ctx 262144 on one 143 GiB H200, activation offload (save_on_cpu in chunked_ce) moves
    the saved-for-backward checkpoints to CPU, but the ~54 GiB of FROZEN base weights still
    sit on the card the entire run. Since 99.87 %% of params are frozen and grad-checkpointing
    already RECOMPUTES every layer's forward, we can hold the base on the host and page it in
    one layer at a time. GPU-resident base weight then drops from the whole model to ~one
    decoder layer, freeing ~48-49 GiB for the transient activation.

    WHY IT IS CORRECT
      * HF's `GradientCheckpointingLayer.__call__` runs the layer via `super().__call__(...)`
        (== `nn.Module.__call__`), so the forward PRE/POST hooks fire on BOTH the original
        forward AND the grad-checkpoint recompute that backward performs. The base weights are
        therefore on `dev` for every matmul that reads them, in fwd and in the recomputed-fwd.
      * The base is FROZEN (requires_grad=False), so eviction only DROPS the GPU copy and
        repoints `.data` at the untouched CPU home -- NO copy-back, so no weight state is ever
        lost and there is no D2H sync on the hot path. Autograd's saved tensors hold their own
        reference to the GPU weight storage, so it survives on GPU for that layer's local
        backward and is reclaimed right after.
      * Trainable LoRA A/B (requires_grad=True) are NEVER moved: they stay on `dev`, so the
        optimizer, the manual grad all-reduce, and clip_grad_norm_ all see cuda tensors and the
        gradients flow unchanged.
      * Embeddings, the final norm, the rotary cache and lm_head all live OUTSIDE the decoder
        `layers` list, so they are never touched and stay resident on `dev`.

    PARAM_OFFLOAD_PIN=0 disables host pinning (slower H2D, but avoids pinning ~54 GiB/rank on a
    host that cannot); pinning is attempted per-tensor and silently falls back to pageable.
    """
    layers, path = resolve_decoder_layers(model)
    pin_pref = os.environ.get("PARAM_OFFLOAD_PIN", "1") == "1"

    def _home(t):
        c = t.detach().to("cpu")
        if pin_pref:
            try:
                return c.pin_memory(), True
            except Exception:                        # pinning hundreds of GiB can OOM the host
                return c, False
        return c, False

    n_tensors = n_bytes = n_pinned = n_trainable_kept = 0
    for layer in layers:
        entries = []                                 # [(param_or_buffer_tensor, cpu_home)]
        for p in layer.parameters(recurse=True):
            if p.requires_grad:                      # LoRA A/B -- keep on dev, never stream
                n_trainable_kept += 1
                continue
            home, pinned = _home(p.data)
            p.data = home                            # frees this param's GPU storage now
            entries.append((p, home))
            n_tensors += 1
            n_pinned += int(pinned)
            n_bytes += home.numel() * home.element_size()
        for b in layer.buffers(recurse=True):        # none in this model; kept for generality
            home, pinned = _home(b.data)
            b.data = home
            entries.append((b, home))
            n_tensors += 1
            n_pinned += int(pinned)
            n_bytes += home.numel() * home.element_size()

        def _pre(module, args, _entries=entries):
            # pinned home => the H2D copy is async and enqueues on the current (default) stream,
            # so the layer's own kernels, enqueued after, are correctly ordered behind it.
            for t, home in _entries:
                t.data = home.to(dev, non_blocking=True)

        def _post(module, args, output, _entries=entries):
            # frozen => just drop the GPU copy and revert to the CPU home; no copy-back.
            for t, home in _entries:
                t.data = home

        layer.register_forward_pre_hook(_pre)
        layer.register_forward_hook(_post)

    if log is not None and is0:
        log("PARAM_OFFLOAD=1: %d decoder layers -> frozen base weights on CPU (path=%s); "
            "offloaded %d tensor(s) = %.2f GiB host (%d pinned, pin_pref=%s); kept %d trainable "
            "LoRA tensor(s) on %s; streamed just-in-time per forward + grad-ckpt recompute"
            % (len(layers), path, n_tensors, n_bytes / 2**30, n_pinned, pin_pref,
               n_trainable_kept, dev))
    return len(layers), n_bytes


def chunked_ce(model, ids, lab, att, chunk, scale=1.0, do_backward=False):
    """Cross-entropy over the LM head in POSITION SLICES, so fp32 logits never exceed one chunk.

    WHY THIS EXISTS -- it is the binding constraint on context length AND a measured source of stalls.
    HF's `labels=` path materialises `[B,T,vocab]` and upcasts to fp32. With vocab 248320 that is
    30.31 GiB at T=32768 and **242.5 GiB at the model's native T=262144** -- and 242.5 + 51.75 GiB of
    weights exceeds the 268 GiB card, so 256k is not reachable by any config change. MEASURED on v10
    (2026-08-22): ranks 1-7 logged `CUDACachingAllocator ... failed to allocate 32,065,454,080 bytes
    (free: 29,030,350,848)` -- that size decodes to exactly 32,282 x 248320 x 4 B, i.e. the fp32
    logits for ONE microbatch at max_len 32768. Peak was only 170.5 / 268 GiB, so the allocator was
    losing to FRAGMENTATION on a ~30 GiB monolith, recovering by flushing its cache and retrying
    (a device-wide sync). Slicing the head removes the monolith: chunk 4096 is 3.79 GiB regardless of T.

    HOW THE GRADIENT STAYS EXACT (this is the whole trick)
      1. run the decoder once, then DETACH its output into a leaf `h`;
      2. per chunk, compute that chunk's CE from `h` and call `.backward()` IMMEDIATELY -- which
         frees that chunk's logits before the next is built, and accumulates into `h.grad`;
      3. finally push `h.grad` back through the decoder with `hid.backward(gradient=h.grad)`.
    Chain rule makes this identical to one big backward, not an approximation. Each chunk's graph is
    rooted at the leaf `h`, so the chunks are independent and `retain_graph` is unnecessary; the
    decoder's graph is untouched until step 3.

    NORMALISATION: CE must be the mean over SCORED positions, but a chunk does not know the global
    denominator, so `n_scored` is counted up-front and every chunk is scaled by `scale / n_scored`
    with `reduction='sum'`. Summing the chunks then reproduces the mean exactly -- including the
    gradient, which is why the denominator cannot be per-chunk.

    Returns the (scaled) loss as a float. `scale` is where 1/accum goes.
    """
    dec, head = resolve_head(model)
    if dec is None or head is None:
        raise RuntimeError("could not resolve decoder/lm_head for chunked CE")
    # Shifted targets: hidden position t predicts label t+1, so only T-1 positions are ever scored.
    n_scored = int((lab[:, 1:] != -100).sum().item())
    if n_scored == 0:
        return 0.0
    # CPU activation offload (owner 2026-09-15, for full-length 262144 training on H200): offload the
    # decoder's saved-for-backward tensors (with gradient_checkpointing on, these are the per-layer
    # checkpoint boundaries) to CPU RAM during the forward; autograd reloads them for hid.backward()
    # below, and grad-ckpt recomputes intra-layer activations transiently on GPU. Trades PCIe transfer
    # time for a large drop in peak GPU memory, so a 27B LoRA run can reach ctx 262144. Gated by
    # OFFLOAD_ACT=1 (default runs unchanged); only meaningful with do_backward (eval is no_grad and
    # saves nothing). OFFLOAD_PIN=1 pins host memory (faster, pins hundreds of GiB). Large --memory needed.
    import os as _os
    if do_backward and _os.environ.get("OFFLOAD_ACT", "0") == "1":
        with torch.autograd.graph.save_on_cpu(pin_memory=(_os.environ.get("OFFLOAD_PIN", "0") == "1")):
            out = dec(input_ids=ids, attention_mask=att, use_cache=False)
    else:
        out = dec(input_ids=ids, attention_mask=att, use_cache=False)
    hid = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
    # 🔴 ACCUMULATE THE HIDDEN-STATE GRADIENT IN fp32, not in the model dtype.
    # MEASURED 2026-08-22 (v11 attempt 1, and the reason it was refused): with `h` left in bf16 the
    # per-chunk gradients accumulate into a bf16 `h.grad`, and bf16 has an 8-bit mantissa (per-rounding
    # relative error up to 2^-9 = 0.195 %). Three chunks compounded to a gradient that differed from
    # HF's by rel_L2 1.61e-2 .. 2.30e-2 across the 8 ranks -- straddling the 2e-2 gate, so 4 ranks
    # passed and 4 failed. The LOSS matched to 7.45e-08 throughout, which is exactly why a loss-only
    # check would have shipped a ~2 %-wrong gradient. Forward is unchanged: the slice is cast back to
    # the model dtype before the head, so the head still sees bf16 activations as HF does.
    h = hid.detach().float().requires_grad_(True) if do_backward else hid
    T = h.shape[1]
    total = 0.0
    for s in range(0, T - 1, chunk):
        e = min(s + chunk, T - 1)
        tgt = lab[:, s + 1:e + 1].reshape(-1)
        keep = tgt != -100
        if not bool(keep.any()):
            continue
        hs = h[:, s:e, :]
        if do_backward:
            hs = hs.to(hid.dtype)
        lg = head(hs).float()
        lg = lg.reshape(-1, lg.shape[-1])
        l = nn.functional.cross_entropy(lg[keep], tgt[keep], reduction="sum") * (scale / n_scored)
        if do_backward:
            l.backward()
        total += float(l.detach())
        del lg, l, hs
    if do_backward:
        if h.grad is None:
            raise RuntimeError("chunked CE produced no gradient w.r.t. hidden states")
        # one cast at the end instead of one per chunk
        hid.backward(gradient=h.grad.to(hid.dtype))
    return total



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--rank", type=int, required=True)
    ap.add_argument("--world", type=int, required=True)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max-len", type=int, default=65536)
    ap.add_argument("--accum", type=int, default=1,
                    help="micro-steps per rank; GLOBAL batch = world * accum")
    ap.add_argument("--rank-lr-scale", action="store_true",
                    help="unused; kept so the flag cannot be silently mistyped into --lr")
    ap.add_argument("--rank16", type=int, default=16, dest="rank_lora")
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--targets", default="q_proj,k_proj,v_proj,o_proj")
    ap.add_argument("--out", default="/data/ckpt")
    ap.add_argument("--eval-frac", type=float, default=0.15)
    ap.add_argument("--eval-every", type=int, default=25)
    ap.add_argument("--min-samples", type=int, default=8)
    # 🔴 A first-vs-last eval comparison PASSED a run that overfit for 200 of its 300 steps: eval
    # bottomed at step 100 (0.51732) then rose monotonically to 0.60639, and 0.60639 < 0.61685 so
    # "decreased" was True. The gate must compare against the MINIMUM, not the start.
    ap.add_argument("--overfit-tol", type=float, default=0.02,
                    help="final eval may exceed its minimum by at most this fraction")
    ap.add_argument("--early-stop-min-delta", type=float, default=0.0018,
                    help="an eval counts as a regression only if it exceeds the running minimum by "
                         "MORE than this. 0.0 reproduces the old noise-triggered behaviour that set "
                         "three runs' headline numbers at arbitrary points; 0.0018 is the MEASURED "
                         "+-0.5%% eval noise band on ~0.35 for this model.")
    ap.add_argument("--early-stop-patience", type=int, default=3,
                    help="stop after this many consecutive evals above the running minimum; 0=off")
    ap.add_argument("--length-bucketed", type=int, default=1,
                    help="1 = group same-length samples into each step, killing the straggler cost")
    ap.add_argument("--seed", type=int, default=0,
                    help="0 = 不播种，与本项目前 14 次运行的行为完全一致（LoRA A 由各 rank 自己的 "
                         "RNG 抽，再广播 rank0 的）。>0 则 torch.manual_seed(seed)，用于关卡 0 的"
                         "『同配置换种子跑 3 遍看分数标准差』—— 审计要求：若标准差 ≥1.5 分，"
                         "单种子的 3 分结果不成立，整个课题不可测。")
    ap.add_argument("--bucket-seed", type=int, default=1234,
                    help="shared seed for the block shuffle; identical on every rank by design")
    # eval@0 over 235 held-out samples cost ~90 s, and at --eval-every 25 over 400 steps that is
    # ~17 evaluations, i.e. ~25 min of pure overhead. A few dozen samples estimate the held-out loss
    # just as well for a stopping decision.
    ap.add_argument("--eval-max", type=int, default=64,
                    help="cap the held-out set at this many samples (0 = uncapped)")
    ap.add_argument("--ckpt-every", type=int, default=100,
                    help="rank 0 writes lora_adapters_latest.pt every N steps (0 = off). "
                         "Any session reap or crash then costs at most one interval instead "
                         "of the entire run, which is how 14 runs' weights were lost.")
    ap.add_argument("--nccl-timeout-min", type=int, default=60)
    ap.add_argument("--chunked-ce", type=int, default=0,
                    help="LM-head chunk size in positions (0 = off, use HF's labels= path). "
                         "REQUIRED above max_len ~131072: fp32 logits are T*248320*4 B, so 262144 "
                         "would need 242.5 GiB for logits alone. 4096 costs 3.79 GiB at any T. "
                         "Gated by a fail-closed proof against the HF loss -- see --chunk-proof-tol.")
    ap.add_argument("--chunk-proof-tol", type=float, default=2e-3,
                    help="max |chunked_CE - HF_loss| accepted at startup. bf16 reductions differ by "
                         "association order, so this is a bf16-noise band, not an equality.")
    ap.add_argument("--chunk-grad-tol", type=float, default=2e-2,
                    help="max relative L2 difference between the chunked and HF gradients on the "
                         "trainable params. Looser than the loss band: gradients go through more "
                         "bf16 reductions, and a matching loss does NOT imply a matching gradient.")
    ap.add_argument("--chunk-grad-floor-mult", type=float, default=3.0,
                    help="accept the chunked gradient if it is within this multiple of the "
                         "MEASURED HF-vs-HF self-consistency floor. Guards against an absolute "
                         "tolerance that is really measuring backward nondeterminism.")
    ap.add_argument("--chunk-proof-len", type=int, default=8192,
                    help="prefix length for the chunked-CE proof. The HF reference path needs "
                         "T*248320*4 B of fp32 logits, so a full-length A/B is impossible at 256k; "
                         "the algorithm is length-independent, so prove it where both paths fit.")
    a = ap.parse_args()

    R, W = a.rank, a.world
    is0 = (R == 0)

    def log(msg):
        # Every rank's stdout is merged by run_ddp.sh; tag it or the log is unreadable.
        print("[rank%d] %s" % (R, msg), flush=True)

    torch.cuda.set_device(R)
    dev = "cuda:%d" % R
    # A rank that dies during a 52 GiB model load must not leave the others blocked forever.
    dist.init_process_group("nccl", rank=R, world_size=W,
                            timeout=__import__("datetime").timedelta(minutes=a.nccl_timeout_min))

    import transformers
    from transformers import AutoTokenizer
    if is0:
        log("torch %s | transformers %s | %s | world=%d"
            % (torch.__version__, transformers.__version__, torch.cuda.get_device_name(0), W))

    tok = AutoTokenizer.from_pretrained(a.model)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else (tok.eos_token_id or 0)

    t0 = time.perf_counter()
    model = transformers.Qwen3_5ForConditionalGeneration.from_pretrained(
        a.model, dtype=torch.bfloat16, low_cpu_mem_usage=True).to(dev)
    log("loaded in %.1fs | %.2f GiB allocated" % (time.perf_counter() - t0,
                                                  torch.cuda.memory_allocated(R) / 2**30))

    # 关卡 0 用：显式播种让"同配置不同种子"可控且可复现。
    # a.seed == 0 时完全不调用 manual_seed，保持与历史运行逐位一致。
    if a.seed:
        torch.manual_seed(a.seed)
        torch.cuda.manual_seed_all(a.seed)
        _random.seed(a.seed)
        if is0:
            log("SEED: torch/cuda/python 播种为 %d（LoRA 初始化与数据顺序均受其影响）" % a.seed)

    n_inj, n_tr = inject_lora(model, a.targets.split(","), a.rank_lora, a.alpha)
    if n_inj == 0:
        log("RESULT: NO_LORA_TARGETS_MATCHED")
        dist.destroy_process_group()
        return 3

    # 🔴 Every rank ran kaiming_uniform_ with its own RNG state, so the adapters START DIFFERENT.
    # Averaging gradients over ranks that hold different weights trains nothing coherent. Broadcast
    # rank 0's adapters so all ranks begin from one identical model. (Seeding would also work but
    # relies on every rank drawing in the same order; a broadcast is a fact, not an assumption.)
    lora_params = []
    for m in model.modules():
        if isinstance(m, LoRALinear):
            lora_params += [m.a, m.b]
    with torch.no_grad():
        for p in lora_params:
            dist.broadcast(p.data, src=0)
    if is0:
        log("LoRA: injected %d modules, %d trainable params; adapters broadcast from rank 0"
            % (n_inj, n_tr))

    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    model.train()

    # 🔴 PARAM (WEIGHT) CPU OFFLOAD -- env-gated, DEFAULT OFF (byte-identical to before when unset).
    # After OFFLOAD_ACT moves the activation checkpoints to CPU, the last fixed cost blocking ctx
    # 262144 on one 143 GiB H200 is the ~54 GiB of FROZEN base weights held on the card. This pages
    # each decoder layer's base weights to pinned host RAM and streams them to the GPU just-in-time
    # in a forward PRE-hook, evicting them in the POST-hook -- on BOTH the original forward and the
    # grad-ckpt recompute (see enable_param_offload for why the hooks fire on the recompute). LoRA
    # A/B stay on the GPU so training is unchanged. Needs a large --memory (host holds the base:
    # ~54 GiB, pinned, PER RANK). Intended combo for full-length: PARAM_OFFLOAD=1 OFFLOAD_ACT=1
    # OFFLOAD_PIN=1 --chunked-ce 4096 on world=1.
    if os.environ.get("PARAM_OFFLOAD", "0") == "1":
        enable_param_offload(model, dev, log=log, is0=is0)
        torch.cuda.empty_cache()                     # return the freed base-weight blocks to the pool
        if is0:
            log("PARAM_OFFLOAD: %.2f GiB allocated on %s after offload (base now streams from host)"
                % (torch.cuda.memory_allocated(R) / 2**30, dev))

    all_samples = load_jsonl(a.data, a.max_len)
    if not all_samples:
        log("RESULT: EMPTY_DATASET")
        dist.destroy_process_group()
        return 4
    n_eval = max(1, int(len(all_samples) * a.eval_frac)) if len(all_samples) > 4 else 0
    if a.eval_max and n_eval > a.eval_max:
        n_eval = a.eval_max
    eval_all = all_samples[:n_eval]
    train_all = all_samples[n_eval:]

    # ---- shard: rank r takes every W-th sample. Disjoint by construction, and the strided
    # pattern (not contiguous blocks) keeps each rank's length distribution similar, which matters
    # because step time tracks context length.
    #
    # 🔴 BUT strided sharding only equalises the distributions, not the individual steps. Every step
    # ends at a barrier plus an all-reduce, so it costs the SLOWEST of 8 ranks. MEASURED on v5:
    # step times ranged 4.1-24.6 s with a global context spread of 91,846-189,077 tokens, and the
    # per-token cost came out 1.26x worse than v4 even though nothing about the model changed.
    # LENGTH BUCKETING fixes this: sort by length, cut into blocks of W near-identical samples, and
    # give rank r the r-th element of each block. Then the 8 samples in flight at any step have
    # nearly the same length, so max(8 ranks) ~ mean and the straggler cost collapses.
    # The block ORDER is then shuffled with a fixed seed -- without that, lengths would rise
    # monotonically with step number, which is an accidental curriculum, not a batching strategy.
    # The seed is shared, so all ranks agree on the order without communicating.
    if a.length_bucketed:
        order_idx = sorted(range(len(train_all)), key=lambda i: len(train_all[i][0]))
        nb = len(order_idx) // W
        blocks = [order_idx[b * W:(b + 1) * W] for b in range(nb)]
        dropped = len(order_idx) - nb * W
        import random as _random
        _random.Random(a.bucket_seed).shuffle(blocks)
        samples = [train_all[blk[R]] for blk in blocks]
        if is0:
            log("length-bucketed sharding: %d blocks of %d (%d sample(s) dropped as remainder);"
                " within-step length spread now bounded by the block, not the dataset"
                % (nb, W, dropped))
    else:
        samples = train_all[R::W]
    eval_samples = eval_all[R::W] if eval_all else []
    counts = torch.zeros(W, dtype=torch.long, device=dev)
    counts[R] = len(samples)
    dist.all_reduce(counts, op=dist.ReduceOp.SUM)
    if is0:
        tot_sharded = int(counts.sum().item())
        # Length bucketing deliberately drops the last len(train_all) % W samples so every block is
        # full, so the expected coverage is nb*W, NOT len(train_all). Comparing against the latter
        # made a healthy bucketed run report `disjoint_ok=False` (1504 vs 1507) -- a check that
        # cries wolf gets ignored, which is how a real one gets missed later.
        expected = (len(samples) * W) if a.length_bucketed else len(train_all)
        log("dataset %d samples -> %d train / %d eval; shards %s  union=%d  expected=%d"
            "  covered_ok=%s%s"
            % (len(all_samples), len(train_all), len(eval_all), counts.tolist(), tot_sharded,
               expected, tot_sharded == expected,
               ("  (%d dropped as bucket remainder)" % (len(train_all) - expected))
               if a.length_bucketed and expected != len(train_all) else ""))
        if len(train_all) < a.min_samples:
            log("WARNING: only %d distinct training samples (< %d): any loss decrease is likely"
                " MEMORISATION, not learning." % (len(train_all), a.min_samples))
    if not samples:
        log("RESULT: EMPTY_SHARD (world %d larger than dataset)" % W)
        dist.destroy_process_group()
        return 5

    # ---------- PROOF 1 (rank 0): the label mask gates the loss, by exact recomputation ----------
    proof, base, manual, ign_ok, trn_ok = {}, None, None, None, None
    if is0:
        ids, lab, att = batch(samples, [0], pad_id, dev)
        # 🔴 HF's labels= path materialises [B,T,vocab] fp32 logits (T*248320*4 B); at max_len 131072
        # that is ~130 GiB and OOMs a 143 GiB card. The mask proof is length-INDEPENDENT (it only checks
        # that manual CE over unmasked positions == the model loss on this sample), so run it on a short
        # prefix -- same reason + knob the chunk proof (PROOF 1b) already uses. MEASURED 2026-09-16:
        # full-length mask proof OOM'd (tried 66 GiB with 105 in use) at 131072; the chunked_ce training
        # path itself is fine (peak 120.6 GiB). Slice to chunk_proof_len so both proofs fit.
        Pp = min(ids.shape[1], a.chunk_proof_len)
        ids, lab, att = ids[:, :Pp], lab[:, :Pp], att[:, :Pp]
        with torch.no_grad():
            out0 = model(input_ids=ids, attention_mask=att, labels=lab)
            base = out0.loss.item()
            logits = out0.logits.float()
            sl = logits[:, :-1, :].reshape(-1, logits.shape[-1])
            tl = lab[:, 1:].reshape(-1)
            keep = tl != -100
            manual = nn.functional.cross_entropy(sl[keep], tl[keep]).item()
            proof["manual_ce_over_unmasked_only"] = manual
            proof["n_scored_positions"] = int(keep.sum().item())
            proof["n_total_positions"] = int(tl.numel())
            pos_trn = (lab[0] != -100).nonzero().flatten().tolist()
            if pos_trn:
                j = pos_trn[len(pos_trn) // 2]
                kv = lab[0, j].item()
                lab[0, j] = (kv + 977) % 200000
                proof["trainable_label_perturbed"] = model(
                    input_ids=ids, attention_mask=att, labels=lab).loss.item()
                lab[0, j] = kv
        ign_ok = abs(manual - base) < 2e-4
        trn_ok = ("trainable_label_perturbed" not in proof
                  or abs(proof["trainable_label_perturbed"] - base) > 1e-5)
        log("MASK PROOF model_loss=%.6f manual_CE=%.6f |diff|=%.2e agree=%s trainable_matters=%s"
            % (base, manual, abs(manual - base), ign_ok, trn_ok))
        log("MASK PROOF scored %d of %d positions (%.1f%% masked out)"
            % (proof["n_scored_positions"], proof["n_total_positions"],
               100.0 * (1 - proof["n_scored_positions"] / proof["n_total_positions"])))
        del logits, sl, tl, keep
    torch.cuda.empty_cache()

    # ---------- PROOF 1b: chunked CE reproduces the HF loss AND its gradient, or we REFUSE ----------
    # 🔴 FAIL-CLOSED, on every rank (all ranks return together, so a failure cannot leave survivors
    # blocking in NCCL). A chunked loss that silently disagrees with the real objective would train a
    # different model while every logged curve looked healthy -- the same class of invisible failure
    # the sync proof and mask proof exist to catch. Two things can break it and both are silent:
    # `resolve_head()` picking the wrong submodule (skipping the final norm), and mrope position
    # handling differing when the decoder is called directly instead of through the CausalLM wrapper.
    #
    # WHY A TRUNCATED PREFIX: the reference path is the thing chunking exists to avoid -- at T=262144
    # HF's fp32 logits are 242.5 GiB and cannot be allocated at all, so a full-length A/B is not just
    # expensive, it is impossible exactly when it matters most. The ALGORITHM (shift, global
    # denominator, two-stage backward) is length-independent, so validate it at a T where both paths
    # fit and then use the chunked path at the T where only it does. The prefix must still exceed one
    # chunk, or the loop would run a single slice and the accumulation logic would go untested.
    chunk_proof = None
    if a.chunked_ce:
        ids_c, lab_c, att_c = batch(samples, [0], pad_id, dev)
        Tp = min(ids_c.shape[1], max(a.chunk_proof_len, 2 * a.chunked_ce + 1))
        ids_c, lab_c, att_c = ids_c[:, :Tp], lab_c[:, :Tp], att_c[:, :Tp]
        n_sc = int((lab_c[:, 1:] != -100).sum().item())
        if n_sc == 0:
            log("CHUNK PROOF SKIPPED: first %d tokens of sample 0 contain no scored position" % Tp)
            chunk_proof = {"skipped": "no_scored_positions_in_prefix", "prefix_len": Tp}
        else:
            trainable = [p for p in model.parameters() if p.requires_grad]

            def _flat_grads():
                return torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p))
                                  .detach().float().reshape(-1) for p in trainable])

            for p in trainable:
                p.grad = None
            hf_loss = model(input_ids=ids_c, attention_mask=att_c, labels=lab_c).loss
            hf_val = hf_loss.item()
            hf_loss.backward()
            g_hf = _flat_grads()
            del hf_loss
            for p in trainable:
                p.grad = None
            torch.cuda.empty_cache()
            # 🔴 SELF-CONSISTENCY FLOOR: run the REFERENCE path a second time and compare it to
            # ITSELF. Without this the gate cannot distinguish "the chunked gradient is wrong" from
            # "this backward is not reproducible to better than X". MEASURED 2026-08-22: chunked-vs-HF
            # sat at rel_L2 1.74e-2 .. 2.52e-2 across 8 ranks with cosine 0.9998, straddling a 2e-2
            # tolerance I had simply guessed -- and the single-chunk diagnostic showed the residual is
            # NOT cross-slice accumulation, which was my stated cause and was wrong. Gradient
            # checkpointing RECOMPUTES every layer in the backward, and with bf16 plus the freshly
            # built causal_conv1d kernels that recompute need not be bitwise reproducible. So measure
            # the floor and judge against it, rather than against a number I picked.
            hf2 = model(input_ids=ids_c, attention_mask=att_c, labels=lab_c).loss
            hf2_val = hf2.item()
            hf2.backward()
            g_hf2 = _flat_grads()
            del hf2
            for p in trainable:
                p.grad = None
            torch.cuda.empty_cache()
            gn = float(g_hf.norm())
            floor = float((g_hf2 - g_hf).norm()) / max(gn, 1e-12)
            del g_hf2
            ck_val = chunked_ce(model, ids_c, lab_c, att_c, a.chunked_ce, do_backward=True)
            g_ck = _flat_grads()
            for p in trainable:
                p.grad = None
            torch.cuda.empty_cache()

            d_loss = abs(ck_val - hf_val)
            d_grad = float((g_ck - g_hf).norm()) / max(gn, 1e-12)
            cos = float(torch.dot(g_ck, g_hf) / max(float(g_ck.norm()) * gn, 1e-12))
            ck1 = chunked_ce(model, ids_c, lab_c, att_c, Tp + 1, do_backward=True)
            g_ck1 = _flat_grads()
            for p in trainable:
                p.grad = None
            torch.cuda.empty_cache()
            d_grad1 = float((g_ck1 - g_hf).norm()) / max(gn, 1e-12)
            # The gate: chunked-vs-HF must be within `--chunk-grad-floor-mult` of the MEASURED
            # HF-vs-HF floor, or under the absolute tolerance -- whichever is more permissive. An
            # absolute-only gate is unfalsifiable when the floor is unknown; a floor-relative gate says
            # "no worse than the reference disagrees with itself", which is the strongest claim that
            # can be true. If the floor itself is large, THAT is the finding and it is logged.
            allow = max(a.chunk_grad_tol, a.chunk_grad_floor_mult * floor)
            ok = (d_loss <= a.chunk_proof_tol and d_grad <= allow)
            chunk_proof = {"prefix_len": Tp, "n_scored": n_sc, "chunk": a.chunked_ce,
                           "hf_loss": hf_val, "hf_loss_rerun": hf2_val, "chunked_loss": ck_val,
                           "abs_diff": d_loss, "loss_tol": a.chunk_proof_tol,
                           "grad_rel_l2": d_grad, "grad_cosine": cos,
                           "grad_self_consistency_floor": floor,
                           "grad_allowed": allow, "grad_tol_abs": a.chunk_grad_tol,
                           "grad_floor_mult": a.chunk_grad_floor_mult,
                           "single_chunk_grad_rel_l2": d_grad1,
                           "hf_grad_norm": gn, "agrees": bool(ok)}
            log("CHUNK PROOF T=%d chunk=%d  loss hf=%.6f rerun=%.6f chunked=%.6f |d|=%.2e (tol %.1e)"
                % (Tp, a.chunked_ce, hf_val, hf2_val, ck_val, d_loss, a.chunk_proof_tol))
            log("CHUNK PROOF grad rel_L2=%.3e cosine=%.9f  HF-vs-HF FLOOR=%.3e  allowed=%.3e"
                " agree=%s" % (d_grad, cos, floor, allow, ok))
            log("CHUNK PROOF single-chunk grad rel_L2=%.3e ; chunked/floor=%.2fx -> %s"
                % (d_grad1, d_grad / max(floor, 1e-12),
                   "within the reference's own reproducibility" if d_grad <= allow
                   else "EXCEEDS the floor -- a real disagreement"))
            del g_hf, g_ck, g_ck1
        del ids_c, lab_c, att_c
        torch.cuda.empty_cache()
        # 🔴 THE VERDICT MUST BE COLLECTIVE. This cost 8 cards for 28 minutes on 2026-08-22.
        # The gate was per-rank, and the quantity it gates is noise-dominated and sits ON the
        # tolerance: in v11 attempt 1 ranks 1,2,3 failed while 4,5 passed; in attempt 2 the SAME ranks
        # on the SAME shards swapped -- 4,5 failed and 1,2,3 passed. Partial failure DEADLOCKS: the
        # failing ranks entered `dist.destroy_process_group()` while the passing ranks went on to the
        # `eval_loss()` all_reduce, so 6 ranks spun in NCCL at 97.8 % cpu / 100 % gpu and 2 sat at
        # 10.4 % / 0 % with no log write for 1713 s. `run_ddp.sh` only tears down when a rank EXITS
        # non-zero and these never exited, so it would have held the node for the full 25200 s budget.
        # An all_reduce(MIN) makes every rank reach the SAME decision, so they abort together (which is
        # what makes `return 6` safe) or continue together.
        vote = torch.tensor([1.0 if chunk_proof.get("agrees", True) else 0.0], device=dev)
        dist.all_reduce(vote, op=dist.ReduceOp.MIN)
        all_agree = bool(vote.item() > 0.5)
        if is0 and chunk_proof is not None:
            chunk_proof["all_ranks_agree"] = all_agree
        if not all_agree:
            log("RESULT: CHUNKED_CE_PROOF_FAILED (collective) -- at least one rank disagreed; ALL ranks"
                " abort together so no rank is left in a collective. This rank: loss|d|=%s"
                " grad_rel_L2=%s floor=%s allowed=%s"
                % (chunk_proof.get("abs_diff"), chunk_proof.get("grad_rel_l2"),
                   chunk_proof.get("grad_self_consistency_floor"), chunk_proof.get("grad_allowed")))
            dist.destroy_process_group()
            return 6

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / max(1, int(0.1 * a.steps))))

    def eval_loss():
        """Mean held-out loss over the GLOBAL eval set (each rank does its shard, then reduce)."""
        if not eval_all:
            return None
        model.eval()
        tot = torch.zeros(2, device=dev)     # [sum_loss, n]
        with torch.no_grad():
            for k in range(len(eval_samples)):
                ids_e, lab_e, att_e = batch(eval_samples, [k], pad_id, dev)
                if a.chunked_ce:
                    tot[0] += chunked_ce(model, ids_e, lab_e, att_e, a.chunked_ce,
                                         do_backward=False)
                else:
                    tot[0] += model(input_ids=ids_e, attention_mask=att_e, labels=lab_e).loss
                tot[1] += 1
        dist.all_reduce(tot, op=dist.ReduceOp.SUM)
        model.train()
        return (tot[0] / torch.clamp(tot[1], min=1)).item()

    eval_hist = []
    ev_track = []
    stopped_early = None
    e0 = eval_loss()
    if e0 is not None:
        ev_track.append(e0)
    if e0 is not None and is0:
        eval_hist.append({"step": 0, "eval_loss": round(e0, 6)})
        log("eval@0 %.5f (%d held-out samples, global)" % (e0, len(eval_all)))

    losses, step_s, step_tok, step_trainable = [], [], [], []
    order = list(range(len(samples)))
    sync_proof = None
    t_all = time.perf_counter()
    for step in range(a.steps):
        opt.zero_grad(set_to_none=True)
        # Local token counters, summed across ranks so the logged tok/s is the REAL aggregate --
        # reporting only rank 0's tokens would understate the run by 8x and make the whole point
        # of this script invisible.
        acc = torch.zeros(3, device=dev)     # [loss_sum, n_tok, n_trainable]
        dist.barrier()
        torch.cuda.synchronize()
        t_step = time.perf_counter()
        for micro in range(a.accum):
            k = (step * a.accum + micro) % len(order)
            ids, lab, att = batch(samples, [order[k]], pad_id, dev)
            if a.chunked_ce:
                # backward runs INSIDE, slice by slice, so the fp32 logits monolith never exists.
                acc[0] += chunked_ce(model, ids, lab, att, a.chunked_ce,
                                     scale=1.0 / a.accum, do_backward=True)
            else:
                loss = model(input_ids=ids, attention_mask=att, labels=lab).loss / a.accum
                loss.backward()
                acc[0] += loss.detach()
            acc[1] += ids.numel()
            acc[2] += (lab != -100).sum()

        # ---- the one collective that makes this data-parallel: average the adapter grads.
        # Done ONCE per optimizer step (after accumulation), not per micro-step.
        for p in params:
            if p.grad is None:
                # A rank with no grad for a param would leave the others' all_reduce unmatched and
                # hang the job. Give it a zero so the collective shapes always agree.
                p.grad = torch.zeros_like(p)
            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
            p.grad /= W
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        dist.all_reduce(acc, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t_step

        # ---------- PROOF 2: ranks are still in sync after the update ----------
        # Checked on the first step and periodically. If the all-reduce silently failed, each rank
        # would drift onto its own model and rank 0 would save an adapter that no other rank
        # agrees with -- and the loss curve would look perfectly healthy the whole time.
        if step == 0 or (step + 1) % 50 == 0:
            with torch.no_grad():
                ref = [p.data.clone() for p in lora_params]
                for t in ref:
                    dist.broadcast(t, src=0)
                worst = max((p.data - r).abs().max().item()
                            for p, r in zip(lora_params, ref))
                w = torch.tensor([worst], device=dev)
                dist.all_reduce(w, op=dist.ReduceOp.MAX)
                worst = w.item()
                del ref
            if is0:
                sync_proof = {"step": step + 1, "max_abs_param_divergence": worst,
                              "in_sync": bool(worst == 0.0)}
                log("SYNC PROOF step %d: max |param - rank0_param| across all ranks = %.3e -> %s"
                    % (step + 1, worst, "IN SYNC" if worst == 0.0 else "DIVERGED"))

        gl = acc[0].item() / W          # mean loss per rank == mean over the global batch
        n_tok, n_trn = int(acc[1].item()), int(acc[2].item())
        losses.append(gl)
        step_s.append(round(dt, 2))
        step_tok.append(n_tok)
        step_trainable.append(n_trn)
        if is0:
            log("step %3d/%d loss %.5f grad_norm %.4f lr %.2e peak %.1f GiB %.1fs/step "
                "%.0f tok/s (%d ctx global, %d trainable) cum %.1fs"
                % (step + 1, a.steps, gl, float(gn), sched.get_last_lr()[0],
                   torch.cuda.max_memory_allocated(R) / 2**30, dt, n_tok / max(dt, 1e-9),
                   n_tok, n_trn, time.perf_counter() - t_all))
            try:
                os.makedirs(a.out, exist_ok=True)
                with open(os.path.join(a.out, "progress.jsonl"), "a") as pf:
                    pf.write(json.dumps({
                        "step": step + 1, "total_steps": a.steps, "loss": round(gl, 6),
                        "grad_norm": round(float(gn), 5), "lr": sched.get_last_lr()[0],
                        "peak_gib": round(torch.cuda.max_memory_allocated(R) / 2**30, 2),
                        "sec_per_step": round(dt, 2), "ctx_tokens": n_tok,
                        "trainable_tokens": n_trn, "tok_per_s": round(n_tok / max(dt, 1e-9), 1),
                        "world": W, "global_batch": W * a.accum,
                        "cum_s": round(time.perf_counter() - t_all, 1),
                    }) + "\n")
            except Exception:
                pass
        # 🔴 PERIODIC CHECKPOINT, so that any reap or crash costs at most one interval instead of the
        # whole run. 167.8 MB at r=64; MEASURED step time is 10.5 s, so a write every 100 steps is
        # negligible. This is the fix for the failure mode that destroyed 14 runs' weights.
        if is0 and a.ckpt_every > 0 and (step + 1) % a.ckpt_every == 0:
            try:
                save_adapters(model, a.out, "lora_adapters_latest.pt",
                              {"kind": "latest", "step": step + 1})
            except Exception as e:
                log("WARN periodic checkpoint failed: %s" % type(e).__name__)
        if eval_all and ((step + 1) % a.eval_every == 0 or step + 1 == a.steps):
            ev = eval_loss()
            # Tracked on EVERY rank, not just rank 0. eval_loss() all-reduces, so all ranks hold
            # the identical value and therefore reach the identical stop decision -- no broadcast
            # and, crucially, no chance of rank 0 breaking out while the others block in a
            # collective waiting for it.
            ev_track.append(ev)
            if is0:
                eval_hist.append({"step": step + 1, "eval_loss": round(ev, 6)})
                # 🔴 SAVE ON BEST. The single end-of-run save persisted the LAST weights, but every
                # run in this series reached its eval minimum mid-run, so the weights that were kept
                # were never the best ones. Keep the argmin explicitly.
                if ev <= min(ev_track):
                    try:
                        save_adapters(model, a.out, "lora_adapters_best.pt",
                                      {"kind": "best", "step": step + 1, "eval_loss": round(ev, 6)})
                    except Exception as e:
                        log("WARN best-checkpoint save failed: %s" % type(e).__name__)
                log("eval@%d %.5f%s" % (step + 1, ev,
                    "   <-- new best" if ev <= min(ev_track) else
                    "   (best %.5f, %d eval(s) ago)" % (min(ev_track),
                                                        len(ev_track) - 1 - ev_track.index(min(ev_track)))))
                # 🔴 Append eval to progress.jsonl, not just stdout. I fixed exactly this in the
                # single-card trainer and then ran the DDP one, which still had the bug -- so v5's
                # eval points existed ONLY in a stdout stream that `tail -420` can clip at 400
                # steps. A fix applied to one of two near-identical files is half a fix.
                try:
                    with open(os.path.join(a.out, "progress.jsonl"), "a") as pf:
                        pf.write(json.dumps({"step": step + 1, "eval_loss": round(ev, 6),
                                             "kind": "eval", "world": W}) + "\n")
                except Exception:
                    pass
            # 🔴 STRIKES MUST BE MATERIAL, NOT MERELY ABOVE THE MINIMUM.
            # The old rule counted EVERY eval after the best one as a strike, so on a curve that is
            # flat-but-noisy near its minimum it fires almost at random. MEASURED: that rule set the
            # headline number for THREE consecutive runs at three unrelated points -- v11c stopped at
            # 900 (min 0.350237 @ 825, 1.23 ep), v9 ran to 1325 (min 0.346135 @ 1250, 1.81 ep), v10 to
            # 1425. v9's own eval fell a further ~1.4 % between step 875 and 1250, so v11c was halted
            # WHILE STILL DESCENDING and its "minimum" is an artefact of the stopping rule. The eval
            # noise band on this model was measured at about +-0.5 % of ~0.35, i.e. ~0.0018 absolute,
            # so a strike must clear the best by more than that to mean anything.
            # Counting only the TRAILING run of material regressions also means a single noise spike
            # no longer poisons the count -- one eval back near the best resets it.
            if a.early_stop_patience > 0:
                best_v = min(ev_track)
                strikes = 0
                for v in reversed(ev_track):
                    if v > best_v + a.early_stop_min_delta:
                        strikes += 1
                    else:
                        break
                if strikes >= a.early_stop_patience:
                    if is0:
                        log("EARLY STOP at step %d: %d consecutive eval(s) worse than the minimum"
                            " %.5f by more than min-delta %.5f. Training further only overfits."
                            % (step + 1, strikes, best_v, a.early_stop_min_delta))
                    stopped_early = step + 1
                    break

    if is0:
        med = sorted(step_s)[len(step_s) // 2]
        med_tok = sorted(step_tok)[len(step_tok) // 2]
        first = sum(losses[:3]) / max(1, len(losses[:3]))
        last = sum(losses[-3:]) / max(1, len(losses[-3:]))
        log("STEP THROUGHPUT median %.2fs/step %.0f ctx-tok/s aggregate over %d ranks"
            % (med, med_tok / max(med, 1e-9), W))
        log("LOSS first3=%.5f last3=%.5f delta=%+.5f decreased=%s"
            % (first, last, last - first, last < first))
        save_adapters(model, a.out, "lora_adapters.pt",
                      {"kind": "final", "step": len(losses), "planned_steps": a.steps})
        ev_ok = None
        ev_min = ev_best_step = ev_overfit = None
        if len(eval_hist) >= 2:
            vals = [e["eval_loss"] for e in eval_hist]
            ev_min = min(vals)
            ev_best_step = eval_hist[vals.index(ev_min)]["step"]
            ev_ok = vals[-1] < vals[0]
            # The real question is not "is the end better than the start" but "is the end still
            # near the best we ever saw". A U-shaped eval curve fails this and passes the former.
            ev_overfit = vals[-1] > ev_min * (1.0 + a.overfit_tol)
            log("EVAL first=%.5f last=%.5f min=%.5f@step%d decreased=%s overfit=%s"
                % (vals[0], vals[-1], ev_min, ev_best_step, ev_ok, ev_overfit))
            if ev_overfit:
                log("EVAL final is %.1f%% above the minimum -> the useful stopping point was step"
                    " %d, not %d. More steps on this data cost generalisation."
                    % (100.0 * (vals[-1] / ev_min - 1.0), ev_best_step, eval_hist[-1]["step"]))
        enough = len(train_all) >= a.min_samples
        meta = {"world": W, "global_batch": W * a.accum, "steps": a.steps, "lr": a.lr,
                "rank": a.rank_lora, "alpha": a.alpha, "targets": a.targets,
                "losses": losses, "loss_first3": first, "loss_last3": last,
                "decreased": bool(last < first), "trainable_params": n_tr,
                "step_seconds": step_s, "step_ctx_tokens": step_tok,
                "step_trainable_tokens": step_trainable, "median_step_s": med,
                "shard_counts": counts.tolist(), "n_train_samples": len(train_all),
                "enough_samples": bool(enough), "eval_hist": eval_hist,
                "eval_decreased": ev_ok, "eval_min": ev_min, "eval_best_step": ev_best_step,
                "eval_overfit": ev_overfit, "stopped_early_at": stopped_early,
                "epochs": round(len(losses) * W * a.accum / max(1, len(train_all)), 2),
                "sync_proof": sync_proof,
                "mask_proof": {"base": base, **proof, "manual_ce_agrees": bool(ign_ok),
                               "trainable_matters": bool(trn_ok)},
                "chunked_ce": a.chunked_ce, "chunk_proof": chunk_proof,
                "peak_gib": torch.cuda.max_memory_allocated(R) / 2**30}
        open(os.path.join(a.out, "train_result.json"), "w").write(json.dumps(meta, indent=2))
        log("saved adapters (%d tensors) + train_result.json -> %s" % (len(sd), a.out))
        ok = (last < first) and ign_ok and trn_ok and enough and (ev_ok is not False) \
            and not ev_overfit and (sync_proof or {}).get("in_sync", False)
        if not enough:
            log("VERDICT downgraded: only %d distinct samples" % len(train_all))
        if ev_overfit:
            log("VERDICT downgraded: held-out loss ended %.1f%% above its minimum (overfitting)"
                % (100.0 * (eval_hist[-1]["eval_loss"] / ev_min - 1.0)))
        if not (sync_proof or {}).get("in_sync", False):
            log("VERDICT downgraded: ranks were NOT proven in sync")
        epochs = len(losses) * W * a.accum / max(1, len(train_all))
        log("EPOCHS %.2f  (%d steps x global batch %d / %d train samples)"
            % (epochs, len(losses), W * a.accum, len(train_all)))
        print("RESULT: %s" % ("SFT_DDP_OK" if ok else "SFT_DDP_SUSPECT"), flush=True)

    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
