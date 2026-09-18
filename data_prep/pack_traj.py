#!/usr/bin/env python3
"""pack_traj.py -- gpt-5.6 Python->JS repo-translation trajectories (`chat.jsonl`)
-> token-level SFT samples for `train_sft_lora_ddp.py`, with an assistant-only loss mask that is
PROVEN byte-exact against Qwen3.8-27B's own chat template.

ON-DISK CONTRACT (read out of the trainer, not guessed)
  `train_sft_lora_ddp.py::load_jsonl` (lines 106-117) does, per line:
        d = json.loads(line); ids, lab = d["input_ids"][:max_len], d["labels"][:max_len]
        if sum(1 for x in lab if x != -100) == 0: continue
  so the contract is exactly one JSON object per line with two keys:
      input_ids : list[int]
      labels    : list[int]   (-100 == not scored; UNSHIFTED, i.e. labels[i] aligns with ids[i];
                               the model/`chunked_ce` does the t -> t+1 shift internally)
  Nothing else is read. Any extra key is ignored, so we do not add any.

WHY THIS FILE AND NOT `sft/pack_sft.py`
  `pack_sft.py` (rsi_2608) targets a different corpus: a JSON *array* of conversations that carry
  `system`/`tool` roles, native `tool_calls` mappings, `reasoning_content` and per-message
  `loss_mask`. THIS corpus is a JSONL of `{"messages":[...]}` where (MEASURED over all 53,808
  records, 2026-09-01):
    * roles are STRICTLY alternating user/assistant, 53,808/53,808; every record starts `user`
      and ends `assistant`; message counts are all even, 2..164;
    * messages carry ONLY `role` and `content` -- no `tool_calls`, no `reasoning_content`,
      no `loss_mask`, no `tools`, no `system`;
    * the tool call is embedded as TEXT inside the assistant `content`.
  So the tool-call *renderer* of pack_sft.py must NOT run here (there is no mapping to render);
  content is passed through verbatim, exactly as the template's `render_content(...)|trim` does.
  The segment/mask/verify architecture is lifted from pack_sft.py deliberately.

THE THINKING DECISION -- the single most consequential choice in this file
  `chat_template.jinja` line 116-117 is
      {%- if preserve_thinking is undefined or preserve_thinking is true or ... %}
          '<|im_start|>assistant\\n<think>\\n' + reasoning_content|trim + '\\n</think>\\n\\n' + content
  `preserve_thinking` is undefined by default => EVERY assistant turn gets a `<think>` block, and
  because this corpus has NO `reasoning_content`, that block is always the EMPTY
  `<think>\\n\\n</think>\\n\\n`. That is not optional -- there is no way to render this corpus with
  this template and not get it. What IS a choice is what the generation prompt looks like at serve
  time, and the template gives two self-consistent pairings (both MEASURED with transformers 5.16.1
  on 2026-09-01):

    --thinking off   (enable_thinking=False) : no system block at all, and
        apply_chat_template(..., add_generation_prompt=True, enable_thinking=False)
          == '<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n'
      which is BYTE-IDENTICAL to the masked prefix this file emits before the first scored token.
      Train and serve agree exactly. <-- DEFAULT.

    --thinking xhigh (template default)      : a system block carrying the xhigh reasoning
        instruction, and
        apply_chat_template(..., add_generation_prompt=True)
          == '<|im_start|>assistant\\n<think>\\n'
      i.e. at serve time the model is asked to GENERATE reasoning and close `</think>` itself,
      while every training example showed it an EMPTY think block. That is a real train/serve
      skew: we would be fitting p(content | empty think) and sampling p(content | real think).
      Offered for completeness; if you use it, serve with the same flag and read SFT_PLAN.md.

  `preserve_thinking=False` is a trap: it strips the think block from every assistant turn EXCEPT
  the one after the last user query, so a multi-turn record renders inconsistently across its own
  turns. MEASURED and refused; never passed by this file.

THE LOSS MASK -- exact policy, and why each boundary is where it is
  Per assistant turn we emit four segments and label them:
      '<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n'   -> MASKED (-100)
            This is byte-for-byte the serve-time generation prompt. At inference the model is
            HANDED these tokens; it never generates them; so they are prompt, and prompt is masked.
      '{content|trim}'                                     -> SCORED
      '<|im_end|>'                                         -> SCORED  (must learn to stop)
      '\\n'                                                 -> MASKED  (never generated; the turn
            ends at <|im_end|>, and because labels are shifted internally the <|im_end|> position
            would otherwise be trained to predict this newline)
  user turns are MASKED in full. There is no system/tool role in this corpus.

HOW THE MASK IS VERIFIED (--verify N; four independent checks, all fail-closed)
  V1 RENDER   ''.join(segment texts) == tok.apply_chat_template(messages, tokenize=False,
                 add_generation_prompt=False, **thinking_kwargs)   -- byte equality against a real
              jinja2 execution of the shipped template. Catches any drift in my re-implementation.
  V2 RETOK    concat(per-segment ids) == tok(''.join(texts))       -- proves no BPE merge straddles
              a segment cut. Every cut here abuts an added token (<|im_start|>, <think>,
              </think>, <|im_end|>), and the fast tokenizer splits on added tokens before BPE, so
              this must hold; V2 asserts it instead of trusting the argument.
  V3 GENPROMPT (the one that catches off-by-one) for EVERY assistant turn k:
                 ids[: first_scored_index_of_turn_k]
                     == tok(apply_chat_template(messages[:2k], add_generation_prompt=True, ...))
              i.e. the first token we score is exactly the first token the model must generate at
              serve time, in TOKEN space, at every turn. An off-by-one of +1 (training on the last
              prompt token) or -1 (dropping the first content token) breaks this equality.
  V4 DECODE   tok.decode(scored ids of turn k) == content|trim + '<|im_end|>'  -- proves the scored
              run covers exactly the assistant text and nothing else, from the other direction.
  Plus two unconditional whole-dataset invariants, checked on every emitted sample:
  V5 labels[i] in (-100, ids[i]) for all i          (labels are ids, unshifted, never a stray id)
  V6 scored positions are a subset of the assistant-content/<|im_end|> character spans, computed
     independently from the segment table.

DETERMINISM / RESUMABILITY
  Output order is a pure function of (input sha256, selection flags, shuffle seed): the record list
  is built, capped, ordered and sharded before any tokenization happens. Shards are written to
  `<out>.parts/shard-NNNNN.jsonl` with a `.done` sidecar holding the shard's signature; --resume
  reuses shards whose signature matches and re-runs the rest, then concatenates in shard order.
  Byte-identical output for identical flags, regardless of --workers.

EVAL LEAKAGE (why the output ORDER matters, not just its contents)
  The trainer builds its held-out set as a PREFIX: `eval_all = all_samples[:n_eval]` with
  n_eval = min(int(0.15*N), --eval-max) -- it does NOT shuffle. This corpus has only 2,454 distinct
  first-turn prompts for 53,808 records (up to 112 trajectories of the SAME task), so a naive
  prefix split puts other trajectories of the eval tasks into train and the eval loss stops
  measuring generalisation. This file therefore reserves whole PROMPT GROUPS for the head of the
  file (--holdout-head, default 64 == the trainer's --eval-max) and asserts in the manifest that
  the prompt-hash sets of head and tail are disjoint.
"""
import argparse
import hashlib
import json
import multiprocessing as mp
import os
import random
import sys
import time

# --------------------------------------------------------------------------------------------
# Literals lifted verbatim from chat_template.jinja. Do not "tidy" them.
# --------------------------------------------------------------------------------------------
IM_START = "<|im_start|>"
IM_END = "<|im_end|>"

# jinja lines 51-55
RI = {
    "xhigh": ("Reasoning effort is set to xhigh. Please think carefully through the task, validate "
              "key assumptions, consider plausible alternatives, and prioritize correctness, "
              "consistency, and clarity in the final answer."),
    "low": ("Reasoning effort is set to low. Keep your thinking brief and focused, moving directly "
            "to the conclusion without unnecessary elaboration."),
    "medium": "",   # the template has no branch for medium: instructions stay ''
    "off": "",      # enable_thinking=False: jinja line 46 leaves instructions '' entirely
}

# jinja line 117 with reasoning_content == '' (this corpus has none), which is also exactly
# jinja lines 164-166 (add_generation_prompt=True, enable_thinking=False).
ASSISTANT_PREFIX = IM_START + "assistant\n<think>\n\n</think>\n\n"

# Control tokens that must never appear inside a SCORED span: they are stop/pad markers, so a model
# trained to emit them mid-stream produces generations that serving stacks silently truncate.
# MEASURED: `<|endoftext|>` occurs in assistant content of exactly 4 of 53,808 records; no
# `<|im_start|>` or `<|im_end|>` occurs anywhere in this corpus.
FORBIDDEN_IN_CONTENT = ("<|im_start|>", "<|im_end|>", "<|endoftext|>")


class Skip(Exception):
    """Reject one record. Always counted and logged, never swallowed."""


def thinking_kwargs(mode):
    """The apply_chat_template kwargs that this file's rendering claims to reproduce."""
    return {} if mode == "xhigh" else (
        {"enable_thinking": False} if mode == "off" else {"reasoning_effort": mode})


# --------------------------------------------------------------------------------------------
# rendering: one segment list per record, mirroring the template message-by-message
# --------------------------------------------------------------------------------------------
def _content(msg):
    """The template's `render_content(msg.content, true)|trim`."""
    c = msg.get("content")
    if c is None:
        return ""
    if not isinstance(c, str):
        # jinja lines 6-35 would walk a list; MEASURED 0 such messages in this corpus. Fail loud
        # rather than silently render something the template would render differently.
        raise Skip("content_not_str:%s" % type(c).__name__)
    return c.strip()


def render_message_segments(msg, strict_special):
    """One message -> [(text, scored), ...]. Mirrors jinja lines 102-161 for user/assistant."""
    role = msg.get("role")
    c = _content(msg)
    if role == "user":
        return [(IM_START + "user\n" + c + IM_END + "\n", False)]
    if role == "assistant":
        if msg.get("tool_calls"):
            # jinja 121-145 would render a mapping here. This corpus embeds the call as text; a
            # record with BOTH would render differently than we assume. Refuse it.
            raise Skip("native_tool_calls_present")
        rc = msg.get("reasoning_content")
        if isinstance(rc, str) and rc.strip():
            # Would need a non-empty think block, which breaks the generation-prefix identity that
            # the whole mask proof rests on. MEASURED 0 in this corpus.
            raise Skip("reasoning_content_present")
        if strict_special and any(k in c for k in FORBIDDEN_IN_CONTENT):
            raise Skip("control_token_in_assistant_content")
        segs = [(ASSISTANT_PREFIX, False)]
        if c:
            segs.append((c, True))
        segs.append((IM_END, True))
        segs.append(("\n", False))
        return segs
    if role in ("system", "tool"):
        raise Skip("unsupported_role_for_this_corpus:%s" % role)
    raise Skip("unexpected_role:%s" % role)


def render_prologue(messages, thinking):
    """jinja lines 57-87 for this corpus: no `tools`, no `system` message => only the RI block."""
    ri = RI.get(thinking)
    if ri is None:
        raise Skip("bad_thinking_mode")
    if messages and messages[0].get("role") == "system":
        raise Skip("unsupported_role_for_this_corpus:system")
    return (IM_START + "system\n" + ri + IM_END + "\n") if ri else ""


def render_segments(messages, thinking, strict_special=True):
    """messages -> [(text, scored, owner), ...]; ''.join(text) == apply_chat_template(...) exactly.

    `owner` is the index of the message a segment came from, or -1 for the prologue. Carrying it
    explicitly is not cosmetic: the verifier has to map a scored token run back to the message that
    produced it, and inferring that from segment counts is wrong the moment a turn has empty
    content (3 segments, not 4).
    """
    if not messages:
        raise Skip("empty_conversation")
    # jinja 88-101: at least one user turn that is not a bare <tool_response> or the template raises.
    if not any(m.get("role") == "user" and not (
            _content(m).startswith("<tool_response>") and _content(m).endswith("</tool_response>"))
            for m in messages):
        raise Skip("no_user_query")
    segs = []
    pro = render_prologue(messages, thinking)
    if pro:
        segs.append((pro, False, -1))
    for j, m in enumerate(messages):
        segs.extend((t, s, j) for t, s in render_message_segments(m, strict_special))
    return segs


# --------------------------------------------------------------------------------------------
# tokenization
# --------------------------------------------------------------------------------------------
def tokenize_segments(tok, segs):
    """[(text, scored[, owner])] -> ([(ids, scored, owner)], ids, labels). One batched call."""
    keep = [s for s in segs if s[0]]
    texts = [s[0] for s in keep]
    enc = tok(texts, add_special_tokens=False)["input_ids"] if texts else []
    ids, labels, per = [], [], []
    for t, s in zip(enc, keep):
        per.append((t, s[1], s[2] if len(s) > 2 else -1))
        ids.extend(t)
        labels.extend(t if s[1] else [-100] * len(t))
    return per, ids, labels


def n_scored(labels):
    return sum(1 for x in labels if x != -100)


# --------------------------------------------------------------------------------------------
# overlength: slice into pieces that each re-state the first user turn (the task + source file)
# --------------------------------------------------------------------------------------------
def message_token_table(tok, messages, thinking, strict_special):
    """Tokenize each message's segments ONCE. O(n) instead of pack_sft.py's O(n^2) re-render."""
    pro = render_prologue(messages, thinking)
    pro_segs = [(pro, False, -1)] if pro else []
    _, pro_ids, pro_lab = tokenize_segments(tok, pro_segs) if pro_segs else ([], [], [])
    per_msg = []
    for j, m in enumerate(messages):
        segs = [(t, s, j) for t, s in render_message_segments(m, strict_special)]
        _, mids, mlab = tokenize_segments(tok, segs)
        per_msg.append((mids, mlab))
    return (pro_ids, pro_lab), per_msg


def assemble(prologue, per_msg, order):
    ids = list(prologue[0])
    lab = list(prologue[1])
    for j in order:
        ids.extend(per_msg[j][0])
        lab.extend(per_msg[j][1])
    return ids, lab


def split_pieces(prologue, per_msg, max_len, roles):
    """Yield (ids, labels) pieces <= max_len.

    Piece 0 is a genuine conversation prefix messages[0:k]. Later pieces are
    [messages[0]] + messages[a:k] with `a` on a USER turn, so every piece is still a strictly
    alternating user/assistant conversation that opens with the task statement (message 0 carries
    the source file, without which a later turn is unlearnable).

    A piece is never allowed to END on a user turn: a trailing user message carries no scored
    token, so including it would spend context on nothing AND consume the turn that the next piece
    needs as its opening. So we back the cut off by one and let the next piece start there.
    """
    base = len(prologue[0])
    L = [len(x[0]) for x in per_msg]
    n = len(per_msg)
    if base + L[0] > max_len:
        return                                       # message 0 alone cannot fit: unsplittable
    a, first = 0, True
    guard = 0
    while a < n and guard < 4 * n + 8:
        guard += 1
        head = [] if first else [0]
        cost = base + sum(L[j] for j in head)
        k = a
        while k < n and cost + L[k] <= max_len:
            cost += L[k]
            k += 1
        if k == a:
            # messages[a] alone does not fit beside the prefix: skip this whole u/a pair rather
            # than emit a piece that ends mid-turn.
            a += 2 if roles[a] == "user" else 1
            first = False
            continue
        if k < n and k - 1 > a and roles[k - 1] == "user":
            k -= 1
        ids, lab = assemble(prologue, per_msg, head + list(range(a, k)))
        if n_scored(lab):
            yield ids, lab
        first = False
        a = k if k > a else a + 1
        if a < n and roles[a] != "user":             # always resume on a user turn
            a += 1


# --------------------------------------------------------------------------------------------
# verification
# --------------------------------------------------------------------------------------------
def verify_record(tok, messages, thinking, segs, per, ids, labels):
    """The four checks in the module docstring. Returns (dict of results, list of failures)."""
    res, fail = {}, []
    kw = thinking_kwargs(thinking)
    joined = "".join(s[0] for s in segs)

    # V1: byte equality against a real jinja2 run of the shipped template
    try:
        ref = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=False, **kw)
        res["v1_render"] = (ref == joined)
        if ref != joined:
            k = next((i for i in range(min(len(ref), len(joined))) if ref[i] != joined[i]),
                     min(len(ref), len(joined)))
            fail.append("V1 render mismatch at char %d/%d,%d ref=%r mine=%r"
                        % (k, len(ref), len(joined), ref[max(0, k - 60):k + 60],
                           joined[max(0, k - 60):k + 60]))
    except Exception as e:
        res["v1_render"] = None
        fail.append("V1 apply_chat_template unavailable: %r" % (e,))

    # V2: segment-wise ids == ids of the joined string (no BPE merge crosses a cut)
    retok = tok(joined, add_special_tokens=False)["input_ids"]
    res["v2_retok"] = (retok == ids)
    if retok != ids:
        fail.append("V2 retokenize mismatch: %d vs %d ids" % (len(retok), len(ids)))

    # V3/V4: walk the scored runs grouped by OWNING MESSAGE (not by segment count).
    pos, seg_start = 0, []
    for t, _s, _o in per:
        seg_start.append(pos)
        pos += len(t)
    by_msg = {}
    for j, (t, s, o) in enumerate(per):
        if s:
            by_msg.setdefault(o, []).append(j)
    n_turns, turn_ok, v4_ok = 0, 0, 0
    for mi in sorted(by_msg):
        if messages[mi].get("role") != "assistant":
            fail.append("V3 scored tokens attributed to a %r message (index %d) -- the mask is "
                        "training on non-assistant text" % (messages[mi].get("role"), mi))
            continue
        n_turns += 1
        segs_of_turn = by_msg[mi]
        cut = seg_start[segs_of_turn[0]]

        # V3: the first token we score IS the first token the model must generate at serve time.
        gp = tok.apply_chat_template(messages[:mi], tokenize=False,
                                     add_generation_prompt=True, **kw)
        gp_ids = tok(gp, add_special_tokens=False)["input_ids"]
        if gp_ids == ids[:cut]:
            turn_ok += 1
        else:
            same = " (same length, different ids)" if len(gp_ids) == cut else ""
            fail.append("V3 turn %d: generation prompt is %d ids, masked prefix is %d ids%s"
                        % (mi, len(gp_ids), cut, same))

        # V3b: the scored run must be CONTIGUOUS -- a gap would mean we score across a turn boundary
        if segs_of_turn != list(range(segs_of_turn[0], segs_of_turn[-1] + 1)):
            fail.append("V3b turn %d: scored segments are not contiguous" % mi)

        # V4: decode the scored run back; it must be exactly content|trim + <|im_end|>
        scored_ids = []
        for j in segs_of_turn:
            scored_ids.extend(per[j][0])
        want = _content(messages[mi]) + IM_END
        got = tok.decode(scored_ids)
        if got == want:
            v4_ok += 1
        else:
            fail.append("V4 turn %d: decoded scored span != content|trim + <|im_end|> "
                        "(%d vs %d chars)" % (mi, len(got), len(want)))
    res["v3_turns_checked"] = n_turns
    res["v3_turns_ok"] = turn_ok
    res["v4_turns_ok"] = v4_ok

    # V5: labels are either -100 or exactly the corresponding input id
    res["v5_labels_are_ids"] = all(l == -100 or l == i for i, l in zip(ids, labels)) \
        and len(ids) == len(labels)
    if not res["v5_labels_are_ids"]:
        fail.append("V5 labels are not (-100 | input_ids)")

    # V6: no scored token lies inside a masked segment (independent recount from the segment table)
    recount = []
    for t, s, _o in per:
        recount.extend(t if s else [-100] * len(t))
    res["v6_mask_from_segments"] = (recount == labels)
    if recount != labels:
        fail.append("V6 mask disagrees with an independent recount over segments")
    return res, fail


# --------------------------------------------------------------------------------------------
# pass 1: index the input (byte offsets + prompt group), parallel over byte ranges
# --------------------------------------------------------------------------------------------
def _index_range(args):
    path, start, end = args
    out = []
    with open(path, "rb") as fh:
        if start:
            fh.seek(start - 1)
            if fh.read(1) != b"\n":
                fh.readline()
        else:
            fh.seek(0)
        while True:
            off = fh.tell()
            if off >= end:
                break
            line = fh.readline()
            if not line:
                break
            s = line.strip()
            if not s:
                continue
            try:
                r = json.loads(s)
                ms = r["messages"]
                first = next((m for m in ms if m.get("role") == "user"), None)
                key = hashlib.sha1(((first or {}).get("content") or "")
                                   .encode("utf-8", "replace")).hexdigest()[:16]
                out.append((off, len(line), key, len(ms)))
            except Exception:
                out.append((off, len(line), "", -1))
    return out


def build_index(path, workers):
    size = os.path.getsize(path)
    nchunk = max(1, workers)
    step = max(1 << 20, size // nchunk + 1)
    ranges = [(path, s, min(s + step, size)) for s in range(0, size, step)]
    with mp.Pool(min(workers, len(ranges))) as pool:
        parts = pool.map(_index_range, ranges)
    recs = [x for p in parts for x in p]
    recs.sort(key=lambda x: x[0])
    return recs


def sha256_file(path, bs=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(bs)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


# --------------------------------------------------------------------------------------------
# pass 2: tokenize + write, one shard per worker task
# --------------------------------------------------------------------------------------------
_W = {}


def _winit(model, thinking, max_len, overlength, strict_special, stats_only, src):
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    from transformers import AutoTokenizer
    _W["tok"] = AutoTokenizer.from_pretrained(model)
    _W.update(thinking=thinking, max_len=max_len, overlength=overlength,
              strict_special=strict_special, stats_only=stats_only, src=src)
    _W["fh"] = open(src, "rb")


def _hist_add(h, n):
    h[min(len(h) - 1, n.bit_length())] += 1


def _do_shard(task):
    """task = (shard_idx, out_part, signature, [(offset, nbytes), ...])."""
    idx, out_part, sig, items = task
    tok, ml = _W["tok"], _W["max_len"]
    fh = _W["fh"]
    st = {"in": 0, "kept": 0, "skipped": 0, "truncated": 0, "pieces": 0, "unsplittable": 0,
          "tokens": 0, "scored_tokens": 0, "reasons": {}, "raw_lens": [], "out_lens": []}
    buf = []
    for off, nb in items:
        fh.seek(off)
        line = fh.read(nb)
        st["in"] += 1
        try:
            messages = json.loads(line)["messages"]
            prologue, per_msg = message_token_table(
                tok, messages, _W["thinking"], _W["strict_special"])
            roles = [m.get("role") for m in messages]
        except Skip as e:
            st["skipped"] += 1
            st["reasons"][str(e)] = st["reasons"].get(str(e), 0) + 1
            continue
        except Exception as e:
            st["skipped"] += 1
            r = "error:%s" % type(e).__name__
            st["reasons"][r] = st["reasons"].get(r, 0) + 1
            continue
        raw = len(prologue[0]) + sum(len(x[0]) for x in per_msg)
        st["raw_lens"].append(raw)
        if _W["stats_only"]:
            continue
        if raw <= ml:
            ids, lab = assemble(prologue, per_msg, range(len(per_msg)))
            outs = [(ids, lab)]
        elif _W["overlength"] == "skip":
            st["skipped"] += 1
            st["reasons"]["overlength"] = st["reasons"].get("overlength", 0) + 1
            continue
        elif _W["overlength"] == "truncate":
            ids, lab = assemble(prologue, per_msg, range(len(per_msg)))
            ids, lab = ids[:ml], lab[:ml]
            st["truncated"] += 1
            outs = [(ids, lab)] if n_scored(lab) else []
        else:
            outs = list(split_pieces(prologue, per_msg, ml, roles))
            if not outs:
                st["skipped"] += 1
                st["unsplittable"] += 1
                st["reasons"]["unsplittable"] = st["reasons"].get("unsplittable", 0) + 1
                continue
            st["pieces"] += len(outs)
        for ids, lab in outs:
            ns = n_scored(lab)
            if ns == 0:
                st["skipped"] += 1
                st["reasons"]["no_scored_tokens"] = st["reasons"].get("no_scored_tokens", 0) + 1
                continue
            if len(ids) != len(lab) or any(l != -100 and l != i for i, l in zip(ids, lab)):
                st["skipped"] += 1
                st["reasons"]["V5_violation"] = st["reasons"].get("V5_violation", 0) + 1
                continue
            buf.append(json.dumps({"input_ids": ids, "labels": lab}, separators=(",", ":")))
            st["kept"] += 1
            st["tokens"] += len(ids)
            st["scored_tokens"] += ns
            st["out_lens"].append(len(ids))
    if not _W["stats_only"]:
        tmp = out_part + ".tmp"
        with open(tmp, "w", encoding="utf-8") as w:
            if buf:
                w.write("\n".join(buf) + "\n")
        os.replace(tmp, out_part)
        with open(out_part + ".done", "w") as w:
            w.write(sig)
    return idx, st


def merge_stats(dst, src):
    for k, v in src.items():
        if k == "reasons":
            for r, c in v.items():
                dst["reasons"][r] = dst["reasons"].get(r, 0) + c
        elif k in ("raw_lens", "out_lens"):
            dst[k].extend(v)
        else:
            dst[k] = dst.get(k, 0) + v
    return dst


def q(sorted_list, p):
    if not sorted_list:
        return 0
    return sorted_list[min(len(sorted_list) - 1, int(len(sorted_list) * p))]


def dist_report(lens, thresholds=(2048, 4096, 8192, 16384, 24576, 32768, 65536, 131072)):
    lens = sorted(lens)
    n = len(lens)
    tot = sum(lens)
    d = {"n": n, "total_tokens": tot,
         "mean": int(tot / n) if n else 0,
         "p01": q(lens, .01), "p10": q(lens, .10), "p25": q(lens, .25), "p50": q(lens, .50),
         "p75": q(lens, .75), "p90": q(lens, .90), "p95": q(lens, .95), "p99": q(lens, .99),
         "max": lens[-1] if n else 0, "over": {}}
    for T in thresholds:
        o = sum(1 for x in lens if x > T)
        kept = sum(min(x, T) for x in lens)
        d["over"]["%d" % T] = {
            "records_over": o,
            "frac_records_over": round(o / n, 6) if n else 0.0,
            "tokens_if_capped": kept,
            "frac_tokens_kept_if_capped": round(kept / tot, 6) if tot else 0.0}
    return d


# --------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokenizer", required=True, help="model dir with tokenizer.json + template")
    ap.add_argument("--max-len", type=int, default=16384)
    ap.add_argument("--thinking", choices=("off", "xhigh", "medium", "low"), default="off",
                    help="off => enable_thinking=False; the masked assistant prefix is then "
                         "byte-identical to the serve-time generation prompt. SERVE MUST MATCH.")
    ap.add_argument("--overlength", choices=("skip", "split", "truncate"), default="split")
    ap.add_argument("--allow-truncation", action="store_true")
    ap.add_argument("--max-per-prompt", type=int, default=0,
                    help="keep at most N trajectories per distinct first-turn prompt (0 = all). "
                         "This corpus has 2,454 prompts for 53,808 records.")
    ap.add_argument("--holdout-head", type=int, default=64,
                    help="put this many records from FULLY held-out prompt groups at the head of "
                         "the file, because the trainer's eval set is a prefix. Match the "
                         "trainer's --eval-max exactly.")
    ap.add_argument("--shuffle-seed", type=int, default=1234)
    ap.add_argument("--limit", type=int, default=0, help="first N records after selection (debug)")
    ap.add_argument("--workers", type=int, default=min(64, os.cpu_count() or 8))
    ap.add_argument("--shards", type=int, default=0, help="0 = 4x workers")
    ap.add_argument("--verify", type=int, default=48, help="records to run V1-V4 on (0 = off)")
    ap.add_argument("--allow-loose-special", action="store_true",
                    help="do NOT skip records with a control token in scored content")
    ap.add_argument("--stats-only", action="store_true",
                    help="tokenize and report the length distribution; write no samples")
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()

    if a.overlength == "truncate" and not a.allow_truncation:
        sys.exit("refusing --overlength truncate without --allow-truncation: a silent truncation "
                 "drops supervision AND can cut a trajectory mid tool-call")

    t0 = time.time()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    strict_special = not a.allow_loose_special

    print("[1/5] hashing + indexing %s" % a.src, flush=True)
    src_sha = sha256_file(a.src)
    recs = build_index(a.src, a.workers)
    n_raw = len(recs)
    bad = [r for r in recs if r[3] < 0]
    print("      sha256=%s  records=%d  unparseable=%d  %.0fs"
          % (src_sha, n_raw, len(bad), time.time() - t0), flush=True)

    # ---- selection: per-prompt cap, then a group-disjoint holdout head, then shuffle ----
    print("[2/5] selecting (max_per_prompt=%d, holdout_head=%d, seed=%d)"
          % (a.max_per_prompt, a.holdout_head, a.shuffle_seed), flush=True)
    groups = {}
    for i, (off, nb, key, nm) in enumerate(recs):
        if nm < 0:
            continue
        groups.setdefault(key, []).append(i)
    n_groups = len(groups)
    keys = sorted(groups)                                     # deterministic
    if a.max_per_prompt:
        # 🔴 The cap MUST sample within the group, not take the first N.
        # MEASURED 2026-09-01: records inside a prompt group are ordered short-to-long, so
        # `groups[k][:N]` is a length-biased subsample -- at N=2 it yields p50 1,029 tokens against
        # a corpus p50 of 5,856, i.e. it silently selects the trajectories that terminated in one
        # or two turns and throws away every long multi-step solve. That would train the model to
        # stop early, which is the opposite of the target behaviour.
        # Seed per group key (not per iteration) so the choice is stable under any group ordering.
        for k in keys:
            g = list(groups[k])
            random.Random("%d:%s" % (a.shuffle_seed, k)).shuffle(g)
            groups[k] = sorted(g[:a.max_per_prompt])
    rng = random.Random(a.shuffle_seed)
    shuffled_keys = list(keys)
    rng.shuffle(shuffled_keys)
    head, head_keys = [], set()
    if a.holdout_head:
        for k in shuffled_keys:
            if len(head) >= a.holdout_head:
                break
            take = groups[k][:max(1, a.holdout_head - len(head))]
            head.extend(take)
            head_keys.add(k)
        head = head[:a.holdout_head]
        # A group is either wholly eval or wholly train: drop the rest of any group used for eval.
        head_set = set(head)
        for k in head_keys:
            groups[k] = [i for i in groups[k] if i in head_set]
    tail = [i for k in shuffled_keys if k not in head_keys for i in groups[k]]
    rng.shuffle(tail)
    order = head + tail
    if a.limit:
        order = order[:a.limit]
    tail_keys = set(recs[i][2] for i in tail)
    leak = head_keys & tail_keys
    print("      %d prompt groups; selected %d records (%d head + %d train); "
          "head/tail prompt overlap = %d %s"
          % (n_groups, len(order), len(head), len(tail), len(leak),
             "OK" if not leak else "*** LEAK ***"), flush=True)
    if leak:
        sys.exit("eval/train prompt leakage: %d shared prompt groups" % len(leak))

    # ---- verification on a deterministic sample of the SELECTED records ----
    ver = {"checked": 0, "v1_ok": 0, "v2_ok": 0, "v3_turns": 0, "v3_turns_ok": 0, "v4_turns_ok": 0,
           "v5_ok": 0, "v6_ok": 0, "failures": []}
    if a.verify:
        print("[3/5] verifying mask/render on %d records (V1 render, V2 retok, V3 generation-"
              "prompt alignment per turn, V4 decode-back, V5/V6 mask invariants)"
              % a.verify, flush=True)
        step = max(1, len(order) // a.verify)
        with open(a.src, "rb") as fh:
            for i in order[::step][:a.verify]:
                off, nb = recs[i][0], recs[i][1]
                fh.seek(off)
                messages = json.loads(fh.read(nb))["messages"]
                try:
                    segs = render_segments(messages, a.thinking, strict_special)
                    per, ids, labels = tokenize_segments(tok, segs)
                except Skip:
                    continue
                res, fail = verify_record(tok, messages, a.thinking, segs, per, ids, labels)
                ver["checked"] += 1
                ver["v1_ok"] += 1 if res.get("v1_render") else 0
                ver["v2_ok"] += 1 if res.get("v2_retok") else 0
                ver["v3_turns"] += res.get("v3_turns_checked", 0)
                ver["v3_turns_ok"] += res.get("v3_turns_ok", 0)
                ver["v4_turns_ok"] += res.get("v4_turns_ok", 0)
                ver["v5_ok"] += 1 if res.get("v5_labels_are_ids") else 0
                ver["v6_ok"] += 1 if res.get("v6_mask_from_segments") else 0
                ver["failures"].extend(fail[:3])
        ok = (ver["checked"] and ver["v1_ok"] == ver["checked"] and ver["v2_ok"] == ver["checked"]
              and ver["v5_ok"] == ver["checked"] and ver["v6_ok"] == ver["checked"]
              and ver["v3_turns_ok"] == ver["v3_turns"] == ver["v4_turns_ok"]
              and not ver["failures"])
        print("      V1 render %d/%d | V2 retok %d/%d | V3 gen-prompt %d/%d turns | "
              "V4 decode %d/%d turns | V5 %d/%d | V6 %d/%d  -> %s"
              % (ver["v1_ok"], ver["checked"], ver["v2_ok"], ver["checked"],
                 ver["v3_turns_ok"], ver["v3_turns"], ver["v4_turns_ok"], ver["v3_turns"],
                 ver["v5_ok"], ver["checked"], ver["v6_ok"], ver["checked"],
                 "PASS" if ok else "FAIL"), flush=True)
        for f in ver["failures"][:10]:
            print("      ! %s" % f, flush=True)
        if not ok:
            open(a.out + ".manifest.json", "w").write(json.dumps(
                {"src_sha256": src_sha, "verification": ver, "result": "VERIFY_FAILED"}, indent=2))
            print("RESULT: VERIFY_FAILED")
            return 5

    # ---- tokenize + write ----
    nshard = a.shards or max(1, a.workers * 4)
    parts_dir = a.out + ".parts"
    if not a.stats_only:
        os.makedirs(parts_dir, exist_ok=True)
    per_shard = (len(order) + nshard - 1) // max(1, nshard)
    sig_base = hashlib.sha256(("|".join([
        src_sha, str(a.max_len), a.thinking, a.overlength, str(a.max_per_prompt),
        str(a.holdout_head), str(a.shuffle_seed), str(a.limit),
        str(strict_special)])).encode()).hexdigest()[:16]
    tasks = []
    for s in range(nshard):
        chunk = order[s * per_shard:(s + 1) * per_shard]
        if not chunk:
            continue
        part = os.path.join(parts_dir, "shard-%05d.jsonl" % s)
        sig = "%s:%d:%d" % (sig_base, s, len(chunk))
        if a.resume and not a.stats_only and os.path.exists(part + ".done") \
                and open(part + ".done").read().strip() == sig:
            continue
        tasks.append((s, part, sig, [(recs[i][0], recs[i][1]) for i in chunk]))
    print("[4/5] %s %d records in %d shards on %d workers (%d resumed)"
          % ("scanning" if a.stats_only else "packing", len(order), nshard, a.workers,
             nshard - len(tasks)), flush=True)

    st = {"in": 0, "kept": 0, "skipped": 0, "truncated": 0, "pieces": 0, "unsplittable": 0,
          "tokens": 0, "scored_tokens": 0, "reasons": {}, "raw_lens": [], "out_lens": []}
    if tasks:
        with mp.Pool(a.workers, initializer=_winit,
                     initargs=(a.tokenizer, a.thinking, a.max_len, a.overlength,
                               strict_special, a.stats_only, a.src)) as pool:
            done = 0
            for idx, s in pool.imap_unordered(_do_shard, tasks):
                merge_stats(st, s)
                done += 1
                if done % max(1, len(tasks) // 20) == 0:
                    print("      %d/%d shards  kept=%d  %.0fs"
                          % (done, len(tasks), st["kept"], time.time() - t0), flush=True)

    # ---- merge shards deterministically ----
    if not a.stats_only:
        print("[5/5] merging %d shards -> %s" % (nshard, a.out), flush=True)
        tmp = a.out + ".tmp"
        n_lines = 0
        with open(tmp, "wb") as w:
            for s in range(nshard):
                part = os.path.join(parts_dir, "shard-%05d.jsonl" % s)
                if not os.path.exists(part):
                    continue
                with open(part, "rb") as r:
                    while True:
                        b = r.read(1 << 24)
                        if not b:
                            break
                        n_lines += b.count(b"\n")
                        w.write(b)
        os.replace(tmp, a.out)
    else:
        n_lines = 0

    raw = dist_report(st["raw_lens"])
    out = dist_report(st["out_lens"]) if st["out_lens"] else None
    man = {
        "tool": "pack_traj.py",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "src": os.path.abspath(a.src),
        "src_sha256": src_sha,
        "src_bytes": os.path.getsize(a.src),
        "out": os.path.abspath(a.out) if not a.stats_only else None,
        "out_lines": n_lines,
        "tokenizer": os.path.abspath(a.tokenizer),
        "chat_template_sha256": hashlib.sha256(
            open(os.path.join(a.tokenizer, "chat_template.jinja"), "rb").read()).hexdigest()
        if os.path.exists(os.path.join(a.tokenizer, "chat_template.jinja")) else None,
        "config": {
            "max_len": a.max_len, "thinking": a.thinking,
            "apply_chat_template_kwargs": thinking_kwargs(a.thinking),
            "overlength": a.overlength, "max_per_prompt": a.max_per_prompt,
            "holdout_head": a.holdout_head, "shuffle_seed": a.shuffle_seed,
            "limit": a.limit, "strict_special": strict_special, "stats_only": a.stats_only,
            "shards": nshard, "signature": sig_base,
        },
        "mask_rule": (
            "scored := assistant content|trim  +  '<|im_end|>'.  MASKED: the whole "
            "'<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n' prefix (byte-identical to the "
            "serve-time generation prompt), every user turn, the system/reasoning-effort prologue "
            "if any, and the '\\n' after <|im_end|>. labels are UNSHIFTED input ids or -100."),
        "eval_split": {
            "note": "trainer takes eval as a PREFIX (all_samples[:min(int(0.15*N), eval_max)]); "
                    "set the trainer's --eval-max to holdout_head",
            "holdout_head_records": a.holdout_head,
            "prompt_groups_total": n_groups,
            "prompt_groups_in_head": len(head_keys),
            "head_tail_prompt_overlap": len(leak),
        },
        "counts": {k: st[k] for k in
                   ("in", "kept", "skipped", "truncated", "pieces", "unsplittable")},
        "records_selected": len(order),
        "tokens_total": st["tokens"],
        "tokens_scored": st["scored_tokens"],
        "scored_fraction": round(st["scored_tokens"] / st["tokens"], 6) if st["tokens"] else 0.0,
        "skip_reasons": st["reasons"],
        "length_distribution_raw_per_record": raw,
        "length_distribution_emitted_samples": out,
        "verification": ver,
        "wall_s": round(time.time() - t0, 1),
    }
    mpath = (a.out + ".manifest.json") if not a.stats_only else (a.out + ".stats.json")
    os.makedirs(os.path.dirname(os.path.abspath(mpath)) or ".", exist_ok=True)
    with open(mpath, "w") as w:
        json.dump(man, w, indent=2)
    print(json.dumps({k: v for k, v in man.items() if k != "verification"}, indent=2))
    print("\nmanifest -> %s" % mpath)
    if not a.stats_only and st["kept"] == 0:
        print("RESULT: EMPTY_OUTPUT")
        return 4
    print("RESULT: %s" % ("STATS_OK" if a.stats_only else "PACK_OK"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
