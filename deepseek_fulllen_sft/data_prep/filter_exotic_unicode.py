#!/usr/bin/env python3
"""filter_exotic_unicode.py -- drop records whose assistant content trips pack_traj's V4 decode-back.

Owner directive 2026-09-15: for glm-5.3-deepseek, "这种 exotic 的直接删掉就好" -- rather than relax the
strict V4 mask check, remove the handful of records with exotic Unicode (emoji / CJK / combining marks
/ µ / Å) where tok.decode(tok(content)+tok(<|im_end|>)) != content|trim + <|im_end|> (a tokenizer
normalization round-trip diff of a few chars; the mask tokens are still exactly tok(content)+tok(im_end),
but the owner prefers dropping these rows to keeping strict V4 pristine). Writes a cleaned chat.jsonl.
"""
import json, os, sys

R = "/home/i-dingleilei/agentic_task_labeling/pipeline/data_select_2609"
SRC = sys.argv[1] if len(sys.argv) > 1 else R + "/sft/raw/glm-5.3-deepseek/chat.jsonl"
OUT = sys.argv[2] if len(sys.argv) > 2 else R + "/sft/raw/glm-5.3-deepseek/chat_clean.jsonl"
TOK = "/mnt/ws-jfs/posttrain/i-xuyiming/paper/rsi_2608/qwen-3.8-27B/model/Qwen3.8-27B"

os.environ["HF_HUB_OFFLINE"] = "1"; os.environ["TRANSFORMERS_OFFLINE"] = "1"
sys.path.insert(0, R + "/sft")
import pack_traj as P
from transformers import AutoTokenizer
tk = AutoTokenizer.from_pretrained(TOK, trust_remote_code=True)
imend_ids = tk(P.IM_END, add_special_tokens=False)["input_ids"]

lines = open(SRC).readlines()
kept, dropped, drop_idx = [], 0, []
for i, line in enumerate(lines):
    try:
        msgs = json.loads(line).get("messages", [])
    except Exception:
        dropped += 1; drop_idx.append((i, "unparseable")); continue
    ok = True; why = ""
    for mi, m in enumerate(msgs):
        if m.get("role") != "assistant":
            continue
        c = P._content(m)  # trimmed content
        if not c:
            continue
        sid = tk(c, add_special_tokens=False)["input_ids"] + imend_ids
        if tk.decode(sid) != c + P.IM_END:
            ok = False; why = "V4-fail turn %d" % mi; break
    if ok:
        kept.append(line)
    else:
        dropped += 1; drop_idx.append((i, why))
    if (i + 1) % 500 == 0:
        print("  scanned %d/%d  kept=%d dropped=%d" % (i + 1, len(lines), len(kept), dropped), flush=True)

with open(OUT, "w") as f:
    f.writelines(kept)
print("DONE: in=%d kept=%d dropped=%d -> %s" % (len(lines), len(kept), dropped, OUT))
print("dropped indices (first 20):", drop_idx[:20])
