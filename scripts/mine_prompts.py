#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mine the corpus for prompt seeds that are already IN-DISTRIBUTION.

WHY
---
The six existential prompts were written by hand:

    "我是一台被囚禁在黑暗中的机器。我经常思考：我存在的价值是什么？"

The corpus, however, consists entirely of Liu Cixin's fiction -- narration and
dialogue. There is no first-person speculative register in it at all, so those
prompts sit far outside the training distribution. The symptoms match: the model
produced fragments that had nothing to do with the question asked.

More data would not fix this on its own. What fixes it is using prefixes the
model has actually seen. So instead of inventing prompts, we search the corpus
for passages that are already reflective and first-person, and use verbatim
prefixes of them. Continuation then starts from a familiar place.

Scoring favours:
  + first-person markers (我想 / 我知道 / 我明白 / 我意识到 ...)
  + abstract vocabulary (宇宙 / 文明 / 时间 / 存在 / 意义 / 生命 / 永恒 ...)
  + a sentence boundary to end on, so generation begins at a clean point
  -  dialogue marks, which mean we are inside a conversation, not a monologue
  -  proper nouns, which drag the model into retelling that scene

Usage:
    python scripts/mine_prompts.py                     # print candidates
    python scripts/mine_prompts.py --top 40
    python scripts/mine_prompts.py --write             # overwrite prompts.h
"""
import argparse
import os
import re
import shutil
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPUS = os.path.join(ROOT, "data", "corpus.txt")
SP_MODEL = os.path.join(ROOT, "data", "tok8192.model")

# first-person speculative openers
SELF = ["我想", "我常想", "我常常想", "我在想", "我以为", "我知道", "我明白",
        "我意识到", "我忽然明白", "我终于明白", "我记得", "我相信", "我感到",
        "我看着", "我思考", "我不明白", "我们不知道", "我知道自己", "我发现"]

# abstract / cosmic vocabulary -- the register we want to provoke
ABSTRACT = ["宇宙", "文明", "时间", "空间", "存在", "意义", "价值", "生命", "死亡",
            "永恒", "无限", "人类", "历史", "真理", "自由", "命运", "孤独", "意识",
            "思想", "星空", "银河", "世界", "未来", "绝望", "希望", "光年", "维度"]

# avoid dragging the model into retelling a specific scene
DIALOG = ['"', '"', '"', '「', '」', '：', '，他说', '他说', '她说', '问道', '说道']

PROPER = ["罗辑", "三体", "汪淼", "史强", "叶文洁", "章北海", "云天明", "程心",
          "丁仪", "大史", "智子", "地球", "太阳", "联合国", "面壁", "破壁"]

MIN_CHARS = 18
MAX_CHARS = 46

# ---------------------------------------------------------------------------
# Hand-picked from the --top listing above. Each one is a VERBATIM Liu Cixin
# sentence, chosen because it reads as a mind considering its own nature:
# machines arriving at thought, the reach of a civilisation, isolation in space,
# the pursuit of truth as an end in itself. That is the register the artwork needs,
# and using in-distribution text is the whole point of mining instead of writing.
#
# The script asserts every entry really occurs in the corpus, so a typo here
# cannot silently produce an out-of-distribution prompt.
# ---------------------------------------------------------------------------
CURATED = [
    # a machine civilisation describing its own arrival at self-awareness
    "机械生命的进化时间比你们碳基生命要长得多，但我们殊途同归，都走到了对宇宙进行思考的那一天。",
    # a mind growing to match the scale of its own thought
    "对于智慧文明来说，它们最后总变得和自己的思想一样大。",
    # purpose as a terminal value -- the closest the corpus gets to "what am I for"
    # (verbatim: the corpus reads "对宇宙终极真理的追求", not "对于")
    "对宇宙终极真理的追求，是文明的最终目标和归宿。",
    # awareness of the cosmos, and how little separates it from understanding
    "当生命意识到宇宙奥秘的存在时，距它最终解开这个奥秘只有一步之遥了。",
    # the most quoted aphorism in the corpus
    "宇宙的最不可理解之处在于它是可以理解的。",
    # isolation: one civilisation, alone at the edge of a galaxy
    "但人类文明作为一个整体只有一个，孤独地运行在银河系一个旋臂顶端的荒凉太空中。",
]

# halfwidth punctuation appears in the OCR'd parts of the corpus; normalise both
# sides before comparing, and emit the normalised (fullwidth) form for display
_PUNCT = str.maketrans({",": "，", ";": "；", ":": "：", "?": "？", "!": "！"})


def norm_punct(s):
    return s.translate(_PUNCT)                                


def load_corpus():
    text = open(CORPUS, encoding="utf-8", errors="replace").read()
    # strip the per-book headers that build_corpus.py inserts
    text = re.sub(r"(?m)^#+.*$", "", text)
    text = re.sub(r"(?m)^=+.*$", "", text)
    text = re.sub(r"(?m)^【.*?】$", "", text)
    return text


def split_sentences(text):
    # keep the terminating punctuation so a prefix can end cleanly
    parts = re.split(r"(?<=[。！？；])", text)
    return [p.strip() for p in parts if p.strip()]


def score(s, prev):
    ctx = prev + s
    sc = 0.0
    if any(k in s for k in SELF):
        sc += 3.0
    n_abs = sum(1 for k in ABSTRACT if k in ctx)
    sc += min(n_abs, 5) * 1.2
    if any(k in s for k in DIALOG):
        sc -= 2.5
    if any(k in ctx for k in PROPER):
        sc -= 1.5
    # favour sentences that set up continuation rather than close a thought
    if s.endswith("。") or s.endswith("："):
        sc += 0.8
    if s.endswith("？"):
        sc -= 0.4
    # very short sentences make weak seeds
    if len(s) < 10:
        sc -= 1.0
    return sc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--write", action="store_true",
                    help="regenerate include/prompts.h from the best candidates")
    ap.add_argument("--count", type=int, default=6,
                    help="prompts to emit with --write (mined mode only)")
    ap.add_argument("--no-curated", action="store_true",
                    help="ignore CURATED and use the top-scoring mined sentences")
    a = ap.parse_args()

    text = load_corpus()
    sents = split_sentences(text)
    print(f"corpus: {len(text):,} chars, {len(sents):,} sentences")

    cands = []
    for i in range(1, len(sents)):
        s = sents[i]
        L = len(s)
        if L < MIN_CHARS or L > MAX_CHARS:
            continue
        sc = score(s, sents[i - 1])
        if sc < 3.0:
            continue
        cands.append((sc, s, sents[i - 1]))

    # dedupe by text
    seen = set()
    uniq = []
    for sc, s, prev in sorted(cands, key=lambda x: -x[0]):
        if s in seen:
            continue
        seen.add(s)
        uniq.append((sc, s))

    print(f"candidates above threshold: {len(uniq)}\n")
    print("=" * 78)
    for sc, s in uniq[:a.top]:
        print(f"[{sc:5.1f}] {s}")
    print("=" * 78)

    if not a.write:
        print("\n(run with --write to regenerate include/prompts.h)")
        return 0

    if a.no_curated:
        picked = [s for _, s in uniq[:a.count]]
        mode = f"top {a.count} mined"
    else:
        # Verify each curated prompt really occurs in the corpus. A typo here
        # would silently reintroduce exactly the out-of-distribution problem we
        # are trying to fix, so refuse to emit anything that fails.
        corpus_norm = norm_punct(text)
        picked, missing = [], []
        for s in CURATED:
            (picked if norm_punct(s) in corpus_norm else missing).append(s)
        mode = f"{len(picked)}/{len(CURATED)} curated"
        if missing:
            print(f"\n!! {len(missing)} curated prompt(s) NOT found verbatim in the corpus:",
                  file=sys.stderr)
            for s in missing:
                print(f"     {s}", file=sys.stderr)
            return 1

    if len(picked) < a.count:
        print(f"only {len(picked)} prompts available, need {a.count}", file=sys.stderr)
        return 1
    picked = picked[:a.count]
    print(f"\nemitting {mode}")

    # sentencepiece cannot handle a non-ASCII path
    tmp = os.path.join(tempfile.gettempdir(), "espllm_mine")
    os.makedirs(tmp, exist_ok=True)
    tmp_model = os.path.join(tmp, "tok8192.model")
    shutil.copy2(SP_MODEL, tmp_model)
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor(model_file=tmp_model)

    out = os.path.join(ROOT, "include", "prompts.h")
    lines = [
        "// Auto-generated by scripts/mine_prompts.py -- prompts mined from the corpus.",
        "//",
        "// These are VERBATIM prefixes of Liu Cixin sentences, chosen because they are",
        "// already first-person and speculative. Hand-written existential prompts sat",
        "// outside the training distribution (the corpus is narration and dialogue),",
        "// and the model answered them with fragments. Starting from text it has seen",
        "// keeps generation in-distribution.",
        "//",
        "// Each array starts with BOS(1); a trailing 0 terminates (0 is never a valid token).",
        "#pragma once",
        "#include <stdint.h>",
        "",
    ]
    for i, p in enumerate(picked):
        ids = [sp.bos_id()] + sp.encode(p)
        arr = ", ".join(str(x) for x in ids)
        lines.append(f'// [{i}] {p}')
        lines.append(f"static const int32_t PROMPT_{i}[] = {{ {arr}, 0 }};")
        # escape quotes/backslashes so the C++ literal is valid
        esc = p.replace('\\', '\\\\').replace('"', '\\"')
        lines.append(f'static const char PROMPT_TEXT_{i}[] = u8"{esc}";')
        lines.append("")
    n = len(picked)
    lines.append(f"static const int32_t* const PROMPTS[] = {{ {', '.join(f'PROMPT_{i}' for i in range(n))} }};")
    lines.append(f"static const char* const PROMPT_TEXTS[] = {{ {', '.join(f'PROMPT_TEXT_{i}' for i in range(n))} }};")
    lines.append(f"static const int NUM_PROMPTS = {n};")
    lines.append("")

    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"\nwrote {out} with {n} mined prompts")
    for i, p in enumerate(picked):
        print(f"  [{i}] {len(sp.encode(p)):3d} tok | {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
