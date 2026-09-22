#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sanity-check include/prompts.h.

The device displays PROMPT_TEXT_i on the OLED while it actually feeds PROMPT_i
to the model. If those two disagree, the artwork would show the AI "thinking"
something different from what it is really conditioned on -- a silent, hard to
spot correctness bug. This script re-encodes every text literal and asserts it
reproduces the token array exactly.

Usage: python scripts/verify_prompts.py
"""
import io
import os
import re
import shutil
import sys
import tempfile

import sentencepiece as spm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HEADER = os.path.join(ROOT, "include", "prompts.h")
MODEL = os.path.join(ROOT, "data", "tok8192.model")


def main():
    if not os.path.exists(HEADER):
        print(f"missing {HEADER} -- run scripts/make_prompts.py first", file=sys.stderr)
        return 1

    text = io.open(HEADER, encoding="utf-8").read()

    # token arrays:  static const int32_t PROMPT_<i>[] = { ... , 0 };
    tok_arrays = {}
    for m in re.finditer(r"static const int32_t PROMPT_(\d+)\[\]\s*=\s*\{([^}]*)\}", text):
        idx = int(m.group(1))
        ids = [int(x) for x in m.group(2).replace("\n", " ").split(",") if x.strip()]
        tok_arrays[idx] = ids

    # text literals: static const char PROMPT_TEXT_<i>[] = u8"...";  (the u8
    # prefix is optional -- gen_prompts.py emits a plain narrow literal, since
    # GCC's default exec charset is UTF-8 and under C++20 u8"" would become
    # char8_t[] and no longer convert to const char[])
    texts = {}
    for m in re.finditer(r'static const char PROMPT_TEXT_(\d+)\[\]\s*=\s*(?:u8)?"(.*?)";', text):
        body = m.group(2)
        # undo the C++ escaping gen_prompts.py applies
        body = body.replace('\\"', '"').replace("\\\\", "\\")
        texts[int(m.group(1))] = body

    if not tok_arrays or set(tok_arrays) != set(texts):
        print(f"index mismatch: tokens={sorted(tok_arrays)} texts={sorted(texts)}",
              file=sys.stderr)
        return 1

    # sentencepiece cannot handle a non-ASCII path, so stage the model in temp
    tmp = os.path.join(tempfile.gettempdir(), "espllm_verify")
    os.makedirs(tmp, exist_ok=True)
    tmp_model = os.path.join(tmp, "tok8192.model")
    shutil.copy2(MODEL, tmp_model)
    sp = spm.SentencePieceProcessor(model_file=tmp_model)

    ok = True
    for i in sorted(tok_arrays):
        ids = tok_arrays[i]
        s = texts[i]

        # the C header text is a raw UTF-8 literal: leftover spaces are real spaces
        expected = [sp.bos_id()] + sp.encode(s)
        # arrays are terminated by a 0 sentinel
        stored = ids[:-1] if ids and ids[-1] == 0 else ids

        match = expected == stored
        ok &= match
        print(f"[{i}] {'OK  ' if match else 'FAIL'}  {len(s):3d} chars  "
              f"{len(stored):3d} tokens  {s[:22]}...")
        # sanity: no stray whitespace should have crept into the literal
        if any(w in s for w in ("  ", " ,", " 。")):
            print(f"     note: suspicious whitespace in the literal: {s!r}")
        if not match:
            print(f"     stored   : {stored}")
            print(f"     re-encode: {expected}")

    shutil.rmtree(tmp, ignore_errors=True)
    print("\nall prompts consistent" if ok else "\nMISMATCH FOUND")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
