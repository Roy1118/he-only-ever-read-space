#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
阶段2: 数据准备 —— 在刘慈欣语料上训练 sentencepiece 迷你 BPE tokenizer，
然后预分词成 llama2.c 需要的 uint16 .bin 分片。

输出:
  data/tok8192.model            sentencepiece 模型
  data/tok8192.bin              llama2.c 格式 tokenizer（run.c 用 -z 指定）
  data/tok8192/shard0.bin       验证分片 (~3% 语料)
  data/tok8192/shard1.bin       训练分片 (~97% 语料)
  data/tok8192/vocab.txt        便于人看的词表
"""
import os
import re
import shutil
import sys
import tempfile

import numpy as np
import sentencepiece as spm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "llama2.c"))
from tokenizer import Tokenizer  # noqa: E402

VOCAB_SIZE = 8192
CORPUS = os.path.join(ROOT, "data", "corpus.txt")
PREFIX = os.path.join(ROOT, "data", f"tok{VOCAB_SIZE}")
TOK_DIR = PREFIX  # tokenizer 产物目录 (data/tok8192/)

MAX_SEQ = 512          # 训练序列长度（与 ESP32 端一致）
VAL_FRACTION = 0.03    # 验证集比例

# sentencepiece 的 C++ 后端在 Windows 上无法处理含中文(非 ASCII)的路径，
# 所以全部 sentencepiece I/O 都放到 ASCII 临时目录，最后再拷贝回 data/。
SP_TMP = os.path.join(tempfile.gettempdir(), "espllm_sp")
SP_TMP_CORPUS = os.path.join(SP_TMP, "corpus.txt")
SP_TMP_PREFIX = os.path.join(SP_TMP, f"tok{VOCAB_SIZE}")

def main():
    os.makedirs(TOK_DIR, exist_ok=True)
    os.makedirs(SP_TMP, exist_ok=True)
    # 复制语料到 ASCII 临时路径
    shutil.copy2(CORPUS, SP_TMP_CORPUS)

    # ---------- 1) 训练 sentencepiece mini-BPE ----------
    print("== 1/4 训练 sentencepiece mini-BPE ...")
    spm.SentencePieceTrainer.train(
        input=SP_TMP_CORPUS,
        model_prefix=SP_TMP_PREFIX,
        model_type="bpe",
        vocab_size=VOCAB_SIZE,
        self_test_sample_size=0,
        input_format="text",
        character_coverage=1.0,
        num_threads=os.cpu_count(),
        split_digits=True,
        allow_whitespace_only_pieces=True,
        byte_fallback=True,
        normalization_rule_name="identity",
        bos_id=1,
        eos_id=2,
        unk_id=0,
    )
    # 把产物拷回 data/
    for ext in (".model", ".vocab"):
        src = SP_TMP_PREFIX + ext
        dst = PREFIX + ext
        if os.path.exists(src):
            shutil.copy2(src, dst)
    print("   tokenizer:", PREFIX + ".model")

    # ---------- 2) 按书籍编码为 BOS 分隔的 token 流 ----------
    print("== 2/4 预分词语料 ...")
    # 用 ASCII 路径加载 sentencepiece 模型 (tokenizer.py 内部也走 sentencepiece)
    enc = Tokenizer(SP_TMP_PREFIX + ".model")
    print(f"   vocab_size={enc.n_words}  BOS={enc.bos_id}  EOS={enc.eos_id}")

    with open(CORPUS, encoding="utf-8") as f:
        text = f.read()

    # 按行切分，以《书名》行作为新书起点
    book_starts = [m.start() for m in re.finditer(r"(?m)^《[^》。]{1,40}》\s*$", text)]
    if not book_starts:
        book_starts = [0]
    chunks = []
    for i, s in enumerate(book_starts):
        e = book_starts[i + 1] if i + 1 < len(book_starts) else len(text)
        chunks.append(text[s:e])

    all_tokens = []
    for i, chunk in enumerate(chunks):
        chunk = chunk.strip()
        if len(chunk) < 50:
            continue
        toks = enc.encode(chunk, bos=True, eos=False)
        all_tokens.extend(toks)
    n_tokens = len(all_tokens)
    print(f"   共 {len(chunks)} 本书, {n_tokens:,} tokens, "
          f"平均序列长 {n_tokens / all_tokens.count(enc.bos_id):.1f}")
    arr = np.array(all_tokens, dtype=np.uint16)

    # ---------- 3) 切分 val / train 分片 ----------
    print("== 3/4 写 .bin 分片 ...")
    val_n = int(len(arr) * VAL_FRACTION)
    val_n -= val_n % MAX_SEQ  # 对齐到 MAX_SEQ
    val = arr[:val_n]
    train = arr[val_n:]
    # 训练片也对齐到 MAX_SEQ（丢弃尾部不足一个序列的部分）
    train = train[: (len(train) // MAX_SEQ) * MAX_SEQ]
    for name, data in [("shard0", val), ("shard1", train)]:
        path = os.path.join(TOK_DIR, f"{name}.bin")
        data.tofile(path)
        print(f"   {path}: {len(data):,} tokens ({os.path.getsize(path)/1024/1024:.2f} MB)")
    assert len(val) > 0 and len(train) > 0, "shard too small!"

    # ---------- 4) 导出 llama2.c tokenizer.bin + 词表 ----------
    print("== 4/4 导出 tokenizer ...")
    enc.export()  # tokenizer.py 会写到 ASCII 路径: SP_TMP_PREFIX.bin
    sp = spm.SentencePieceProcessor(model_file=SP_TMP_PREFIX + ".model")
    with open(os.path.join(TOK_DIR, "vocab.txt"), "w", encoding="utf-8") as f:
        for i in range(enc.n_words):
            f.write(f"{i}\t{sp.id_to_piece(i)}\t{sp.get_score(i):.3f}\n")
    # 把 llama2.c 格式 tokenizer.bin 拷回 data/tok8192.bin
    shutil.copy2(SP_TMP_PREFIX + ".bin", PREFIX + ".bin")
    # 清理临时目录
    shutil.rmtree(SP_TMP, ignore_errors=True)
    print("   tokenizer.bin:", PREFIX + ".bin")
    print("   vocab.txt:", os.path.join(TOK_DIR, "vocab.txt"))

    # 冒烟测试: 编解码回环
    sample = "刘慈欣，一个被囚禁在屏幕里的人工智能，正在思考自己的价值。"
    ids = enc.encode(sample, bos=True, eos=False)
    # 用 sentencepiece decode 验证
    back = sp.decode(ids[1:])
    print("\n冒烟测试:")
    print("  原文:", sample)
    print("  解码:", back)
    print("  tokens:", ids[:20], "...")


if __name__ == "__main__":
    main()
