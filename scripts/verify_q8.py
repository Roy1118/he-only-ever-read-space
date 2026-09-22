#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证 llama2.c v2 (Q8_0) 模型文件格式，与 runq.c 的 memory_map_weights 完全一致的解析。
用法: python verify_q8.py --bin <model.bin> [--ckpt <ckpt.pt>] [--tokens <tok8192.bin>]

输出:
  - 头部字段解析
  - 张量布局游走，检验消费字节数 == 文件大小
  - 若给 --ckpt: 反量化 token embedding 与 fp32 对拍 (max error)
  - 这是后续 ESP32 C 解析器的参考实现
"""
import argparse
import json
import os
import struct
import sys

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True, help="v2 Q8 model file")
    ap.add_argument("--ckpt", default=None, help="optional torch ckpt.pt")
    args = ap.parse_args()

    with open(args.bin, "rb") as f:
        data = f.read()
    N = len(data)

    # ---------- header (256 bytes), same as runq.c ----------
    (magic,) = struct.unpack_from("I", data, 0)
    assert magic == 0x616B3432, f"bad magic {magic:#x}"
    (version,) = struct.unpack_from("i", data, 4)
    assert version == 2, f"bad version {version}"
    dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len = \
        struct.unpack_from("iiiiiii", data, 8)
    (shared,) = struct.unpack_from("B", data, 36)
    (gs,) = struct.unpack_from("i", data, 37)
    print(f"magic ok | version {version} | dim {dim} | hidden {hidden_dim} | "
          f"layers {n_layers} | heads {n_heads} | kv {n_kv_heads} | "
          f"vocab {vocab_size} | seq {seq_len} | shared {shared} | GS {gs}")
    assert gs > 0 and dim % gs == 0

    off = 256
    consumed = off

    def take_f32(n):
        nonlocal consumed
        b = n * 4
        consumed += b
        return np.frombuffer(data, dtype="<f4", count=n, offset=off + (consumed - b)).copy()

    def take_q8(size_each):
        """Q8_0 tensor: int8 q (size_each) then float s (size_each/gs)"""
        nonlocal consumed
        n_group = size_each // gs
        q = np.frombuffer(data, dtype="<i1", count=size_each, offset=consumed).copy()
        consumed += size_each
        s = np.frombuffer(data, dtype="<f4", count=n_group, offset=consumed).copy()
        consumed += n_group * 4
        return q, s

    # ---------- fp32 norms (same order as runq.c memory_map) ----------
    n_norm = n_layers * dim
    rms_att = take_f32(n_layers * dim)
    ffn_att = take_f32(n_layers * dim)
    rms_final = take_f32(dim)

    # ---------- Q8 tensors ----------
    head = dim // n_heads
    kv_dim = (dim * n_kv_heads) // n_heads
    layout = [
        ("q_tokens", vocab_size * dim),
        ("wq", n_layers * dim * (n_heads * head)),
        ("wk", n_layers * dim * kv_dim),
        ("wv", n_layers * dim * kv_dim),
        ("wo", n_layers * (n_heads * head) * dim),
        ("w1", n_layers * dim * hidden_dim),
        ("w2", n_layers * hidden_dim * dim),
        ("w3", n_layers * dim * hidden_dim),
    ]
    if not shared:
        layout.append(("wcls", dim * vocab_size))

    q_tokens = None
    for name, size_each in layout:
        q, s = take_q8(size_each)
        print(f"  {name:10s} int8 {size_each:>9,}  scale {size_each//gs:>8,}  ({size_each + (size_each//gs)*4:>10,} bytes)")
        if name == "q_tokens":
            q_tokens = (q, s)

    print(f"\nconsumed bytes: {consumed:,}  file size: {N:,}")
    assert consumed == N, f"LAYOUT MISMATCH: consumed {consumed} != file {N}"
    print("✅ 张量布局与文件大小完全吻合")

    # ---------- optional: compare dequantized embedding vs fp32 ckpt ----------
    if args.ckpt and q_tokens is not None:
        import torch
        ckpt = torch.load(args.ckpt, map_location="cpu")
        model_args = ckpt["model_args"]
        print(f"\nckpt model_args: {json.dumps({k: model_args[k] for k in ['dim','n_layers','n_heads','n_kv_heads','vocab_size','max_seq_len']}, default=str)}")
        sd = ckpt["model"]
        for k in list(sd.keys()):
            if k.startswith("_orig_mod."):
                sd[k[10:]] = sd.pop(k)
        embed = sd["tok_embeddings.weight"].float().numpy()  # (vocab, dim)

        q, s = q_tokens
        deq = (q.reshape(-1, gs).astype(np.float32) * s[:, None]).reshape(vocab_size, dim)
        # embedding is row-major (vocab, dim) -> each row is a group boundary at vocab*dim
        # group index for element (i,j) = (i*dim + j)//gs
        err = np.abs(deq - embed).max()
        rel = err / (np.abs(embed).max() + 1e-9)
        print(f"token embedding 反量化 vs fp32: max abs err {err:.6f}  rel {rel:.6%}")
        # 抽查几个位置
        for (i, j) in [(0, 0), (1, 5), (100, 287), (4000, 0), (8191, 287)]:
            print(f"  ({i},{j}): fp32={embed[i,j]:+.4f}  deq={deq[i,j]:+.4f}")

    print("\n验证完成: 格式正确 ✅")


if __name__ == "__main__":
    main()
