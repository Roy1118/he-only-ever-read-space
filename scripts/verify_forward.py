#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Recompute reference logits from model_esp.bin (the ESP32 interleaved layout).

This mirrors runq.c's forward pass term for term -- rmsnorm, per-group Q8
quantization, the integer-accumulating matmul, adjacent-pair RoPE, GQA attention,
SwiGLU, tied classifier -- but reads the LAYER-MAJOR layout that
tools/convert_esp.py writes.

The firmware built with -DESPLLM_VERIFY prints the same top-8 blocks, so diffing
the two pinpoints a layout or math bug on the device. This matters because a
byte-count layout assertion cannot detect a wrong tensor order: every ordering
consumes exactly the same number of bytes.

Usage:
    python scripts/verify_forward.py llama2.c/model_esp.bin
"""
import math
import struct
import sys

import numpy as np

# must match runVerify() in src/main.cpp
SEQ = [1, 464, 325, 4375]
TOP_K = 8

MAGIC = 0x4C4C4D45  # "LLME"


def tensor_bytes(n, gs):
    return (n // gs) * (gs + 4)


class Model:
    def __init__(self, path):
        with open(path, "rb") as f:
            self.buf = f.read()
        b = self.buf

        magic, version = struct.unpack_from("<II", b, 0)
        assert magic == MAGIC, f"bad magic {magic:#x}, expected 'LLME'"
        assert version == 1, f"unsupported version {version}"

        (self.dim, self.hidden, self.n_layers, self.n_heads, self.n_kv,
         self.vocab, self.seq_len, self.gs) = struct.unpack_from("<8i", b, 8)
        (shared,) = struct.unpack_from("<i", b, 40)
        assert shared == 1, "expected tied embeddings"

        self.head_size = self.dim // self.n_heads
        self.kv_dim = self.dim * self.n_kv // self.n_heads
        self.kv_mul = self.n_heads // self.n_kv

        off = 64

        def take_f32(n):
            nonlocal off
            v = np.frombuffer(b, dtype="<f4", count=n, offset=off).copy()
            off += n * 4
            return v

        def take_q8(n):
            nonlocal off
            ng = n // self.gs
            a = np.frombuffer(b, dtype=np.uint8, count=ng * (self.gs + 4), offset=off)
            a = a.reshape(ng, self.gs + 4)
            q = a[:, :self.gs].copy().view(np.int8).reshape(-1)
            s = a[:, self.gs:self.gs + 4].copy().view("<f4").reshape(-1)
            off += ng * (self.gs + 4)
            return q, s

        self.rms_att = take_f32(self.n_layers * self.dim).reshape(self.n_layers, self.dim)
        self.rms_ffn = take_f32(self.n_layers * self.dim).reshape(self.n_layers, self.dim)
        self.rms_final = take_f32(self.dim)

        # LAYER-MAJOR: all seven tensors of layer 0, then all seven of layer 1.
        dim, hidden, gs = self.dim, self.hidden, self.gs
        kv_dim = self.kv_dim
        sizes = [
            ("wq", tensor_bytes(dim * dim, gs), dim * dim),
            ("wk", tensor_bytes(dim * kv_dim, gs), dim * kv_dim),
            ("wv", tensor_bytes(dim * kv_dim, gs), dim * kv_dim),
            ("wo", tensor_bytes(dim * dim, gs), dim * dim),
            ("w1", tensor_bytes(dim * hidden, gs), dim * hidden),
            ("w2", tensor_bytes(hidden * dim, gs), hidden * dim),
            ("w3", tensor_bytes(dim * hidden, gs), dim * hidden),
        ]
        self.W = {name: [] for name, _, _ in sizes}
        self.S = {name: [] for name, _, _ in sizes}
        for _ in range(self.n_layers):
            for name, _sz, n in sizes:
                q, s = take_q8(n)
                self.W[name].append(q)
                self.S[name].append(s)

        self.lm_q, self.lm_s = take_q8(self.vocab * dim)

        assert off == len(b), f"consumed {off} of {len(b)} bytes"

        # dequantized embedding rows (only used for the lookup)
        eq = self.lm_q.reshape(self.vocab, dim).astype(np.float32)
        es = self.lm_s.reshape(self.vocab, dim // gs)
        self.embed = eq * np.repeat(es, gs, axis=1)

        self.key_cache = np.zeros((self.n_layers, self.seq_len, kv_dim), dtype=np.float32)
        self.val_cache = np.zeros((self.n_layers, self.seq_len, kv_dim), dtype=np.float32)


def rmsnorm(x, w):
    ss = float((x.astype(np.float32) ** 2).sum()) / len(x) + 1e-5
    inv = np.float32(1.0 / math.sqrt(ss))
    return (w * (x * inv)).astype(np.float32)


def quantize(x, gs):
    """Bit-faithful port of the firmware's quantize(): per-group max/127 scale,
    round-half-away via rint, clamp to +-127."""
    ng = len(x) // gs
    xr = x.reshape(ng, gs).astype(np.float32)
    wmax = np.abs(xr).max(axis=1)
    sc = (wmax / np.float32(127.0)).astype(np.float32)
    pos = sc > 0
    inv = np.zeros(ng, dtype=np.float32)
    inv[pos] = np.float32(1.0) / sc[pos]
    q = np.clip(np.rint(xr * inv[:, None]), -127, 127).astype(np.int8)
    return q.reshape(-1), sc


def matmul(xq, xs, wq, ws, n, d, gs):
    """W (d,n) @ x (n,). Integer accumulation inside each group, then the float
    scale multiply -- exactly the order runq.c uses."""
    ng = n // gs
    xr = xq.reshape(ng, gs).astype(np.int32)
    wr = wq.reshape(d, ng, gs).astype(np.int32)
    ival = np.einsum("gk,dgk->dg", xr, wr).astype(np.float32)
    out = (ival * ws.reshape(d, ng) * xs[None, :]).sum(axis=1)
    return out.astype(np.float32)


def softmax(x):
    m = x.max()
    e = np.exp((x - m).astype(np.float32))
    return (e / e.sum()).astype(np.float32)


def forward(m, token, pos):
    dim, gs = m.dim, m.gs
    hs, kv_dim, kv_mul = m.head_size, m.kv_dim, m.kv_mul

    x = m.embed[token].copy()

    for l in range(m.n_layers):
        xb = rmsnorm(x, m.rms_att[l])
        xq, xs = quantize(xb, gs)

        q = matmul(xq, xs, m.W["wq"][l], m.S["wq"][l], dim, dim, gs)
        k = matmul(xq, xs, m.W["wk"][l], m.S["wk"][l], dim, kv_dim, gs)
        v = matmul(xq, xs, m.W["wv"][l], m.S["wv"][l], dim, kv_dim, gs)

        # adjacent-pair RoPE, same as train.py's apply_rotary_emb / runq.c
        for i in range(0, dim, 2):
            hd = i % hs
            freq = np.float32(1.0) / np.float32(10000.0 ** (float(hd) / float(hs)))
            ang = np.float32(pos) * freq
            fcr = np.float32(math.cos(float(ang)))
            fci = np.float32(math.sin(float(ang)))
            vecs = (q, k) if i < kv_dim else (q,)
            for vec in vecs:
                v0 = np.float32(vec[i])
                v1 = np.float32(vec[i + 1])
                vec[i] = np.float32(v0 * fcr - v1 * fci)
                vec[i + 1] = np.float32(v0 * fci + v1 * fcr)

        m.key_cache[l, pos] = k
        m.val_cache[l, pos] = v

        for h in range(m.n_heads):
            qh = q[h * hs:(h + 1) * hs]
            off = (h // kv_mul) * hs
            scores = np.empty(pos + 1, dtype=np.float32)
            for t in range(pos + 1):
                kh = m.key_cache[l, t, off:off + hs]
                scores[t] = np.float32(np.dot(qh, kh) / math.sqrt(hs))
            a = softmax(scores)

            acc = np.zeros(hs, dtype=np.float32)
            for t in range(pos + 1):
                acc += np.float32(a[t]) * m.val_cache[l, t, off:off + hs]
            xb[h * hs:(h + 1) * hs] = acc

        xq, xs = quantize(xb, gs)
        x = x + matmul(xq, xs, m.W["wo"][l], m.S["wo"][l], dim, dim, gs)

        xb = rmsnorm(x, m.rms_ffn[l])
        xq, xs = quantize(xb, gs)
        hb = matmul(xq, xs, m.W["w1"][l], m.S["w1"][l], dim, m.hidden, gs)
        hb2 = matmul(xq, xs, m.W["w3"][l], m.S["w3"][l], dim, m.hidden, gs)
        hb = (hb * (np.float32(1.0) / (np.float32(1.0) + np.exp(-hb))) * hb2).astype(np.float32)

        hq, hq_s = quantize(hb, gs)
        x = x + matmul(hq, hq_s, m.W["w2"][l], m.S["w2"][l], m.hidden, dim, gs)

    x = rmsnorm(x, m.rms_final)
    xq, xs = quantize(x, gs)
    return matmul(xq, xs, m.lm_q, m.lm_s, dim, m.vocab, gs)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2

    m = Model(sys.argv[1])
    print(f"VERIFY-BEGIN gs={m.gs} dim={m.dim} layers={m.n_layers}")

    for pos, token in enumerate(SEQ):
        logits = forward(m, token, pos)
        print(f"POS {pos} TOKEN {token}")
        order = np.argsort(-logits)[:TOP_K]
        for i in order:
            print(f"  {int(i):5d} {float(logits[i]):12.6f}")

    print("VERIFY-END")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
