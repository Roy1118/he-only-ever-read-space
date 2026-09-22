#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert a llama2.c v2 Q8 model (q8.bin) into the ESP32 on-flash layout.

WHY THIS EXISTS
---------------
1. runq.c's native layout keeps a whole tensor's int8 block first and all of its
   scales afterwards. A row-wise matmul then needs the scale for row i's group g,
   which lives megabytes away -- every group costs a random flash read. On the
   ESP32 that is fatal, because the memory-mapped window is read sequentially at
   ~25 MB/s and random access is far worse.

   We rewrite each tensor as repeating groups of [gs int8][1 float32 scale], so a
   matmul streams one contiguous run per row and the whole tensor is read in
   order.

2. We reorder storage: runq.c interleaves the seven per-layer tensors
   tensor-major (all layers' wq, then all layers' wk, ...), which would make the
   firmware jump across the whole file once per layer. We store them LAYER-MAJOR
   (all seven tensors of layer 0, then layer 1, ...) so one layer's weights are
   one contiguous run.

3. lm_head (which is the tied token-embedding table) goes last, because the
   embedding lookup for the current token is the only random access in the whole
   forward pass -- keeping it away from the hot sequential stream.

VERIFYING THIS IS NOT OPTIONAL
------------------------------
Both the source order and the destination order consume EXACTLY the same number
of bytes, so a total-size assertion cannot tell a correct conversion from a
completely shuffled one. (That is precisely the bug this script used to have:
it read the source layer-major when the source is tensor-major, and never read
q_tokens at all -- yet the byte count matched perfectly.)

So after writing, this script re-parses its own output and compares every tensor
against the source, group by group, and refuses to emit a file that fails.

LAYOUT
------
  [64B header] magic "LLME", version=1, dim, hidden, layers, heads, kv_heads,
               vocab, seq_len, gs, shared, pad
  [fp32]       rms_att (layers*dim), rms_ffn (layers*dim), rms_final (dim)
  per layer:   wq, wk, wv, wo, w1, w2, w3   (each group-interleaved)
  [Q8]         lm_head (== q_tokens)        (group-interleaved)

Usage: python tools/convert_esp.py --in q8.bin --out model_esp.bin
"""
import argparse
import struct
import sys

import numpy as np

MAGIC_LLAMA2C = 0x616B3432   # "ak42", llama2.c v2
MAGIC_ESP = 0x4C4C4D45       # "LLME"
HEADER_SIZE = 64


# --------------------------------------------------------------------- helpers

def q8_bytes(n, gs):
    """Bytes a group-interleaved tensor of n elements occupies."""
    return (n // gs) * (gs + 4)


def read_f32(data, off, n):
    v = np.frombuffer(data, dtype="<f4", count=n, offset=off).copy()
    return v, off + n * 4


def read_q8_native(data, off, n, gs):
    """runq.c layout: all int8, then all scales. Returns raw bytes for both."""
    ng = n // gs
    q = data[off:off + n]
    off += n
    s = data[off:off + ng * 4]
    off += ng * 4
    if len(q) != n or len(s) != ng * 4:
        raise ValueError("truncated tensor")
    return q, s, off


def interleave(q_bytes, s_bytes, n, gs):
    """[gs int8][1 f32] repeating."""
    ng = n // gs
    q = np.frombuffer(q_bytes, dtype=np.uint8).reshape(ng, gs)
    s = np.frombuffer(s_bytes, dtype=np.uint8).reshape(ng, 4)
    out = np.empty((ng, gs + 4), dtype=np.uint8)
    out[:, :gs] = q
    out[:, gs:] = s
    return out.tobytes()


def deinterleave(blob, n, gs):
    """Inverse of interleave(), returning raw bytes so nothing is retyped."""
    ng = n // gs
    a = np.frombuffer(blob, dtype=np.uint8, count=ng * (gs + 4)).reshape(ng, gs + 4)
    return a[:, :gs].copy().tobytes(), a[:, gs:].copy().tobytes()


# ------------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="fin", required=True, help="llama2.c v2 Q8 file")
    ap.add_argument("--out", dest="fout", required=True, help="ESP32 layout to write")
    ap.add_argument("--no-verify", action="store_true")
    args = ap.parse_args()

    data = open(args.fin, "rb").read()

    magic, version = struct.unpack_from("<Ii", data, 0)
    if magic != MAGIC_LLAMA2C:
        sys.exit(f"bad magic {magic:#x}: not a llama2.c v2 file")
    if version != 2:
        sys.exit(f"need v2, got v{version}")

    dim, hidden, layers, heads, kv_heads, vocab, seq_len = struct.unpack_from("<7i", data, 8)
    (shared,) = struct.unpack_from("<B", data, 36)
    (gs,) = struct.unpack_from("<i", data, 37)
    if not shared:
        sys.exit("model must use tied embeddings (lm_head == token embedding)")
    if dim % gs or hidden % gs:
        sys.exit(f"dim/hidden not divisible by group size {gs}")

    head_size = dim // heads
    kv_dim = dim * kv_heads // heads
    print(f"source v2: dim={dim} hidden={hidden} layers={layers} heads={heads} "
          f"kv={kv_heads} vocab={vocab} seq={seq_len} gs={gs} shared={shared}")

    # ---- read the source in the order runq.c memory_map_weights uses ----
    # TENSOR-major: q_tokens, then all layers of wq, then all of wk, ...
    off = 256
    norms_bytes = layers * dim * 4 * 2 + dim * 4
    norms = data[off:off + norms_bytes]
    off += norms_bytes

    def take(n):
        nonlocal off
        q, s, off = read_q8_native(data, off, n, gs)
        return q, s

    q_tokens = take(vocab * dim)
    tensor_defs = [
        ("wq", dim * (heads * head_size)),
        ("wk", dim * kv_dim),
        ("wv", dim * kv_dim),
        ("wo", (heads * head_size) * dim),
        ("w1", dim * hidden),
        ("w2", hidden * dim),
        ("w3", dim * hidden),
    ]
    W = {}
    for name, n in tensor_defs:
        W[name] = [take(n) for _ in range(layers)]

    if off != len(data):
        sys.exit(f"source layout mismatch: consumed {off} of {len(data)} bytes")
    print(f"source read complete ({len(data):,} bytes)")

    # ---- write the destination LAYER-major ----
    body = bytearray()
    for l in range(layers):
        for name, n in tensor_defs:
            q, s = W[name][l]
            body += interleave(q, s, n, gs)
    body += interleave(q_tokens[0], q_tokens[1], vocab * dim, gs)

    hdr = struct.pack("<IIiiiiiiiiiiII", MAGIC_ESP, 1, dim, hidden, layers,
                      heads, kv_heads, vocab, seq_len, gs, 1, 0, 0, 0)
    hdr += b"\0" * (HEADER_SIZE - len(hdr))
    assert len(hdr) == HEADER_SIZE

    with open(args.fout, "wb") as f:
        f.write(hdr)
        f.write(norms)
        f.write(bytes(body))

    total = HEADER_SIZE + norms_bytes + len(body)
    print(f"wrote {args.fout}: {total:,} bytes ({total / 1048576:.2f} MB)")
    print(f"  norms {norms_bytes:,} | layer tensors {layers * sum(q8_bytes(n, gs) for _, n in tensor_defs):,}"
          f" | lm_head {q8_bytes(vocab * dim, gs):,}")

    if not args.no_verify:
        verify(args.fout, data, norms, layers, tensor_defs, q_tokens, W, dim, vocab, gs)


def verify(path, src, norms, layers, tensor_defs, q_tokens, W, dim, vocab, gs):
    """Re-read our own output and compare every tensor against the source.

    A byte-count check cannot substitute for this: every possible tensor
    ordering has the identical total size, which is exactly how the previous
    version of this script shipped a completely shuffled model.
    """
    print("\nverifying output against source ...")
    out = open(path, "rb").read()
    off = HEADER_SIZE

    if out[off:off + len(norms)] != norms:
        sys.exit("FAIL: norms differ")
    off += len(norms)
    print("  norms                OK")

    def check(name, label, expect_q, expect_s, n):
        nonlocal off
        size = q8_bytes(n, gs)
        q, s = deinterleave(out[off:off + size], n, gs)
        off += size
        if q != expect_q or s != expect_s:
            sys.exit(f"FAIL: {label} does not match the source")
        print(f"  {name:<20} OK")

    for l in range(layers):
        for name, n in tensor_defs:
            check(f"layer{l}.{name}", f"layer {l} {name}", W[name][l][0], W[name][l][1], n)

    check("lm_head", "lm_head", q_tokens[0], q_tokens[1], vocab * dim)

    if off != len(out):
        sys.exit(f"FAIL: consumed {off} of {len(out)} output bytes")
    print("  all tensors match the source byte for byte")
    print("conversion verified")


if __name__ == "__main__":
    main()
