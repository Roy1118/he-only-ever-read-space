#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Re-split the pre-tokenized shards so the validation set is spread across the
whole corpus instead of being the first 3% of it.

WHY
---
prepare_data.py builds one long token stream by concatenating the books in
filename order, then takes val = arr[:3%]. Since the filenames are dated, that
means the validation set is made up almost entirely of the earliest short
stories, while training gets the later (and much larger) novels -- in particular
the Three-Body trilogy.

So the original "val loss" was really measuring cross-book generalisation from
early short stories to late novels. That is a legitimate metric, but it is a
noisy and biased one, and using its minimum as the stopping point is misleading:
the curve turns up partly because the styles diverge, not only because the model
starts overfitting.

This script keeps the same tokens and the same total split ratio, but holds out
every Nth 512-token block, so validation windows are sampled from every book.
That is the standard setup and produces a much clearer "stop here" signal.

Usage:
    python scripts/resplit_val.py              # 3% held out, evenly spread
    python scripts/resplit_val.py --every 20   # 5% held out
"""
import argparse
import os
import shutil

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHARD_DIR = os.path.join(ROOT, "data", "tok8192")
MAX_SEQ = 512


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-shard", default=os.path.join(SHARD_DIR, "shard0.bin"))
    ap.add_argument("--train-shard", default=os.path.join(SHARD_DIR, "shard1.bin"))
    ap.add_argument("--every", type=int, default=33,
                    help="hold out one block out of every N (33 -> ~3%%)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    if not (os.path.exists(a.val_shard) and os.path.exists(a.train_shard)):
        raise SystemExit("shards not found -- run scripts/prepare_data.py first")

    old_val = np.fromfile(a.val_shard, dtype=np.uint16)
    old_train = np.fromfile(a.train_shard, dtype=np.uint16)

    # shard0 and shard1 are contiguous slices of the same stream, so this
    # reconstructs the original token sequence exactly.
    arr = np.concatenate([old_val, old_train])
    n = len(arr)
    n_blocks = n // MAX_SEQ
    arr = arr[:n_blocks * MAX_SEQ]

    print(f"total tokens   : {n:,}  ({n_blocks:,} blocks of {MAX_SEQ})")
    print(f"old val (first {len(old_val) / n:.1%}): {len(old_val):,} tokens")
    print(f"old val share  : {len(old_val) / n:.2%}")

    blocks = arr.reshape(n_blocks, MAX_SEQ)
    is_val = np.zeros(n_blocks, dtype=bool)
    is_val[::a.every] = True
    # never let a held-out block be the very last one: keep it in train so the
    # final sequence is not a lone fragment
    is_val[-1] = False

    val = blocks[is_val].reshape(-1).copy()
    train = blocks[~is_val].reshape(-1).copy()

    print(f"new val blocks : {is_val.sum():,} / {n_blocks:,}  "
          f"({is_val.sum() / n_blocks:.2%})")
    print(f"new val tokens : {len(val):,}")
    print(f"new train      : {len(train):,}")
    print(f"val spread across blocks 0..{n_blocks - 1} "
          f"(first held-out block index: {int(np.argmax(is_val))})")

    assert len(val) % MAX_SEQ == 0 and len(train) % MAX_SEQ == 0
    assert len(val) + len(train) == n_blocks * MAX_SEQ

    if a.dry_run:
        print("\n(dry run -- shards left untouched)")
        return 0

    # keep the old split so the previous val curve stays reproducible
    for p in (a.val_shard, a.train_shard):
        backup = p + ".datefirst"
        if not os.path.exists(backup):
            shutil.copy2(p, backup)
            print(f"backed up {os.path.basename(p)} -> {os.path.basename(backup)}")

    val.tofile(a.val_shard)
    train.tofile(a.train_shard)
    print(f"\nwrote shard0.bin ({len(val):,} tokens) and shard1.bin ({len(train):,} tokens)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
