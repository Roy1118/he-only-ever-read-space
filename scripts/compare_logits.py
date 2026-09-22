#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare the firmware's -DESPLLM_VERIFY logits against the numpy reference.

Indices must match exactly (a different argmax means a real math bug). Values are
allowed to differ by a small epsilon: the device computes in float32 while the
reference uses numpy float32 too, but they take different code paths (per-group
integer accumulation order, libm cosf/sinf vs numpy), so exact bit equality is
not expected.

Usage:
    python scripts/compare_logits.py ref_logits.txt dev_logits.txt
"""
import re
import sys

LINE = re.compile(r"^\s*(\d+)\s+(-?\d+(?:\.\d+)?(?:e[+-]?\d+)?)\s*$", re.I)
POS = re.compile(r"^POS\s+(\d+)\s+TOKEN\s+(\d+)")
# cosf/sinf and the order of the group accumulation differ, so allow ~1e-4
TOLERANCE = 5e-4


def parse(path):
    blocks = {}
    cur = None
    for raw in open(path, encoding="utf-8", errors="replace"):
        line = raw.rstrip("\n")
        m = POS.match(line.strip())
        if m:
            cur = int(m.group(1))
            blocks[cur] = {"token": int(m.group(2)), "top": []}
            continue
        if cur is None:
            continue
        m = LINE.match(line)
        if m:
            blocks[cur]["top"].append((int(m.group(1)), float(m.group(2))))
    return blocks


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    ref = parse(sys.argv[1])
    dev = parse(sys.argv[2])

    if set(ref) != set(dev):
        print(f"FAIL: positions differ: ref={sorted(ref)} dev={sorted(dev)}")
        return 1

    ok = True
    for pos in sorted(ref):
        r, d = ref[pos], dev[pos]
        if r["token"] != d["token"]:
            print(f"POS {pos}: token differs: ref={r['token']} dev={d['token']}")
            ok = False
            continue
        if len(r["top"]) != len(d["top"]):
            print(f"POS {pos}: entry count {len(r['top'])} vs {len(d['top'])}")
            ok = False
            continue

        ri = [i for i, _ in r["top"]]
        di = [i for i, _ in d["top"]]
        maxd = max(abs(a - b) for (_, a), (_, b) in zip(r["top"], d["top"]))
        same = ri == di
        ok &= same
        print(f"POS {pos} token {r['token']}: indices {'MATCH' if same else 'DIFFER'}, "
              f"max |delta| = {maxd:.2e}")
        if not same:
            print(f"   ref: {ri}")
            print(f"   dev: {di}")
        elif maxd > TOLERANCE:
            print(f"   !! max delta exceeds tolerance {TOLERANCE:g}")

    print("\nFORWARD PASS VERIFIED: device matches the reference" if ok
          else "\nMISMATCH: device and reference disagree")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
