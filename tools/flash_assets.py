#!/usr/bin/env python3
"""Flash the model + tokenizer blobs into their raw partitions.

The firmware reads them via esp_partition_mmap()/esp_partition_read(), not via a
filesystem, so we write the raw bytes straight to the partition offsets declared
in partitions.csv. Offsets are parsed from that file rather than hard-coded, so
they can never drift out of sync.

Usage:
    python tools/flash_assets.py                       # smoke model + tok8192
    python tools/flash_assets.py --model llama2.c/model_esp.bin
    python tools/flash_assets.py --dry-run             # just show what would go where
"""
import argparse
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PARTS_CSV = os.path.join(ROOT, "partitions.csv")

DEFAULT_MODEL = os.path.join(ROOT, "llama2.c", "model_esp.bin")
FALLBACK_MODEL = os.path.join(ROOT, "llama2.c", "model_esp_smoke.bin")
DEFAULT_TOK = os.path.join(ROOT, "data", "tok8192.bin")


def parse_partitions(path):
    """Return {label: (offset, size)} from an ESP-IDF partition CSV."""
    out = {}
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            cols = [c.strip() for c in line.split(",")]
            if len(cols) < 5:
                continue
            label, _type, _sub, offset, size = cols[0], cols[1], cols[2], cols[3], cols[4]
            out[label] = (int(offset, 0), int(size, 0))
    return out


def human(n):
    return f"{n:,} bytes ({n / 1048576:.2f} MB)"


def find_esptool():
    """Locate PlatformIO's bundled esptool.

    esptool is NOT installed into the PlatformIO penv as an importable module;
    it lives in the separate `tool-esptoolpy` package as a thin wrapper script
    next to the `esptool` package directory. Running that wrapper with the
    package dir as cwd makes `import esptool` resolve.

    Returns (argv_prefix, cwd) or (None, None).
    """
    core = os.environ.get("PLATFORMIO_CORE_DIR") or os.path.expanduser("~/.platformio")
    wrapper = os.path.join(core, "packages", "tool-esptoolpy", "esptool.py")
    if os.path.exists(wrapper):
        return [sys.executable, "esptool.py"], os.path.dirname(wrapper)
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="COM3")
    ap.add_argument("--baud", type=int, default=921600)
    ap.add_argument("--model", default=None, help="ESP-format model (magic 'LLME')")
    ap.add_argument("--tok", default=DEFAULT_TOK)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    model = a.model or (DEFAULT_MODEL if os.path.exists(DEFAULT_MODEL) else FALLBACK_MODEL)

    parts = parse_partitions(PARTS_CSV)
    if "model" not in parts or "tok" not in parts:
        print("partitions.csv is missing 'model' or 'tok'", file=sys.stderr)
        return 1

    jobs = []
    for label, path in (("model", model), ("tok", a.tok)):
        if not os.path.exists(path):
            print(f"!! {label}: {path} does not exist", file=sys.stderr)
            return 1
        off, size = parts[label]
        fsize = os.path.getsize(path)
        status = "OK" if fsize <= size else "TOO BIG"
        print(f"{label:5s} {os.path.relpath(path, ROOT)}")
        print(f"      -> offset 0x{off:06X}  file {human(fsize)}  partition {human(size)}  [{status}]")
        if fsize > size:
            return 1
        jobs.append((off, path))

    if a.dry_run:
        print("\n(dry run -- nothing written)")
        return 0

    prefix, cwd = find_esptool()
    if prefix is None:
        print("could not find PlatformIO's tool-esptoolpy package", file=sys.stderr)
        return 1

    cmd = prefix + ["--chip", "esp32s3", "--port", a.port, "--baud", str(a.baud),
                    "--before", "default_reset", "--after", "hard_reset",
                    "write_flash"]
    for off, path in jobs:
        # esptool runs with cwd set to its own package directory, so the paths
        # it receives must be absolute.
        cmd += [f"0x{off:X}", os.path.abspath(path)]

    print("\nmodel partition is 8.95 MB, this takes a minute...")
    rc = subprocess.call(cmd, cwd=cwd)
    if rc != 0:
        print(f"esptool failed with code {rc}", file=sys.stderr)
        return rc
    print("\nassets flashed -- run tools/serial_read.py to re-run the self-test")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
