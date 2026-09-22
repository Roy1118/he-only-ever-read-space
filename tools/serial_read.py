#!/usr/bin/env python3
"""Capture ESP32 serial output, optionally resetting the board first.

The self-test prints everything once, at boot, so we must reset *after* the port
is open -- otherwise the banner scrolls past before we start listening.

Usage:
    python tools/serial_read.py -s 30 -o selftest.log
    python tools/serial_read.py --no-reset -s 8      # just tail the heartbeat
"""
import argparse
import sys
import time

import serial


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--port", default="COM3")
    ap.add_argument("-b", "--baud", type=int, default=115200)
    ap.add_argument("-s", "--seconds", type=float, default=30.0,
                    help="how long to listen")
    ap.add_argument("-o", "--out", default=None, help="also write raw log here")
    ap.add_argument("--no-reset", action="store_true",
                    help="do not pulse the reset line (just listen)")
    a = ap.parse_args()

    try:
        ser = serial.Serial(a.port, a.baud, timeout=0.2)
    except Exception as exc:
        print(f"cannot open {a.port}: {exc}", file=sys.stderr)
        return 1

    try:
        if not a.no_reset:
            # ESP32-S3 auto-reset: GPIO0 must be high (DTR=False) while EN is
            # pulsed low then high again (RTS True -> False).
            ser.setDTR(False)
            ser.setRTS(True)
            time.sleep(0.20)
            ser.setRTS(False)
            time.sleep(0.10)
            ser.reset_input_buffer()
            print(f"--- reset {a.port} @ {a.baud}, listening {a.seconds:g}s ---",
                  file=sys.stderr)

        deadline = time.time() + a.seconds
        chunks = []
        while time.time() < deadline:
            data = ser.read(8192)
            if data:
                text = data.decode("utf-8", "replace")
                sys.stdout.write(text)
                sys.stdout.flush()
                chunks.append(text)
    except KeyboardInterrupt:
        pass
    finally:
        ser.close()

    if a.out:
        with open(a.out, "w", encoding="utf-8", errors="replace") as f:
            f.write("".join(chunks))
        print(f"\n--- raw log written to {a.out} ---", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
