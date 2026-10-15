#!/usr/bin/env python3
"""Verify the pinned SDK and official headers without loading code."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    lock = json.loads((ROOT / "assets.lock.json").read_text())
    for name, digest in lock["files"].items():
        actual = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        if actual != digest:
            raise SystemExit(f"SHA-256 mismatch: {name}")
    library = ROOT / "vendor/sdk/libmediapipe.so"
    data = library.read_bytes().lower()
    for forbidden in (b"clearcut", b"play.googleapis.com", b"googleapis.com/log"):
        if forbidden in data:
            raise SystemExit(f"Unexpected string: {forbidden!r}")
    symbols = subprocess.check_output(["nm", "-D", "--defined-only", str(library)], text=True)
    exported = {line.split()[-1].split("@")[0] for line in symbols.splitlines() if line.split()}
    for symbol in ("MpFaceLandmarkerCreate", "MpFaceLandmarkerDetectForVideo",
                   "MpFaceLandmarkerCloseResult", "MpFaceLandmarkerClose",
                   "MpImageCreateFromFile", "MpImageCreateFromUint8Data", "MpImageFree",
                   "MpErrorFree"):
        if symbol not in exported:
            raise SystemExit(f"Missing export: {symbol}")
    print(f"PASS: {len(lock['files'])} SHA-256 checks, expected C API exports, string scan")
    print("This verifies pinned bytes and selected properties, not absence of malicious code.")


if __name__ == "__main__":
    main()
