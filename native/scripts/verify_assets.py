#!/usr/bin/env python3
"""Verify pinned headers and SDK files without executing the library."""
import hashlib
import json
from pathlib import Path
from fetch_assets import host_platform

ROOT = Path(__file__).resolve().parents[1]


def main():
    lock = json.loads((ROOT / "assets.lock.json").read_text())
    for name, digest in lock["files"].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
            raise SystemExit(f"SHA-256 mismatch: {name}")
    sdk = ROOT / "vendor/sdk"
    for line in (sdk / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        name = name.lstrip(" *")
        if Path(name).name != name or hashlib.sha256((sdk / name).read_bytes()).hexdigest() != digest:
            raise SystemExit(f"SDK SHA-256 mismatch: {name}")
    spec = lock["sdk"]["platforms"][host_platform()]
    data = (sdk / spec["library"]).read_bytes().lower()
    for forbidden in (b"clearcut", b"play.googleapis.com", b"googleapis.com/log"):
        if forbidden in data:
            raise SystemExit(f"Unexpected string: {forbidden!r}")
    if lock["sdk"]["source_sha"] not in (sdk / "SOURCE.txt").read_text():
        raise SystemExit("SDK source revision mismatch")
    print(f"PASS: pinned header hashes, SDK member checksums, source revision and string scan ({host_platform()})")
    print("The SDK archive digest is checked by fetch_assets.py; this is not a proof of absence of malicious code.")


if __name__ == "__main__":
    main()
