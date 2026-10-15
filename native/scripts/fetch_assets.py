#!/usr/bin/env python3
"""Restore missing pinned downloads. Not used by the C++ application at runtime."""
import hashlib
import json
from pathlib import Path
import urllib.request
import zipfile
import io

ROOT = Path(__file__).resolve().parents[1]


def fetch(url, digest):
    print(f"Downloading {url}", flush=True)
    request = urllib.request.Request(url, headers={"User-Agent": "nijitrack-cpp-prototype/0.1"})
    with urllib.request.urlopen(request, timeout=180) as response:
        data = response.read(64 * 1024 * 1024 + 1)
    if len(data) > 64 * 1024 * 1024:
        raise RuntimeError("Download size limit exceeded")
    if hashlib.sha256(data).hexdigest() != digest:
        raise RuntimeError(f"SHA-256 mismatch for {url}")
    return data


def put(relative, data):
    destination = ROOT / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    temporary.write_bytes(data)
    temporary.replace(destination)


def main():
    lock = json.loads((ROOT / "assets.lock.json").read_text())
    sdk = lock["sdk"]
    members = ("libmediapipe.so", "LICENSE.mediapipe", "LICENSE.opencv", "SOURCE.txt", "SHA256SUMS")
    if any(not (ROOT / "vendor/sdk" / name).is_file() for name in members):
        archive = fetch(sdk["url"], sdk["archive_sha256"])
        with zipfile.ZipFile(io.BytesIO(archive)) as z:
            if set(z.namelist()) != set(members):
                raise RuntimeError("Unexpected SDK ZIP members")
            for name in members:
                if z.getinfo(name).file_size > 64 * 1024 * 1024:
                    raise RuntimeError("SDK member too large")
                data = z.read(name)
                path = "vendor/sdk/" + name
                if hashlib.sha256(data).hexdigest() != lock["files"][path]:
                    raise RuntimeError(f"SDK member hash mismatch: {name}")
                put(path, data)
    for path, url in lock["downloads"].items():
        if not (ROOT / path).is_file():
            put(path, fetch(url, lock["files"][path]))
    for path in lock["files"]:
        if path.startswith("vendor/include/") and not (ROOT / path).is_file():
            upstream_path = path.removeprefix("vendor/include/")
            url = "https://raw.githubusercontent.com/google-ai-edge/mediapipe/" + sdk["source_sha"] + "/" + upstream_path
            put(path, fetch(url, lock["files"][path]))
    from verify_assets import main as verify
    verify()


if __name__ == "__main__":
    main()
