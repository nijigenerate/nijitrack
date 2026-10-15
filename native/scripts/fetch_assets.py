#!/usr/bin/env python3
"""Fetch the pinned host SDK only. Never download models, images or videos."""
import hashlib
import io
import json
import platform
from pathlib import Path
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def host_platform():
    system = {"Linux": "linux", "Windows": "windows", "Darwin": "macos"}.get(platform.system())
    machine = {"AMD64": "x86_64", "x86_64": "x86_64", "arm64": "arm64", "aarch64": "arm64"}.get(platform.machine())
    return f"{system}-{machine}"


def fetch(url, digest):
    print(f"Downloading {url}", flush=True)
    request = urllib.request.Request(url, headers={"User-Agent": "nijitrack-native/0.1"})
    with urllib.request.urlopen(request, timeout=180) as response:
        data = response.read(256 * 1024 * 1024 + 1)
    if len(data) > 256 * 1024 * 1024:
        raise RuntimeError("Download size limit exceeded")
    if hashlib.sha256(data).hexdigest() != digest:
        raise RuntimeError(f"SHA-256 mismatch for {url}")
    return data


def main():
    lock = json.loads((ROOT / "assets.lock.json").read_text())
    key = host_platform()
    if key not in lock["sdk"]["platforms"]:
        raise SystemExit(f"No pinned SDK for {key}")
    spec = lock["sdk"]["platforms"][key]
    url = lock["sdk"]["repository"] + "/releases/download/" + lock["sdk"]["release"] + "/" + spec["archive"]
    archive = fetch(url, spec["sha256"])
    with zipfile.ZipFile(io.BytesIO(archive)) as z:
        names = z.namelist()
        required = {spec["library"], "LICENSE.mediapipe", "LICENSE.opencv", "SOURCE.txt", "SHA256SUMS"}
        if key == "windows-x86_64":
            required.add("opencv_world4140.dll")
        if set(names) != required or len(names) != len(required):
            raise RuntimeError("Unexpected SDK ZIP members")
        for name in names:
            if z.getinfo(name).file_size > 256 * 1024 * 1024:
                raise RuntimeError("SDK member too large")
        if lock["sdk"]["source_sha"] not in z.read("SOURCE.txt").decode():
            raise RuntimeError("Unexpected SDK source revision")
        checksums = z.read("SHA256SUMS").decode().splitlines()
        for line in checksums:
            digest, name = line.split(maxsplit=1)
            name = name.lstrip(" *")
            if name not in required or hashlib.sha256(z.read(name)).hexdigest() != digest:
                raise RuntimeError("SDK member hash mismatch")
        sdk = ROOT / "vendor/sdk"
        sdk.mkdir(parents=True, exist_ok=True)
        for name in names:
            temporary = sdk / (name + ".partial")
            temporary.write_bytes(z.read(name))
            temporary.replace(sdk / name)
    print(f"Verified and installed {key} SDK; no image or model downloaded")


if __name__ == "__main__":
    main()
