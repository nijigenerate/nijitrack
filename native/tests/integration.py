#!/usr/bin/env python3
"""Offline integration tests. Python is test tooling, never the tracking runtime."""
import argparse
import json
import math
import tempfile
from pathlib import Path
import socket
import statistics
import struct
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]


def osc_decode(data):
    def string(offset):
        end = data.index(b"\0", offset)
        return data[offset:end].decode(), (end + 4) & ~3
    address, pos = string(0)
    tags, pos = string(pos)
    values = []
    for tag in tags[1:]:
        if tag == "s":
            value, pos = string(pos)
        elif tag in ("f", "i"):
            value, = struct.unpack_from(">" + tag, data, pos)
            pos += 4
        else:
            raise AssertionError(f"Unexpected OSC type {tag}")
        values.append(value)
    assert pos == len(data)
    return address, tags, values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", default="build-cmake/nijitrack-cpp")
    parser.add_argument("--image", required=True, help="Existing local face image; never downloaded or copied")
    parser.add_argument("--video", required=True, help="Existing local video with a face in every frame (at least 5 frames)")
    parser.add_argument("--model", default="../face_landmarker_v2_with_blendshapes.task")
    args = parser.parse_args()
    image = str(Path(args.image).resolve())
    video_input = str(Path(args.video).resolve())
    temporary = tempfile.TemporaryDirectory(prefix="nijitrack-tests-")
    temporary_path = Path(temporary.name)
    binary = str((ROOT / args.binary).resolve())
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    checks = []

    def run(name, *options, ok=True):
        result = subprocess.run([binary, "--model", args.model, *options], cwd=ROOT, capture_output=True, text=True,
                                timeout=90)
        (out / f"{name}.stderr.log").write_text(result.stderr)
        assert (result.returncode == 0) == ok, (name, result.returncode, result.stderr)
        checks.append(name)
        return result

    run("still", "--image", image, "--repeat", "30", "--output",
        "results/still.jsonl", "--overlay", "results/overlay.ppm")
    frames = [json.loads(s) for s in (out / "still.jsonl").read_text().splitlines()]
    assert len(frames) == 30
    for f in frames:
        assert f["face_count"] == 1 and len(f["landmarks"]) == 478
        assert len(f["blendshapes"]) == 52
        assert all(math.isfinite(x) for p in f["landmarks"] for x in p)
        assert all(math.isfinite(x) and 0 <= x <= 1 for x in f["blendshapes"].values())
        pose = f["pose"]
        assert abs(sum(x*x for x in pose["quaternion_xyzw"]) - 1) < 1e-6
        assert all(-1 <= x <= 1 for key in ("gaze_left", "gaze_right") for x in pose[key])
        m = f["facial_transformation_matrix"]
        assert m["rows"] == m["cols"] == 4 and len(m["data"]) == 16
        assert all(math.isfinite(x) for x in m["data"])
    assert all(b["timestamp_ms"] > a["timestamp_ms"] for a, b in zip(frames, frames[1:]))
    checks.append("landmarks-blendshapes-pose-matrix-contract")

    run("flip", "--image", image, "--output", "results/flip.jsonl",
        "--overlay", "results/flip.ppm", "--flip")
    flipped = json.loads((out / "flip.jsonl").read_text())
    for key in ("landmarks", "blendshapes", "pose", "facial_transformation_matrix"):
        assert flipped[key] == frames[0][key], key
    original = (out / "overlay.ppm").read_bytes().split(b"\n", 3)
    mirror = (out / "flip.ppm").read_bytes().split(b"\n", 3)
    assert original[:3] == mirror[:3]
    w, h = map(int, original[1].split())
    expected = b"".join(b"".join(original[3][(y*w+x)*3:(y*w+x+1)*3]
                                for x in reversed(range(w))) for y in range(h))
    assert expected == mirror[3]
    checks.append("flip-is-display-only")

    black = temporary_path / "black.ppm"
    black.write_bytes(b"P6\n640 480\n255\n" + bytes(640*480*3))
    run("no-face", "--image", str(black), "--repeat", "3", "--output",
        "results/no-face.jsonl")
    for line in (out / "no-face.jsonl").read_text().splitlines():
        f = json.loads(line)
        assert f["face_count"] == 0 and f["landmarks"] == [] and f["pose"] is None

    run("video", "--video", video_input, "--output", "results/video.jsonl")
    video = [json.loads(s) for s in (out / "video.jsonl").read_text().splitlines()]
    assert len(video) >= 5 and all(f["face_count"] == 1 for f in video)
    assert all(b["timestamp_ms"] > a["timestamp_ms"] for a, b in zip(video, video[1:]))
    assert video[0]["landmarks"] != video[-1]["landmarks"]
    checks.append("continuous-changing-video")
    run("video-limit", "--video", video_input, "--frames", "5",
        "--output", "results/video-limit.jsonl")
    assert len((out / "video-limit.jsonl").read_text().splitlines()) == 5

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024*1024)
    sock.bind(("127.0.0.1", 0))
    sock.settimeout(0.1)
    command = [binary, "--model", args.model, "--image", image, "--repeat", "3", "--osc-host",
               "127.0.0.1", "--osc-port", str(sock.getsockname()[1]), "--output",
               "results/osc.jsonl"]
    packets = []
    with subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          text=True) as process:
        deadline = time.monotonic() + 30
        idle = 0
        while time.monotonic() < deadline:
            try:
                packets.append(osc_decode(sock.recv(65535)))
                idle = 0
            except socket.timeout:
                if process.poll() is not None:
                    idle += 1
                    if idle == 3:
                        break
        stdout, stderr = process.communicate(timeout=5)
        assert process.returncode == 0, (stdout, stderr)
        (out / "osc.stderr.log").write_text(stderr)
    sock.close()
    assert len(packets) == 3*58, len(packets)
    json_frames = [json.loads(s) for s in (out / "osc.jsonl").read_text().splitlines()]
    for index, f in enumerate(json_frames):
        group = packets[index*58:(index+1)*58]
        address, tags, values = group[0]
        assert address == "/VMC/Ext/Bone/Pos" and tags == ",sfffffff" and values[0] == "Head"
        expected = f["pose"]["position_normalized"] + f["pose"]["quaternion_xyzw"]
        assert all(abs(a-b) < 1e-6 for a, b in zip(values[1:], expected))
        assert group[-1] == ("/VMC/Ext/Blend/Apply", ",", [])
        blends = {v[0]: v[1] for a, t, v in group[1:-1]
                  if a == "/VMC/Ext/Blend/Val" and t == ",sf"}
        assert len(blends) == 56
        for name, score in f["blendshapes"].items():
            assert abs(blends[name] - score) < 1e-6
    checks.append("OSC-174-datagrams-types-values-order")

    devices = run("devices", "--list-devices")
    assert isinstance(json.loads(devices.stdout), list)
    run("bad-model", "--image", image, "--model", "missing.task", ok=False)
    run("corrupt-model", "--image", image, "--model", image,
        ok=False)
    run("bad-image", "--image", "models/face_landmarker.task", ok=False)
    run("bad-fps", "--image", image, "--fps", "nan", ok=False)
    run("missing-value", "--image", ok=False)
    run("protect-input", "--image", image, "--output", image,
        ok=False)
    invalid = temporary_path / "invalid.mp4"
    invalid.write_bytes(b"invalid")
    run("bad-video", "--video", str(invalid), ok=False)

    times = [f["inference_ms"] for f in video]
    report = {"status": "PASS", "checks": checks, "video_frames": len(video),
              "faces_detected": sum(f["face_count"] == 1 for f in video),
              "video_mean_inference_ms": statistics.mean(times),
              "video_median_inference_ms": statistics.median(times),
              "video_max_inference_ms": max(times), "osc_packets_verified": len(packets),
              "camera_devices": json.loads(devices.stdout),
              "input_policy": "Explicit local inputs; no bundled or downloaded face images/videos."}
    (out / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    temporary.cleanup()


if __name__ == "__main__":
    main()
