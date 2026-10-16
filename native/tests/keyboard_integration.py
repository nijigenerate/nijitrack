#!/usr/bin/env python3
"""Real X11 RECORD -> key conversion -> OSC test using isolated Xvfb/XTEST."""
import argparse
import json
import os
from pathlib import Path
import select
import socket
import secrets
import struct
import subprocess
import tempfile
import time

from integration import osc_decode

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"


def ready_line(process, needle, timeout=15):
    end = time.monotonic() + timeout
    lines = []
    while time.monotonic() < end:
        if select.select([process.stderr], [], [], 0.1)[0]:
            line = process.stderr.readline()
            lines.append(line)
            if needle in line:
                return lines
        if process.poll() is not None:
            break
    raise RuntimeError("Keyboard startup failed: " + "".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Existing local face image; never downloaded or copied")
    parser.add_argument("--model", default=str(ROOT / "vendor/models/face_landmarker.task"))
    args = parser.parse_args()
    image = str(Path(args.image).resolve())
    OUT.mkdir(exist_ok=True)
    env = os.environ.copy()
    xvfb = env.get("NIJI_XVFB", "Xvfb")
    xdotool = env.get("NIJI_XDOTOOL", "xdotool")
    tcp = env.get("NIJI_X11_TCP") == "1"
    temporary = tempfile.TemporaryDirectory(prefix="niji-x11-")
    auth_path = Path(temporary.name) / "Xauthority"
    cookie = secrets.token_bytes(16)

    def authority(number):
        fields = [b"", str(number).encode(),
                  b"MIT-MAGIC-COOKIE-1", cookie]
        auth_path.write_bytes(struct.pack(">H", 65535) + b"".join(struct.pack(">H", len(f)) + f for f in fields))
        auth_path.chmod(0o600)

    server_args = [xvfb, "-displayfd", "1", "-screen", "0", "800x600x24",
                   "+extension", "RECORD", "+extension", "XTEST"]
    if tcp:
        authority(0)
        env["XAUTHORITY"] = str(auth_path)
        server_args += ["-nolisten", "unix", "-nolisten", "local", "-nolisten", "inet6",
                        "-listen", "tcp", "-auth", str(auth_path)]
    else:
        server_args += ["-nolisten", "tcp", "-ac"]
    with (OUT / "xvfb.log").open("w") as server_log:
        server = subprocess.Popen(server_args, stdout=subprocess.PIPE,
                                  stderr=server_log, text=True, env=env)
        try:
            if not select.select([server.stdout], [], [], 15)[0]:
                raise RuntimeError("Xvfb startup timed out; see results/xvfb.log")
            number = server.stdout.readline().strip()
            if not number:
                raise RuntimeError("Xvfb startup failed; see results/xvfb.log")
            if tcp:
                authority(number)
            env["DISPLAY"] = ("127.0.0.1:" if tcp else ":") + number
            env["XDG_SESSION_TYPE"] = "x11"
            env.pop("WAYLAND_DISPLAY", None)

            def inject():
                subprocess.run([xdotool, "keydown", "a", "keyup", "a", "keydown", "Shift_L",
                                "keydown", "b", "keyup", "Shift_L", "keyup", "b", "key", "Return", "key", "space"],
                               env=env, check=True, timeout=5, capture_output=True)

            probe_binary = env.get("NIJI_KEYBOARD_PROBE", str(ROOT / "build-sdl-static/keyboard/keyboard-probe"))
            with subprocess.Popen([probe_binary, "1000"],
                                  env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                  text=True) as probe:
                startup = ready_line(probe, "READY")
                inject()
                stdout, stderr = probe.communicate(timeout=5)
                assert probe.returncode == 0, stderr
                events = [json.loads(s) for s in stdout.splitlines()]
                expected = [{"pressed": 1, "text": "a", "codepoint": 97},
                            {"pressed": 0, "text": "a", "codepoint": 97},
                            {"pressed": 1, "text": "B", "codepoint": 66},
                            {"pressed": 0, "text": "B", "codepoint": 66}]
                assert events == expected, events
                (OUT / "keyboard-probe.stderr.log").write_text("".join(startup) + stderr)

            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024*1024)
            sock.bind(("127.0.0.1", 0))
            sock.settimeout(0.05)
            tracker_binary = env.get("NIJI_TRACKER_BIN", str(ROOT / "build-sdl-static/nijitrack"))
            command = [tracker_binary, "--model", args.model, "--image", image,
                       "--repeat", "100", "--keyboard", "--osc-host", "127.0.0.1", "--osc-port",
                       str(sock.getsockname()[1]), "--output", "results/keyboard-face.jsonl"]
            key_packets = []
            total = 0
            with subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE, text=True) as process:
                startup = ready_line(process, "keyboard=x11-xrecord")
                inject()
                deadline = time.monotonic() + 20
                idle = 0
                while time.monotonic() < deadline:
                    try:
                        packet = osc_decode(sock.recv(65535))
                        total += 1
                        if packet[0] == "/VMC/Ext/Key":
                            key_packets.append(packet)
                        idle = 0
                    except socket.timeout:
                        if process.poll() is not None:
                            idle += 1
                            if idle == 3:
                                break
                stdout, stderr = process.communicate(timeout=5)
                assert process.returncode == 0, (stdout, stderr)
                (OUT / "keyboard-face.stderr.log").write_text("".join(startup) + stderr)
            sock.close()
            assert key_packets == [("/VMC/Ext/Key", ",isi", [e["pressed"], e["text"], e["codepoint"]])
                                   for e in expected], key_packets
            frames = [json.loads(s) for s in (OUT / "keyboard-face.jsonl").read_text().splitlines()]
            assert len(frames) == 100 and all(f["face_count"] == 1 for f in frames)

            report = {"status": "PASS", "backend": "x11-xrecord", "environment": "Xvfb + XTEST",
                      "events": events, "osc_key_packets": key_packets,
                      "concurrent_face_frames": len(frames), "total_osc_packets_received": total,
                      "special_key_policy": "Shift, Return and Space are ignored, matching the old character-only OSC behavior",
                      "release_policy": "Release keeps the character captured at key-down even when Shift is released first",
                      "windows_macos": "Source backends added, not compiled or runtime-tested in this Linux environment"}
            (OUT / "keyboard-verification.json").write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(report, indent=2))
        finally:
            server.terminate()
            try:
                server.wait(timeout=5)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()
            temporary.cleanup()


if __name__ == "__main__":
    main()
