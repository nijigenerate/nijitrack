#!/usr/bin/env bash
# Explicit CI integration test. Fixtures stay in results, never in the package.
set -euo pipefail
native_root=$(CDPATH= cd "$(dirname "$0")/.." && pwd -P)
fixtures="$native_root/results/ci-fixtures"
mkdir -p "$fixtures"
sha256() {
  local result
  if command -v sha256sum >/dev/null; then result=$(sha256sum "$1"); else result=$(shasum -a 256 "$1"); fi
  printf '%s\n' "${result%% *}"
}
while read -r name expected url; do
  [[ -n "$name" ]] || continue
  if [[ ! -f "$fixtures/$name" ]]; then
    curl --fail --location --retry 3 "$url" -o "$fixtures/$name.partial"
    [[ $(sha256 "$fixtures/$name.partial") == "$expected" ]] || { printf 'Fixture checksum mismatch: %s\n' "$name" >&2; exit 1; }
    mv "$fixtures/$name.partial" "$fixtures/$name"
  fi
  [[ $(sha256 "$fixtures/$name") == "$expected" ]] || { printf 'Fixture checksum mismatch: %s\n' "$name" >&2; exit 1; }
done < "$native_root/tests/tracking-fixtures.txt"
executable="$native_root/build-sdl-static/tracking-sdk-smoke"
[[ ! -f "$executable.exe" ]] || executable="$executable.exe"
"$executable" "$fixtures/face_landmarker.task" "$fixtures/hand_landmarker.task" \
  "$fixtures/portrait.jpg" "$fixtures/right_hands.jpg"
