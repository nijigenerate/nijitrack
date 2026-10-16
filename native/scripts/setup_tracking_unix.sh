#!/usr/bin/env bash
set -euo pipefail
. "$(dirname "$0")/tracking_unix.sh"
offline=${1:-OFF}
spec="$native_root/toolchain.unix.lock.json"
mkdir -p "$tools" "$downloads" "$sources"
bp=$(jq -er ".\"$platform\".bazel_platform" "$spec")
bh=$(jq -er ".\"$platform\".bazel_sha256" "$spec")
download "https://github.com/bazelbuild/bazel/releases/download/7.4.1/bazel-7.4.1-$bp" "$tools/bazel" "$bh"
chmod +x "$tools/bazel"
hp=$(jq -er ".\"$platform\".halide_platform" "$spec")
hh=$(jq -er ".\"$platform\".halide_sha256" "$spec")
hn="Halide-15.0.1-$hp"
download "https://github.com/halide/Halide/releases/download/v15.0.1/$hn-4c63f1befa1063184c5982b11b6a2cc17d4e5815.tar.gz" "$downloads/$hn.tar.gz" "$hh"
[[ -d "$tools/$hn" ]] || tar -xf "$downloads/$hn.tar.gz" -C "$tools"
archive="$native_root/$(asset_value file)"
download "$(asset_value url)" "$archive" "$(asset_value sha256)"
[[ -d "$native_root/$(asset_value directory)" ]] || tar -xf "$archive" -C "$sources"
download "$(jq -er '.sdl3.url' "$spec")" "$downloads/sdl3-3.4.18-source.tar.gz" "$(jq -er '.sdl3.sha256' "$spec")"
[[ -d "$sources/SDL3-3.4.18" ]] || tar -xf "$downloads/sdl3-3.4.18-source.tar.gz" -C "$sources"
cmake -S "$sources/SDL3-3.4.18" -B "$native_root/build-sdl3" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
  "-DCMAKE_INSTALL_PREFIX=$tools/sdl3" -DCMAKE_INSTALL_LIBDIR=lib \
  -DSDL_SHARED=ON -DSDL_STATIC=OFF -DSDL_TESTS=OFF -DSDL_EXAMPLES=OFF
cmake --build "$native_root/build-sdl3" --parallel 4
cmake --install "$native_root/build-sdl3"
jq -n --arg cmake "$(command -v cmake)" --arg bazel "$tools/bazel" \
  --arg halide "$tools/$hn" --arg sdl3 "$tools/sdl3" \
  --arg mediapipe "$native_root/$(asset_value directory)" \
  '{cmake:$cmake,bazel:$bazel,halide:$halide,sdl3:$sdl3,mediapipe:$mediapipe}' > "$tools/environment.json"
