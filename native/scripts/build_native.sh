#!/usr/bin/env bash
set -euo pipefail
. "$(dirname "$0")/tracking_unix.sh"
load_environment
build="$native_root/build-sdl-static"
"$cmake" -S "$native_root" -B "$build" -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=clang++ "-DCMAKE_PREFIX_PATH=$sdl3" \
  -DNIJITRACK_REQUIRE_CLANG=ON -DNIJITRACK_KEYBOARD=ON -DBUILD_TESTING=ON
"$cmake" --build "$build" --parallel 2
ctest --test-dir "$build" --output-on-failure
ctest --test-dir "$work/halide" --output-on-failure
bash "$native_root/scripts/verify_native_dependencies.sh" "$build"
if [[ ${1:-} == --install ]]; then
  "$cmake" --install "$build" --prefix "$native_root/install"
  bash "$native_root/scripts/verify_native_dependencies.sh" "$native_root/install/bin"
fi
