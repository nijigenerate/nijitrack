#!/usr/bin/env bash
set -euo pipefail
. "$(dirname "$0")/tracking_unix.sh"
load_environment
manifest=$(cat "$native_root/patches/mediapipe/manifest.json")
version=$(asset_value version)
[[ "$version" == $(printf '%s\n' "$manifest" | jq -r '.upstream_version') ]] || die 'Source/patch versions differ; rebase patches'
archive="$native_root/$(asset_value file)"
check_hash "$archive" "$(asset_value sha256)"
safe_remove_source "$work/source-staging"
mkdir -p "$work/source-staging"
tar -xf "$archive" -C "$work/source-staging"
fresh="$work/source-staging/mediapipe-$version"
[[ $(jq -r '.sdk.source_sha' "$native_root/assets.lock.json") == "$version" ]] || die 'C API header/source versions differ'
while IFS= read -r path; do
  relative=${path#vendor/include/}
  expected=$(jq -er --arg path "$path" '.files[$path]' "$native_root/assets.lock.json")
  check_hash "$fresh/$relative" "$expected"
  check_hash "$native_root/$path" "$expected"
done < <(jq -r '.files | keys[] | select(startswith("vendor/include/"))' "$native_root/assets.lock.json")
patch_series "$fresh" "$manifest" no
copy_overlays "$fresh" "$manifest"
rm -f "$work/configuration.json"
safe_remove_source "$work/src"
mv "$fresh" "$work/src"
safe_remove_source "$work/source-staging"
source="$work/src"
package="$source/mediapipe/tasks/c/nijitrack_tracking"
mkdir -p "$package" "$work/no-python-repository" "$work/no-swift-repository" "$guard"
cp "$native_root/toolchain/mediapipe/face_hand.BUILD.bazel" "$package/BUILD.bazel"
cp "$native_root/toolchain/mediapipe/tracking_archive.bzl" "$package/"
stub="$work/no-python-repository"
: > "$stub/WORKSPACE"
: > "$work/no-swift-repository/WORKSPACE"
: > "$work/no-swift-repository/BUILD.bazel"
cp "$native_root/toolchain/mediapipe/templates/python.BUILD.bazel" "$stub/BUILD.bazel"
cp "$native_root/toolchain/mediapipe/templates/py_version.bzl" "$stub/"
printf 'interpreter = "%s"\n' "$guard/python" > "$stub/defs.bzl"
clang++ -std=c++17 -O2 "$native_root/toolchain/smoke/no_python.cpp" -o "$guard/python"
cp "$guard/python" "$guard/python3"
cp "$guard/python" "$guard/py"
"$cmake" -S "$native_root/toolchain/mediapipe/halide" -B "$work/halide" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ "-DHalide_ROOT=$halide"
"$cmake" --build "$work/halide" --parallel 2
generated="$source/mediapipe/calculators/tensor/nijitrack_halide"
mkdir -p "$generated"
cp "$work/halide/generated/nijitrack_warp.a" "$work/halide/generated/nijitrack_warp.h" "$halide/include/HalideRuntime.h" "$generated/"
cat > "$generated/BUILD.bazel" <<'BUILD'
load("@rules_cc//cc:cc_library.bzl", "cc_library")
cc_library(name = "warp", srcs = ["nijitrack_warp.a"], hdrs = ["nijitrack_warp.h", "HalideRuntime.h"], visibility = ["//mediapipe/calculators/tensor:__pkg__"])
BUILD
jq -n --arg source "$source" --arg python_stub "$stub" --arg upstream_version "$version" \
  --arg source_archive_sha256 "$(asset_value sha256)" --arg recipe_sha256 "$(recipe_hash)" \
  '{source:$source,python_stub:$python_stub,upstream_version:$upstream_version,source_archive_sha256:$source_archive_sha256,recipe_sha256:$recipe_sha256}' > "$work/configuration.json"
printf 'Prepared verified source: %s\n' "$source"
