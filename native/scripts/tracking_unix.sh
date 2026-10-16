#!/usr/bin/env bash
# Shared shell helpers for Linux/macOS. No interpreter-based build generators.
set -euo pipefail
native_root=$(CDPATH= cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
work="$native_root/build-mediapipe"
tools="$native_root/vendor/toolchain"
downloads="$native_root/vendor/downloads"
sources="$native_root/vendor/sources"
export LC_ALL=C
die() { printf '%s\n' "$*" >&2; exit 1; }
for tool in cmake ninja clang clang++ git jq curl tar; do
  command -v "$tool" >/dev/null || die "Required build tool: $tool"
done
case $(uname -s):$(uname -m) in
  Linux:x86_64) platform=linux-x86_64; command -v readelf >/dev/null || die 'Install binutils (readelf)' ;;
  Darwin:arm64) platform=macos-arm64; command -v otool >/dev/null || die 'Install Xcode command line tools' ;;
  *) die 'Supported shell hosts: Linux x86_64 and macOS arm64' ;;
esac
sha256() {
  local result
  if command -v sha256sum >/dev/null; then result=$(sha256sum "$1"); else result=$(shasum -a 256 "$1"); fi
  printf '%s\n' "${result%% *}"
}
source_asset=$(jq -ce '.assets[] | select(.name == "mediapipe-source")' "$native_root/toolchain.lock.json")
asset_value() { printf '%s\n' "$source_asset" | jq -er ".$1"; }
check_hash() { [[ $(sha256 "$1") == "$2" ]] || die "Checksum mismatch: $1"; }
download() {
  local url=$1 path=$2 hash=$3
  mkdir -p "$(dirname "$path")"
  if [[ ! -f "$path" ]]; then
    [[ ${offline:-OFF} != ON ]] || die "Missing offline asset: $path"
    curl --fail --location --retry 3 "$url" -o "$path.partial"
    check_hash "$path.partial" "$hash"
    mv "$path.partial" "$path"
  fi
  check_hash "$path" "$hash"
}
safe_remove_source() {
  case "$1" in "$work/src"|"$work/source-staging") ;; *) die "Unexpected cleanup target: $1" ;; esac
  [[ ! -L "$1" ]] || die "Workspace root is a link: $1"
  rm -rf -- "$1"
}
child_path() {
  case "$2" in /*|../*|*/../*|*/..) die "Path escapes root: $2" ;; esac
  printf '%s/%s\n' "$1" "$2"
}
git_apply() {
  local root=$1 patch=$2; shift 2
  GIT_CEILING_DIRECTORIES=$(dirname "$root") git -c core.autocrlf=false -C "$root" apply "$@" "$patch"
}
patch_series() {
  local root=$1 manifest=$2 allow_applied=$3 count index files f path before after actual state item patch
  local pending=()
  count=$(printf '%s\n' "$manifest" | jq '.series | length')
  for ((index=0; index<count; index++)); do
    item=$(printf '%s\n' "$manifest" | jq -c ".series[$index]")
    files=$(printf '%s\n' "$item" | jq '.files | length')
    state=''
    for ((f=0; f<files; f++)); do
      path=$(printf '%s\n' "$item" | jq -r ".files[$f].path")
      before=$(printf '%s\n' "$item" | jq -r ".files[$f].before")
      after=$(printf '%s\n' "$item" | jq -r ".files[$f].after")
      actual=$(sha256 "$(child_path "$root" "$path")")
      if [[ "$actual" == "$before" ]]; then
        [[ "$state" != applied ]] || die 'Partially applied patch'
        state=original
      elif [[ "$allow_applied" == yes && "$actual" == "$after" ]]; then
        [[ "$state" != original ]] || die 'Partially applied patch'
        state=applied
      else die "Unsupported source content: $root/$path. Rebase patches."; fi
    done
    if [[ "$state" == original ]]; then
      patch=$(printf '%s\n' "$item" | jq -r '.patch')
      git_apply "$root" "$(child_path "$native_root/patches" "$patch")" --check
      pending+=("$index")
    fi
  done
  for index in ${pending[@]+"${pending[@]}"}; do
    item=$(printf '%s\n' "$manifest" | jq -c ".series[$index]")
    patch=$(printf '%s\n' "$item" | jq -r '.patch')
    git_apply "$root" "$(child_path "$native_root/patches" "$patch")" --whitespace=nowarn
    files=$(printf '%s\n' "$item" | jq '.files | length')
    for ((f=0; f<files; f++)); do
      path=$(printf '%s\n' "$item" | jq -r ".files[$f].path")
      after=$(printf '%s\n' "$item" | jq -r ".files[$f].after")
      check_hash "$(child_path "$root" "$path")" "$after"
    done
  done
}
copy_overlays() {
  local root=$1 manifest=$2 count index source target before
  count=$(printf '%s\n' "$manifest" | jq '.overlays | length')
  for ((index=0; index<count; index++)); do
    source=$(printf '%s\n' "$manifest" | jq -r ".overlays[$index].source")
    target=$(printf '%s\n' "$manifest" | jq -r ".overlays[$index].target")
    before=$(printf '%s\n' "$manifest" | jq -r ".overlays[$index].before // empty")
    target=$(child_path "$root" "$target")
    [[ -z "$before" ]] || check_hash "$target" "$before"
    mkdir -p "$(dirname "$target")"
    cp "$(child_path "$native_root" "$source")" "$target"
  done
}
recipe_hash() {
  (cd "$native_root"
    { find patches -type f \( -name '*.patch' -o -name '*.json' \)
      find toolchain/mediapipe -type f
      find scripts -type f -name '*.sh'
      printf '%s\n' build.sh toolchain.lock.json assets.lock.json
    } | sort | while IFS= read -r path; do printf '%s:%s\n' "$path" "$(sha256 "$path")"; done
  ) > "$work/recipe.inputs"
  sha256 "$work/recipe.inputs"
}
load_environment() {
  local file="$tools/environment.json"
  [[ -f "$file" ]] || die 'Run bash native/build.sh setup first'
  cmake=$(jq -er '.cmake' "$file")
  bazel=$(jq -er '.bazel' "$file")
  halide=$(jq -er '.halide' "$file")
  sdl3=$(jq -er '.sdl3' "$file")
  mediapipe=$(jq -er '.mediapipe' "$file")
  guard="$tools/no-python"
  export PATH="$guard:$PATH" PYTHON_BIN_PATH="$guard/python" PYTHON3_BIN_PATH="$guard/python"
  export NIJITRACK_PYTHON_GUARD_LOG="$native_root/results/build-environment/python-guard.log"
  export Halide_ROOT="$halide" SDL3_ROOT="$sdl3"
}
