#!/usr/bin/env bash
set -euo pipefail
. "$(dirname "$0")/tracking_unix.sh"
load_environment
config="$work/configuration.json"
[[ $(jq -er '.recipe_sha256' "$config") == $(recipe_hash) ]] || die 'Recipe changed; run prepare again'
source=$(jq -er '.source' "$config")
stub=$(jq -er '.python_stub' "$config")
patch_dependencies() {
  local count index entry repository found external
  count=$(jq 'length' "$native_root/patches/dependencies/manifest.json")
  for ((index=0; index<count; index++)); do
    entry=$(jq -c ".[$index]" "$native_root/patches/dependencies/manifest.json")
    repository=$(printf '%s\n' "$entry" | jq -r '.repository')
    found=no
    for external in "$work"/bazel/*/external; do
      [[ -d "$external/$repository" ]] || continue
      patch_series "$external/$repository" "$entry" yes
      copy_overlays "$external/$repository" "$entry"
      found=yes
    done
    [[ "$found" == yes ]] || die "Expected dependency missing: $repository"
  done
}
startup=(--nosystem_rc --nohome_rc --noworkspace_rc --batch "--output_user_root=$work/bazel")
repositories=(--enable_bzlmod=true --experimental_repo_remote_exec "--repo_env=PATH=$PATH" \
  "--override_repository=python_version_repo=$stub" "--override_repository=python=$stub")
cd "$source"
if [[ ${NIJITRACK_OFFLINE:-OFF} == ON ]]; then
  repositories+=(--nofetch)
else
  "$bazel" "${startup[@]}" fetch "${repositories[@]}" --repo=@llvm-raw --repo=@XNNPACK --repo=@com_google_protobuf
fi
patch_dependencies
"$bazel" "${startup[@]}" build "${repositories[@]}" --compilation_mode=opt --jobs=4 \
  --define=MEDIAPIPE_DISABLE_OPENCV=1 --define=MEDIAPIPE_ENABLE_HALIDE=1 \
  --copt=-DNDEBUG --copt=-DLITERT_DISABLE_NPU --cxxopt=-std=c++20 --host_cxxopt=-std=c++20 \
  --repo_env=CC=clang --repo_env=CXX=clang++ \
  "--repo_env=PYTHON_BIN_PATH=$PYTHON_BIN_PATH" "--action_env=PATH=$PATH" \
  "--action_env=PYTHON_BIN_PATH=$PYTHON_BIN_PATH" \
  "--override_repository=local_execution_config_python=$stub" --extra_toolchains=@python//:py_toolchain \
  --output_groups=+linkdeps,+linkopts //mediapipe/tasks/c/nijitrack_tracking:mediapipe_tracking
artifacts="$source/bazel-bin/mediapipe/tasks/c/nijitrack_tracking"
while IFS= read -r dependency || [[ -n "$dependency" ]]; do
  case "$dependency" in ''|//mediapipe/calculators/tensor/nijitrack_halide:warp|@//mediapipe/calculators/tensor/nijitrack_halide:warp|@@//mediapipe/calculators/tensor/nijitrack_halide:warp) ;;
    *) die "Unpackaged SDK dependency: $dependency" ;;
  esac
done < "$artifacts/mediapipe_tracking_linkdeps.txt"
sdk="$native_root/vendor/sdk-halide"
mkdir -p "$sdk/licenses"
cp "$artifacts/libmediapipe_tracking.a" "$artifacts/mediapipe_tracking_linkopts.txt" "$sdk/"
cp "$work/halide/generated/nijitrack_warp.a" "$sdk/libnijitrack_warp.a"
cp "$mediapipe/LICENSE" "$sdk/LICENSE.mediapipe"
cp "$halide/share/doc/Halide/LICENSE.txt" "$sdk/LICENSE.halide"
printf 'MediaPipe %s\nSource archive SHA256: %s\nRecipe SHA256: %s\nStatic face/hand + Halide AOT; no OpenCV/FFmpeg/Python.\nhttps://github.com/google-ai-edge/mediapipe\n' \
  "$(jq -r '.upstream_version' "$config")" "$(jq -r '.source_archive_sha256' "$config")" "$(jq -r '.recipe_sha256' "$config")" > "$sdk/SOURCE.txt"
for external in "$work"/bazel/*/external; do
  for repo in "$external"/*; do
    [[ -d "$repo" ]] || continue
    for license in "$repo"/LICENSE* "$repo"/COPYING* "$repo"/NOTICE*; do
      [[ -f "$license" ]] || continue
      mkdir -p "$sdk/licenses/$(basename "$repo")"
      cp "$license" "$sdk/licenses/$(basename "$repo")/"
    done
  done
done
printf 'Packaged static tracking SDK: %s\n' "$sdk"
