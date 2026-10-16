#!/usr/bin/env bash
# Linux/macOS entry point; run with: bash native/build.sh all
set -euo pipefail
native_root=$(CDPATH= cd "$(dirname "$0")" && pwd -P)
action=${1:-all}
offline=OFF
[[ ${2:-} != --offline ]] || offline=ON
export NIJITRACK_OFFLINE=$offline
run() { bash "$native_root/scripts/$1.sh" "${@:2}"; }
case "$action" in
  setup) run setup_tracking_unix "$offline" ;;
  prepare) run prepare_mediapipe_halide ;;
  sdk) run build_mediapipe_halide ;;
  build) run build_native ;;
  install) run build_native --install ;;
  all)
    run setup_tracking_unix "$offline"
    run prepare_mediapipe_halide
    run build_mediapipe_halide
    run build_native --install ;;
  *) printf '%s\n' 'Usage: bash native/build.sh {setup|prepare|sdk|build|install|all} [--offline]' >&2; exit 2 ;;
esac
