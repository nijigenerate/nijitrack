#!/usr/bin/env bash
set -euo pipefail
. "$(dirname "$0")/tracking_unix.sh"
directory=${1:?runtime directory required}
[[ -d "$directory" ]] || die "Missing directory: $directory"
scan=("$directory")
if [[ $(basename "$directory") == bin && -d "$directory/../lib" ]]; then scan+=("$directory/../lib"); fi
report="$native_root/results/build-environment/runtime-dependencies.txt"
mkdir -p "$(dirname "$report")"
: > "$report"
count=0
for root in "${scan[@]}"; do
  for file in "$root"/*; do
    [[ -f "$file" ]] || continue
    magic=$(od -An -tx1 -N4 "$file" | tr -d ' \n')
    case "$platform:$magic" in
      linux-*:7f454c46) dependencies=$(readelf -d "$file" | sed -n 's/.*(NEEDED).*\[\([^]]*\)\].*/\1/p') ;;
      macos-*:cffaedfe|macos-*:cefaedfe|macos-*:feedfacf|macos-*:feedface|macos-*:cafebabe|macos-*:bebafeca|macos-*:cafebabf|macos-*:bfbafeca)
        dependencies=$(otool -L "$file" | sed -n 's/^[[:space:]]*\(.*\) (compatibility version.*/\1/p') ;;
      *) continue ;;
    esac
    while IFS= read -r dependency; do
      name=$(basename "$dependency" | tr '[:upper:]' '[:lower:]')
      case "$name" in
        *opencv*|*ffmpeg*|*ffprobe*|libavcodec.*|libavdevice.*|libavfilter.*|libavformat.*|libavutil.*|libswresample.*|libswscale.*|libmediapipe*|libpython*|libhalide.*)
          die "Forbidden runtime dependency in $file: $dependency" ;;
      esac
    done < <(printf '%s\n%s\n' "$file" "$dependencies")
    printf '%s: %s\n' "$(basename "$file")" "$dependencies" >> "$report"
    count=$((count+1))
  done
done
[[ $count -gt 0 ]] || die "No runtime binaries inspected: $directory"
printf 'Verified %s runtime binaries: no OpenCV/FFmpeg/Python/shared MediaPipe/Halide compiler\n' "$count"
