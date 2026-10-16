#!/usr/bin/env bash
set -euo pipefail
native=$(CDPATH= cd "$(dirname "$0")/.." && pwd -P)
appdir="$native/results/distribution-linux/nijitrack.AppDir"
tools="$native/results/appimage-tools"
[[ ! -e "$appdir" ]] || { echo "Package staging already exists: $appdir" >&2; exit 1; }
mkdir -p "$appdir/usr" "$tools" "$native/dist"
cp -a "$native/install/." "$appdir/usr/"
cp "$native/packaging/AppRun" "$appdir/"
cp "$native/packaging/nijitrack.desktop" "$native/packaging/nijitrack.svg" "$appdir/"
chmod +x "$appdir/AppRun"
while read -r hash url name; do
  [[ -n "$hash" && "$hash" != \#* ]] || continue
  if [[ ! -f "$tools/$name" ]]; then curl --fail --location --retry 3 "$url" -o "$tools/$name"; fi
  echo "$hash  $tools/$name" | sha256sum --check --status
done < "$native/packaging/appimage-tools.lock"
runtime_sources="$appdir/usr/share/nijitrack-cpp/licenses/appimage-runtime"
mkdir -p "$runtime_sources"
cp "$tools/runtime-source.tar.gz" "$tools/fuse-3.15.0.tar.xz" "$runtime_sources/"
cp "$native/packaging/LICENSE.appimage-runtime" "$native/packaging/AppImage-runtime.md" "$runtime_sources/"
tar -xOf "$tools/fuse-3.15.0.tar.xz" fuse-3.15.0/LGPL2.txt > "$runtime_sources/LGPL2.txt"
# Bundle the X11 client libraries, not the display server or hardware drivers.
# glibc, EGL/OpenGL drivers, audio servers and device services remain host APIs.
for binary in "$appdir/usr/bin/nijitrack" "$appdir/usr/bin/keyboard-probe"; do
  while read -r library _ resolved _; do
    case "$library" in
      libX*.so*|libxcb*.so*|libxkbcommon*.so*|libbsd.so*|libmd.so*)
        [[ -f "$resolved" ]] || { echo "Missing $library" >&2; exit 1; }
        cp -L "$resolved" "$appdir/usr/lib/$library"
        package=$(dpkg-query -S "$resolved" 2>/dev/null | head -n1 | cut -d: -f1 || true)
        if [[ -n "$package" && -f "/usr/share/doc/$package/copyright" ]]; then
          mkdir -p "$appdir/usr/share/nijitrack-cpp/licenses/$package"
          cp "/usr/share/doc/$package/copyright" "$appdir/usr/share/nijitrack-cpp/licenses/$package/"
        fi ;;
    esac
  done < <(ldd "$binary")
done
chmod +x "$tools/appimagetool"
export ARCH=x86_64 VERSION=0.1.0
APPIMAGE_EXTRACT_AND_RUN=1 "$tools/appimagetool" --no-appstream \
  --runtime-file "$tools/runtime-x86_64" "$appdir" \
  "$native/dist/nijitrack-native-linux-x86_64.AppImage"
