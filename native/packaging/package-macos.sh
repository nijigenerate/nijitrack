#!/usr/bin/env bash
set -euo pipefail
native=$(CDPATH= cd "$(dirname "$0")/.." && pwd -P)
stage="$native/results/distribution-macos"
app="$stage/nijitrack.app"
[[ ! -e "$app" ]] || { echo "Package staging already exists: $app" >&2; exit 1; }
mkdir -p "$app/Contents/"{MacOS,Frameworks,Resources} "$native/dist"
cp "$native/install/bin/nijitrack" "$app/Contents/MacOS/"
cp "$native/install/bin/keyboard-probe" "$app/Contents/MacOS/"
cp "$native/install/lib/libSDL3.0.dylib" "$app/Contents/Frameworks/"
cp -R "$native/install/share/nijitrack/models" "$app/Contents/Resources/"
cp -R "$native/install/share/nijitrack-cpp" "$app/Contents/Resources/licenses"
cp "$native/packaging/Info.plist" "$app/Contents/Info.plist"
plutil -lint "$app/Contents/Info.plist"
install_name_tool -rpath '@loader_path/../lib' '@loader_path/../Frameworks' "$app/Contents/MacOS/nijitrack"
# Ad-hoc signing preserves local integrity on Apple Silicon. This is not
# Developer ID signing or notarization; no signing credentials are assumed.
codesign --force --sign - "$app/Contents/Frameworks/libSDL3.0.dylib"
codesign --force --sign - "$app/Contents/MacOS/keyboard-probe"
codesign --force --sign - "$app"
codesign --verify --deep --strict --verbose=2 "$app"
ditto -c -k --keepParent "$app" "$native/dist/nijitrack-native-macos-arm64.zip"
