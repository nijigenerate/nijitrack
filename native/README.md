# nijitrack native

The C++20 implementation uses SDL3 for camera capture and preview, and a static
MediaPipe face/hand tracking SDK with Halide AOT preprocessing. Camera and still
image inputs are supported. Setup, build, tests and execution do not require
Python, OpenCV or FFmpeg.

Windows builds the application and MediaPipe dependencies with `/MT`, without
MSVC redistributable DLLs. The package includes the SDL3 shared library. Bazel
and the Halide compiler are build tools, not application runtime dependencies.

## Replacing the Python command

The build and package contain one tracker executable: `nijitrack`
(`nijitrack.exe` on Windows). The CMake target is named `nijitrack-cpp`.

```text
nijitrack --list-devices
nijitrack --device 0 --osc-host 127.0.0.1 --osc-port 39540 --flip --show --show-video --show-wire --show-text --show-blend
```

Existing options, camera-mode OSC defaults, input mirroring before inference,
and the `v/w/t/b` preview toggles are preserved, except that `--video` file input
has been removed. See [COMPATIBILITY.md](COMPATIBILITY.md).

Camera indices are zero-based in SDL3 enumeration order; check `--list-devices`
when using several cameras. Reuse `face_landmarker_v2_with_blendshapes.task` or
specify `--model FILE`. The default search checks the executable and working
directories and their parents. Normal setup does not download models.
Paths use UTF-8 on every OS. The Windows executable embeds a UTF-8 process
manifest so command-line arguments and MediaPipe file access support Unicode
paths without changing the system locale (Windows 10 version 1903 or later).

## Building

Windows x86-64, Linux x86-64 and macOS arm64 use the same pinned MediaPipe source,
patch series, Halide implementation and static face/hand SDK. Windows uses
PowerShell; Linux and macOS use Bash and do not require PowerShell.

| OS | Prerequisites |
|---|---|
| Windows | Visual Studio 2022 C++ tools, Windows SDK, CMake/Ninja, Git for Windows, PowerShell 7 |
| Linux | Bash, Clang, CMake 3.24+, Ninja, Git, curl, jq, pkg-config, readelf, X11/Xtst, SDL camera/audio/display and EGL/GLES development libraries |
| macOS | Bash, Xcode command line tools, CMake 3.24+, Ninja, Git, curl, jq, pkg-config |

Run from the repository root. On Windows:

```powershell
& native/scripts/setup_native_toolchain.ps1
& native/scripts/prepare_mediapipe_halide.ps1
& native/scripts/build_mediapipe_halide.ps1
& native/scripts/build_native.ps1 -Install
```

On Linux and macOS:

```bash
bash native/build.sh all
```

Individual Bash stages are `setup`, `prepare`, `sdk`, `build` and `install`.
Use `all --offline` or `setup --offline` for previously downloaded assets;
Windows setup accepts `-Offline`. Bazel dependencies must also be available
locally. Downloads are checked against pinned URLs and SHA256 values.

Dependencies are placed under `vendor`, disposable MediaPipe sources and build
outputs under `build-mediapipe`, the application under `build-sdl-static`, and
installed files under `install`. See [patch management](patches/README.md): edit
the patches or `toolchain/mediapipe` overlays rather than the prepared source.
Preparation checks that source versions and C API headers match.

Always use the prepare/build scripts rather than invoking the upstream build
directly. They replace Python repository initialization with native tooling:
Starlark for LLVM overlays, and C++ byte concatenation plus CMake SHA256 for the
XNNPACK identifier. C++ guards fail accidental Python interpreter invocations.
There is no fallback to the old OpenCV SDK.

Build scripts run CTest and runtime dependency audits using LLVM on Windows,
readelf on Linux and otool on macOS. They reject OpenCV, FFmpeg, Python, shared
MediaPipe and the Halide compiler; Windows also rejects MSVC runtime DLLs.
The Makefile and compatibility build entry points delegate to these scripts.

macOS requires camera and input-monitoring permissions. Linux global keyboard
capture uses X11 RECORD; Wayland-only sessions are not supported. Use
`--no-keyboard` to disable global key capture. Hardware camera testing has been
performed on Windows; CI build and inference tests cover all three platforms.

## Capture, rendering and preprocessing

SDL3 camera frames are converted to RGB and fitted within `--width`/`--height`
(640x480 by default). `--fps` requests a camera rate, defaulting to 30. Still
images are loaded through the MediaPipe image C API. `--show-video` controls the
camera preview; there is no video-file decoder or FFmpeg dependency.

Halide AOT handles arbitrary-angle face/hand ROIs, bilinear interpolation,
boundary behavior and tensor normalization. Its compiler is not needed at
runtime. The application uses the C API's default CPU delegate. No explicit
GPU-disable flag is passed to MediaPipe, although the pinned upstream source
disables GPU support on Windows. Linux/macOS GPU execution remains untested.
CMake `WHOLE_ARCHIVE` preserves calculator registration objects.

Capture, inference/OSC and display use standard C++ threads and synchronization.
Capture and preview retain the latest frame. SDL events and rendering run on
the main thread; shutdown signals and joins the worker threads. The
`performance` log reports capture/processing FPS, frame intervals, inference
time and queue delay every five seconds and at shutdown.

Preview uses SDL lines, points, bars and debug ASCII text. Fonts and appearance
are not identical to the Python preview.

```text
nijitrack --image portrait.jpg --repeat 3 --no-osc --output results/image.jsonl
nijitrack --device 0 --frames 300 --no-keyboard --output results/camera.jsonl
```

## Tests, CI and nightly packages

```text
ctest --test-dir native/build-mediapipe/halide --output-on-failure
ctest --test-dir native/build-sdl-static --output-on-failure
```

C++ tests cover CLI/OSC behavior, RGB conversion and resizing, SDL rendering and
events, process handling and threaded shutdown. Halide tests cover rotated ROI
sampling, boundaries and normalization, using generated C++ test images.

`native-build.yml` runs the same source build on all three platforms, followed
by tests, face/hand inference, dependency audits and installed-command startup.
Tool/download caches and the static SDK cache are separate. Exact keys include
OS/CPU, runner image, compiler/system SDK, pinned dependencies, patches and SDK
build recipes. Successful stages save their caches immediately, so a later
application failure does not discard a completed SDK. The application and its
tests run on every build even when the SDK is reused.

`native-nightly.yml` builds **main** daily at 03:17 JST and updates the
`native-nightly` prerelease only after all three platform jobs succeed. Its
manual entry point is also restricted to main.

For the initial main migration, repository variable `NATIVE_CACHE_SEED_RUN_ID`
identifies a successful native-experimental build exporting the existing tools
and SDK as artifacts. Import validates the source run, exact cache keys and
archive SHA256, then saves the restored data in the **executing ref's own
cache**. Subsequent main cache hits skip the transfer entirely. No checkout of
native-experimental is required, so deleting that branch does not prevent the
initial import while the exported artifacts remain available. Transfer
artifacts are retained for seven days. Missing caches or changed SDK recipes
require a fresh build when no matching transfer is available.

CI downloads official models and images from fixed URLs with SHA256 checks into
`results/ci-fixtures`. These are not downloaded by normal setup and are not
packaged. Inference tests verify 478 face landmarks, 52 blendshapes and a
transformation matrix, plus 21 landmarks per hand, world coordinates and
handedness over three timestamps. To run with explicit fixtures:

```text
tracking-sdk-smoke FACE_MODEL HAND_MODEL FACE_IMAGE HAND_IMAGE [--gpu]
```

Distribute the licenses and attribution under `share/nijitrack-cpp`. Packages
exclude models, images and capture results. Fixed acquisition metadata is in
`toolchain.lock.json` and `toolchain.unix.lock.json`; transitive MediaPipe
dependencies are pinned by its upstream WORKSPACE/MODULE definitions.
