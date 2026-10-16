# nijitrack

nijitrack is a real-time face tracking system built on top of MediaPipe. It extracts facial landmarks, blendshapes, and head pose information, and supports Windows, Linux, and macOS. Two implementations are available: a native C++20 application using SDL3 and a statically linked MediaPipe SDK, and the original Python application using OpenCV and MediaPipe.

## Features

- **Real-Time Face Tracking:** Captures facial landmarks, blendshape scores, and head orientation (position + quaternion) in real time.
- **Cross-Platform Compatibility:** Both implementations support Windows, Linux, and macOS. The native application uses SDL3 for camera capture, device enumeration, and preview.
- **Model Availability:** Native packages include the official face and hand model bundles and use the face model by default. The Python setup script downloads its required model file (`face_landmarker_v2_with_blendshapes.task`) during installation.
- **OSC Communication:** Uses the OSC (Open Sound Control) protocol to send facial tracking data to external applications (e.g., for Virtual Motion Capture systems).
- **Flexible Command-Line Options:** Offers multiple command-line arguments to customize behavior, including device selection, video display options, and more.

## Native application

The native application does not require Python, OpenCV, or FFmpeg for setup,
build, or execution. It preserves the Python command's camera and OSC options;
video-file input (`--video`) is not supported. See the
[compatibility notes](native/COMPATIBILITY.md) for details.

### Download and run

Download a package from the [native nightly release](https://github.com/nijigenerate/nijitrack/releases/tag/native-nightly)
and extract the complete ZIP where applicable. For changes not yet published
to nightly, use the artifacts from the corresponding
[Native build run](https://github.com/nijigenerate/nijitrack/actions/workflows/native-build.yml).

| Platform | Package | Bundled model location |
|---|---|---|
| Windows x86-64 | ZIP containing `bin/nijitrack.exe` and SDL3.dll | `share/nijitrack/models` |
| macOS arm64 | ZIP containing `nijitrack.app` | `nijitrack.app/Contents/Resources/models` |
| Linux x86-64 | AppImage | `usr/share/nijitrack/models` inside the AppImage |

From the extracted Windows package directory:

```powershell
.\bin\nijitrack.exe --list-devices
.\bin\nijitrack.exe --device 0 --flip --show
```

From the directory containing the macOS application:

```bash
./nijitrack.app/Contents/MacOS/nijitrack --list-devices
./nijitrack.app/Contents/MacOS/nijitrack --device 0 --flip --show
```

On Linux:

```bash
chmod +x ./nijitrack-native-linux-x86_64.AppImage
./nijitrack-native-linux-x86_64.AppImage --list-devices
./nijitrack-native-linux-x86_64.AppImage --device 0 --flip --show
```

Models are discovered within the package; no separate model download is needed.
Use `--model FILE` to select another compatible face model, or `--help` to list
the available options. Camera-mode OSC output defaults to `127.0.0.1:39540`.

Windows requires Windows 10 version 1903 or later and statically links the MSVC
runtime. The current macOS build requires macOS 15.0 or later; its bundle is
ad-hoc signed, without Developer ID signing or notarization. Allow camera and
input-monitoring permissions as needed. The Linux build requires glibc 2.38+,
GLIBCXX_3.4.32, host graphics drivers, and camera/device services. If FUSE is
unavailable, launch with `APPIMAGE_EXTRACT_AND_RUN=1`. Linux global keyboard
capture requires X11; use `--no-keyboard` in Wayland-only sessions.

### Build from source

See [the native build instructions](native/README.md#building) for prerequisites
and the Windows PowerShell / Linux and macOS Bash commands. Builds use pinned
MediaPipe sources and patches, verify the checked-in model hashes, and run native
tests and dependency audits. CI validates the final packages and performs face
and hand inference on all three platforms.

## Python installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/nijigenerate/nijitrack.git
   cd nijitrack
   ```

2. **Install Dependencies:**

   Use `pip` to install the package:

   ```bash
   pip install .
   ```

   On Windows, additional dependencies (e.g., the `wmi` module) are managed via `extras_require`.

3. **Model Download:**

   During installation, the setup script checks if the required model file exists and downloads it automatically if necessary. Ensure that your network connection is active during the first run.

## Python usage

nijitrack is provided as a console script. After installation, run the tool using:

```bash
nijitrack --device 0 [other options]
```

### Command-Line Options

- `--list-devices`  
  Lists available video devices in JSON format.

- `--device <number>`  
  Specifies the camera device to use (e.g., 0 for the first device).

- `--osc-host` and `--osc-port`  
  Set the OSC server's host and port (default: `127.0.0.1` and port `39540`).

- `--flip`  
  Horizontally flips the video feed.

- `--show`, `--show-video`, `--show-wire`, `--show-text`, `--show-blend`  
  Toggle various display options for video output (e.g., showing the video window, wireframe overlays, text overlays, and blendshape bars).

## Development

The C++ implementation is in [`native/`](native/README.md). See its build,
packaging, and test instructions, and the [patch management guide](native/patches/README.md)
when updating the tracking SDK. Keep upstream changes in the patch series or
overlays rather than editing disposable prepared source directories.

nijitrack is developed with cross-platform compatibility in mind. Developers should:

- **Ensure Cross-Platform Support:**  
  Test on Windows, Linux, and macOS especially when modifying OS-specific code.
  
- **Manage Dependencies Appropriately:**  
  For Python, update `install_requires` and `extras_require` in `setup.py`. For native builds, update the pinned dependency metadata and CMake configuration as appropriate.
  
- **Model File Handling:**  
  Keep native model hashes, licenses, package discovery, and inference tests consistent. Check Python setup downloads separately when changing that implementation.

## License

This project is released under the 2-Clause BSD License. See the LICENSE file for more details.

Native distributions also include third-party license notices. MediaPipe and
the bundled task models use Apache-2.0; the AppImage runtime includes an LGPL-2.1
libfuse component with corresponding sources and replacement instructions.
Keep these notices and sources with redistributed packages. See
[native distribution details](native/README.md#distribution-packages).

## Contributing

Contributions are welcome! If you encounter bugs, or have suggestions, please open an issue or submit a pull request on GitHub.

## Contact

For questions or feedback, please contact the author or open an issue at https://github.com/nijigenerate/nijitrack/issues 
