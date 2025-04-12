# nijitrack

nijitrack is a real-time face tracking system built on top of MediaPipe. It leverages OpenCV and MediaPipe to extract facial landmarks, blendshapes, and head pose information, and supports cross-platform usage on Windows, Linux, and macOS.

## Features

- **Real-Time Face Tracking:** Captures facial landmarks, blendshape scores, and head orientation (position + quaternion) in real time.
- **Cross-Platform Compatibility:** Designed to work on Windows, Linux, and macOS. The system includes OS-specific implementations for retrieving detailed camera device information using methods such as WMI on Windows, v4l2-ctl on Linux, and system_profiler on macOS.
- **Automatic Model Download:** The required model file (`face_landmarker_v2_with_blendshapes.task`) is automatically downloaded during installation via the setup script.
- **OSC Communication:** Uses the OSC (Open Sound Control) protocol to send facial tracking data to external applications (e.g., for Virtual Motion Capture systems).
- **Flexible Command-Line Options:** Offers multiple command-line arguments to customize behavior, including device selection, video display options, and more.

## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/yourusername/nijitrack.git
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

## Usage

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

nijitrack is developed with cross-platform compatibility in mind. Developers should:

- **Ensure Cross-Platform Support:**  
  Test on Windows, Linux, and macOS especially when modifying OS-specific code.
  
- **Manage Dependencies Appropriately:**  
  Update `install_requires` and `extras_require` in `setup.py` to avoid unnecessary dependencies.
  
- **Model File Handling:**  
  Make sure that the model file handling in the setup process works as expected in different network environments.

## License

This project is released under the 2-Clause BSD License. See the LICENSE file for more details.

## Contributing

Contributions are welcome! If you encounter bugs, or have suggestions, please open an issue or submit a pull request on GitHub.

## Contact

For questions or feedback, please contact the author or open an issue in the repository.
