#!/usr/bin/env python
import os
import platform
from setuptools import setup, find_packages
import requests

def download_model(model_url, dest_path):
    """
    Downloads the model file from model_url to dest_path if it doesn't exist.
    This functionality is preserved to ensure the required model is available.
    """
    if not os.path.exists(dest_path):
        print(f"Downloading model from {model_url} to {dest_path}...")
        response = requests.get(model_url, stream=True)
        response.raise_for_status()
        with open(dest_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        print("Model download complete.")
    else:
        print("Model file already exists.")

here = os.path.abspath(os.path.dirname(__file__))

# Define the model URL and destination path.
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
MODEL_DEST = os.path.join(here, "face_landmarker_v2_with_blendshapes.task")

# Ensure the model file is downloaded.
download_model(MODEL_URL, MODEL_DEST)

# Set up OS-specific extra requirements.
# Although nijitrack.py contains OS-specific implementations (e.g., using subprocess on Linux,
# wmi on Windows, and system_profiler on Darwin), we conditionally include modules
# which are not part of the standard library.
extra_requires = {}
current_os = platform.system()
if current_os == "Windows":
    extra_requires["windows"] = ["wmi"]
elif current_os == "Linux":
    extra_requires["linux"] = []  # No additional Python modules required.
elif current_os == "Darwin":
    extra_requires["darwin"] = []  # No additional Python modules required.
else:
    extra_requires["other"] = []

setup(
    name="nijitrack",
    version="0.1.0",
    description="Real-time face tracking using MediaPipe with nijitrack",
    long_description="This package implements real-time face tracking using MediaPipe.",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/nijitrack",
    packages=find_packages(),
    install_requires=[
        "opencv-python",
        "mediapipe",
        "numpy",
        "python-osc",
        "scipy",
        "requests",
        "pynput"
    ],
    extras_require=extra_requires,
    entry_points={
        "console_scripts": [
            # Entry point for the nijitrack command.
            "nijitrack = nijitrack.nijitrack:main",
        ],
    },
    classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: BSD License",
         "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
