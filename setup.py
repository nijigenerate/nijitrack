#!/usr/bin/env python
import os
import platform
from setuptools import setup, find_packages
import urllib.request
import shutil

def download_model(model_url, dest_path):
    """
    Downloads the model file from model_url to dest_path if it doesn't exist,
    using only the standard library (urllib).
    """
    if not os.path.exists(dest_path):
        print(f"Downloading model from {model_url} to {dest_path}...")
        with urllib.request.urlopen(model_url) as response, open(dest_path, 'wb') as out_file:
            shutil.copyfileobj(response, out_file)
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
extra_requires = []
current_os = platform.system()
if current_os == "Windows":
    extra_requires = ["pygrabber"]
elif current_os == "Linux":
    extra_requires = []
elif current_os == "Darwin":
    extra_requires = ["pyobjc-framework-AVFoundation","pyobjc-core"]

setup(
    name="nijitrack",
    version="0.1.0",
    description="Real-time face tracking using MediaPipe with nijitrack",
    long_description="This package implements real-time face tracking using MediaPipe.",
    author="seagetch",
    author_email="seagetch@users.noreply.github.com",
    url="https://github.com/nijigenerate/nijitrack",
    packages=find_packages(),
    install_requires=[
        "opencv-python",
        "mediapipe",
        "numpy",
        "python-osc",
        "scipy",
        "pynput"
        # Removed "requests"
    ] + extra_requires,
    entry_points={
        "console_scripts": [
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
