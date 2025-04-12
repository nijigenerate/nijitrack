from setuptools import setup, find_packages
import os
import urllib.request

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
MODEL_FILENAME = "face_landmarker_v2_with_blendshapes.task"

def download_model():
    if not os.path.exists(MODEL_FILENAME):
        print(f"Downloading {MODEL_FILENAME} ...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_FILENAME)
        print(f"Downloaded {MODEL_FILENAME}")
    else:
        print(f"Model file '{MODEL_FILENAME}' already exists.")

# Run model download before setup
download_model()

setup(
    name="nijitrack",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "mediapipe>=0.10.9",
        "numpy",
        "opencv-python",
        "python-osc",
        "scipy"
    ],
    entry_points={
        "console_scripts": [
            "nijitrack=nijitrack:main"
        ]
    },
    python_requires=">=3.8",
)