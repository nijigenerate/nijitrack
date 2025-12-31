import argparse
import json
import cv2
import time
import math
import re
import sys
import os
import numpy as np
from mediapipe import Image, ImageFormat
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from pythonosc.udp_client import SimpleUDPClient
from scipy.spatial.transform import Rotation as R
from face_mesh_connections import (
    FACEMESH_TESSELATION,
    FACEMESH_CONTOURS,
    FACEMESH_IRISES,
)

osc_client = None

from pynput import keyboard
from threading import Thread

# Global keybord listner
def on_key_press(key):
    try:
        osc_client.send_message("/VMC/Ext/Key",[1, key.char, ord(key.char)])
    except AttributeError:
        print(f"[SpecialKey] {key}")
    except Exception:
        pass

def on_key_release(key):
    try:
        osc_client.send_message("/VMC/Ext/Key",[0, key.char, ord(key.char)])
    except AttributeError:
        print(f"[SpecialKey] {key}")
    except Exception:
        pass

def start_keyboard_listener():
    listener = keyboard.Listener(on_press=on_key_press, on_release=on_key_release)
    listener.daemon = True
    listener.start()

FACE_CONNECTIONS = (
    FACEMESH_TESSELATION
    | FACEMESH_CONTOURS
    | FACEMESH_IRISES
)

def list_video_devices(max_devices=10):
    import platform, subprocess
    devices = []
    current_os = platform.system()
    try:
        if current_os == "Linux":
            output = subprocess.check_output(['v4l2-ctl', '--list-devices'], text=True)
            lines = output.splitlines()

            device_list = []
            current_device = None
            for line in lines:
                if line.strip() == "":
                    continue
                if not line.startswith('\t'):  # device header line
                    if current_device:
                        device_list.append(current_device)
                    current_device = {
                        "name": line.strip(),
                        "paths": []
                    }
                else:
                    path = line.strip()
                    if path.startswith('/dev/video'):
                        current_device["paths"].append(path)
            if current_device:
                device_list.append(current_device)
 
            for device in device_list:
                for path in device["paths"]:
                    try:
                        desc = subprocess.check_output(['v4l2-ctl', '--device=' + path, '--all'], text=True)
                        device_caps_match = re.search(r'Device Caps\s+:.*?\n((?:\s+.+\n)+)', desc)
                        matched = False
                        if device_caps_match:
                            caps_block = device_caps_match.group(1)
                            if 'Video Capture' in caps_block:
                                matched = True
                        if matched:
                            try:
                                id_int = int(path.replace("/dev/video", ""))
                                devices.append({"id": id_int, "name": device["name"], "path": path})
                                break 
                            except Exception:
                                pass
                    except subprocess.CalledProcessError:
                        continue

        elif current_os == "Windows":
            from pygrabber.dshow_graph import FilterGraph

            graph = FilterGraph()
            device_names = graph.get_input_devices()
            devices = [{"id": i, "name": name} for i, name in enumerate(device_names)]

        elif current_os == "Darwin":
            import AVFoundation
            deviceList = AVFoundation.AVCaptureDevice.devicesWithMediaType_(AVFoundation.AVMediaTypeVideo)
            deviceList = [d for d in deviceList]
            deviceList = sorted(deviceList, key=lambda d: d.uniqueID())
            for i, device in enumerate(deviceList):
                name = device.localizedName()
                unique_id = device.uniqueID()
                devices.append({"id": i, "name": name, "uuid": device.uniqueID()})

    except Exception as e:
        import traceback
        traceback.print_exc()
        for i in range(max_devices):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                devices.append({"id": i, "name": f"Device {i}"})
                cap.release()
    return devices

def normalize(v):
    norm = np.linalg.norm(v)
    return v / norm if norm > 1e-6 else v

def compute_fixed_rotation_matrix(landmarks):
    LEFT_EYE_OUTER = 33
    RIGHT_EYE_OUTER = 263
    CHIN = 152
    NOSE_TIP = 1

    left_eye = landmarks[LEFT_EYE_OUTER]
    right_eye = landmarks[RIGHT_EYE_OUTER]
    chin = landmarks[CHIN]

    z_axis = normalize(np.cross(right_eye - left_eye, chin - left_eye))
    x_axis = normalize(right_eye - left_eye)
    y_axis = normalize(np.cross(z_axis, x_axis))
    x_axis = normalize(np.cross(y_axis, z_axis))

    R_mat = np.stack([x_axis, y_axis, z_axis], axis=1)
    return R_mat, landmarks[NOSE_TIP]

def rotation_matrix_to_euler_zxy(R_mat):
    pitch = math.asin(-R_mat[1, 2])
    yaw = math.atan2(R_mat[0, 2], R_mat[2, 2])
    roll = math.atan2(R_mat[1, 0], R_mat[1, 1])
    return math.degrees(yaw), math.degrees(pitch), math.degrees(roll)

def rotation_matrix_to_quaternion(R_mat):
    quat = R.from_matrix(R_mat).as_quat()
    return quat[0], quat[1], quat[2], quat[3]

def get_gaze_offset(center, outer, inner, top, bottom):
    if (inner.x - outer.x) != 0:
        gaze_x = (center.x - outer.x) / (inner.x - outer.x) * 2 - 1
    else:
        gaze_x = 0
    middle_y = (top.y + bottom.y) / 2
    half_height = (bottom.y - top.y) / 2
    if half_height != 0:
        gaze_y = (center.y - middle_y) / half_height
    else:
        gaze_y = 0
    return np.clip(gaze_x, -1, 1), np.clip(gaze_y, -1, 1)

def get_gaze_right(landmarks):
    return get_gaze_offset(landmarks[468], landmarks[33], landmarks[133], landmarks[159], landmarks[145])

def get_gaze_left(landmarks):
    return get_gaze_offset(landmarks[473], landmarks[362], landmarks[263], landmarks[386], landmarks[374])

def render_frame(frame, width, height, landmarks, blendshapes, yaw, pitch, roll, nose, right_gaze, left_gaze, show_video, show_wire, show_text, show_blend):
    display = frame.copy() if show_video else np.zeros_like(frame)

    if show_wire and landmarks:
        for connection in FACE_CONNECTIONS:
            start_idx, end_idx = connection
            pt1 = landmarks[start_idx]
            pt2 = landmarks[end_idx]
            x1, y1 = int(pt1.x * width), int(pt1.y * height)
            x2, y2 = int(pt2.x * width), int(pt2.y * height)
            cv2.line(display, (x1, y1), (x2, y2), (0, 255, 0), 1)
        for lm in landmarks:
            cx, cy = int(lm.x * width), int(lm.y * height)
            cv2.circle(display, (cx, cy), 1, (0, 0, 255), -1)

    if show_text:
        cv2.putText(display, f"Yaw: {yaw:.1f}  Pitch: {pitch:.1f}  Roll: {roll:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
        cv2.putText(display, f"Pos X: {nose[0]:.3f}  Y: {nose[1]:.3f}  Z: {nose[2]:.3f}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
        cv2.putText(display, f"Gaze R: x={right_gaze[0]:+.2f}  y={right_gaze[1]:+.2f}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
        cv2.putText(display, f"Gaze L: x={left_gaze[0]:+.2f}  y={left_gaze[1]:+.2f}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)

    if show_blend and blendshapes:
        y = 120
        bar_x = display.shape[1] - 160
        for bs in blendshapes:
            bar_len = int(bs.score * 150)
            cv2.rectangle(display, (bar_x, y), (bar_x + bar_len, y + 6), (0, 255, 255), -1)
            cv2.putText(display, f"{bs.category_name}: {bs.score:.2f}", (bar_x - 200, y + 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            y += 10

    return display

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--list-devices", action="store_true")
    parser.add_argument("--device", type=int)
    parser.add_argument("--osc-host", type=str, default="127.0.0.1")
    parser.add_argument("--osc-port", type=int, default=39540)
    parser.add_argument("--flip", action="store_true")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--show-video", action="store_true")
    parser.add_argument("--show-wire", action="store_true")
    parser.add_argument("--show-text", action="store_true")
    parser.add_argument("--show-blend", action="store_true")
    args = parser.parse_args()

    if args.list_devices:
        print(json.dumps(list_video_devices(), indent=2, ensure_ascii=False))
        sys.exit(0)

    if args.device is None:
        print("Error: Please specify device with --device.")
        sys.exit(1)

    osc_client = SimpleUDPClient(args.osc_host, args.osc_port)
    start_keyboard_listener()

    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "face_landmarker_v2_with_blendshapes.task")
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=True,
        running_mode=vision.RunningMode.VIDEO
    )
    landmarker = vision.FaceLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(args.device)
    if not cap.isOpened():
        print(f"Error: Cannot open camera device {args.device}.")
        sys.exit(2)

    show_video = args.show_video
    show_wire = args.show_wire
    show_text = args.show_text
    show_blend = args.show_blend

    if args.show:
        cv2.namedWindow("MediaPipe Face", cv2.WINDOW_NORMAL | cv2.WINDOW_AUTOSIZE)

    while True:
        if args.show:
            try:
                if cv2.getWindowProperty("MediaPipe Face", cv2.WND_PROP_VISIBLE) < 1:
                    break
            except cv2.error:
                break

        ret, frame = cap.read()
        if not ret:
            sys.exit(-1)

        if args.flip:
            frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = Image(image_format=ImageFormat.SRGB, data=rgb)
        timestamp = int(time.time() * 1000)
        result = landmarker.detect_for_video(mp_image, timestamp)

        landmarks = None
        yaw, pitch, roll, nose = (None,None,None,None)
        right_gaze, left_gaze = (None, None)

        if result.face_landmarks:
            landmarks = result.face_landmarks[0]
            pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
            R_mat, nose = compute_fixed_rotation_matrix(pts)
            quat = rotation_matrix_to_quaternion(R_mat)
            quat = normalize(np.array(quat))
            yaw, pitch, roll = rotation_matrix_to_euler_zxy(R_mat)
            right_gaze = get_gaze_right(landmarks)
            left_gaze = get_gaze_left(landmarks)

            osc_client.send_message("/VMC/Ext/Bone/Pos", [
                "Head",
                nose[0], nose[1], nose[2],
                quat[0], quat[1], quat[2], quat[3]
            ])

            osc_client.send_message("/VMC/Ext/Blend/Val", ["GazeRightX", right_gaze[0]])
            osc_client.send_message("/VMC/Ext/Blend/Val", ["GazeRightY", right_gaze[1]])
            osc_client.send_message("/VMC/Ext/Blend/Val", ["GazeLeftX", left_gaze[0]])
            osc_client.send_message("/VMC/Ext/Blend/Val", ["GazeLeftY", left_gaze[1]])

            if result.face_blendshapes:
                blendshapes = result.face_blendshapes[0]
                for bs in blendshapes:
                    osc_client.send_message("/VMC/Ext/Blend/Val", [bs.category_name, bs.score])
                osc_client.send_message("/VMC/Ext/Blend/Apply", [])

        if args.show:
            height, width = frame.shape[:2]
            display = render_frame(
                frame, width, height, landmarks,
                blendshapes if result.face_blendshapes else [],
                yaw, pitch, roll, nose,
                right_gaze, left_gaze,
                show_video, show_wire, show_text, show_blend
            )
            cv2.imshow("MediaPipe Face", display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('v'):
                show_video = not show_video
            elif key == ord('w'):
                show_wire = not show_wire
            elif key == ord('t'):
                show_text = not show_text
            elif key == ord('b'):
                show_blend = not show_blend

    cap.release()
    if args.show:
        cv2.destroyAllWindows()
    sys.exit(0)
