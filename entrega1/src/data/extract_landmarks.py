import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import os
import time
from math import atan2, degrees

mp_pose = mp.solutions.pose

def angle_3d(a, b, c):
    """Calculate the angle (in degrees) between three 3D points."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

def process_video(video_path, output_csv):
    cap = cv2.VideoCapture(video_path)
    pose = mp_pose.Pose(static_image_mode=False, model_complexity=1)
    frame_count = 0
    prev_cm = None
    prev_vx, prev_vy = 0, 0
    prev_time = time.time()
    rows = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)
        if not results.pose_landmarks:
            continue

        lms = results.pose_landmarks.landmark

        # Extract required joints by name
        def get_lm(name):
            idx = mp_pose.PoseLandmark[name].value
            lm = lms[idx]
            return (lm.x, lm.y, lm.z, lm.visibility)

        nose = get_lm("NOSE")
        l_sh = get_lm("LEFT_SHOULDER")
        r_sh = get_lm("RIGHT_SHOULDER")
        l_hip = get_lm("LEFT_HIP")
        r_hip = get_lm("RIGHT_HIP")
        l_knee = get_lm("LEFT_KNEE")
        r_knee = get_lm("RIGHT_KNEE")
        l_ank = get_lm("LEFT_ANKLE")
        r_ank = get_lm("RIGHT_ANKLE")
        l_wrist = get_lm("LEFT_WRIST")
        r_wrist = get_lm("RIGHT_WRIST")

        # Derived features
        # Trunk inclination (angle between shoulders and hips)
        trunk_vec = np.array([(l_sh[0]+r_sh[0])/2 - (l_hip[0]+r_hip[0])/2,
                              (l_sh[1]+r_sh[1])/2 - (l_hip[1]+r_hip[1])/2])
        trunk_lateral_inclination = degrees(atan2(trunk_vec[0], trunk_vec[1]))

        # Center of mass (midpoint of hips)
        cm_x = (l_hip[0] + r_hip[0]) / 2
        cm_y = (l_hip[1] + r_hip[1]) / 2

        # Knee angles
        left_knee_angle = angle_3d(l_hip[:3], l_knee[:3], l_ank[:3])
        right_knee_angle = angle_3d(r_hip[:3], r_knee[:3], r_ank[:3])

        # Person height (nose to ankle distance)
        person_height = np.linalg.norm(np.array(nose[:3]) - np.array([(l_ank[0]+r_ank[0])/2, (l_ank[1]+r_ank[1])/2, (l_ank[2]+r_ank[2])/2]))

        # Center mass velocity and acceleration (finite differences)
        now = time.time()
        dt = now - prev_time if prev_time else 1e-6
        vx = (cm_x - prev_cm[0]) / dt if prev_cm else 0
        vy = (cm_y - prev_cm[1]) / dt if prev_cm else 0
        ax = (vx - prev_vx) / dt
        ay = (vy - prev_vy) / dt
        prev_time, prev_cm, prev_vx, prev_vy = now, (cm_x, cm_y), vx, vy

        # Body orientation (angle between shoulders)
        dx, dy = r_sh[0] - l_sh[0], r_sh[1] - l_sh[1]
        body_orientation = degrees(atan2(dy, dx))

        # Torso–leg ratio
        torso_len = np.linalg.norm(np.array([(l_sh[0]+r_sh[0])/2]) - np.array([(l_hip[0]+r_hip[0])/2]))
        leg_len = np.linalg.norm(np.array([(l_knee[0]+r_knee[0])/2]) - np.array([(l_ank[0]+r_ank[0])/2]))
        torso_leg_ratio = torso_len / (leg_len + 1e-6)

        timestamp = time.time()

        row = {
            "nose_x": nose[0], "nose_y": nose[1], "nose_z": nose[2], "nose_visibility": nose[3],
            "left_shoulder_x": l_sh[0], "left_shoulder_y": l_sh[1], "left_shoulder_z": l_sh[2], "left_shoulder_visibility": l_sh[3],
            "right_shoulder_x": r_sh[0], "right_shoulder_y": r_sh[1], "right_shoulder_z": r_sh[2], "right_shoulder_visibility": r_sh[3],
            "left_hip_x": l_hip[0], "left_hip_y": l_hip[1], "left_hip_z": l_hip[2], "left_hip_visibility": l_hip[3],
            "right_hip_x": r_hip[0], "right_hip_y": r_hip[1], "right_hip_z": r_hip[2], "right_hip_visibility": r_hip[3],
            "left_knee_x": l_knee[0], "left_knee_y": l_knee[1], "left_knee_z": l_knee[2], "left_knee_visibility": l_knee[3],
            "right_knee_x": r_knee[0], "right_knee_y": r_knee[1], "right_knee_z": r_knee[2], "right_knee_visibility": r_knee[3],
            "left_ankle_x": l_ank[0], "left_ankle_y": l_ank[1], "left_ankle_z": l_ank[2], "left_ankle_visibility": l_ank[3],
            "right_ankle_x": r_ank[0], "right_ankle_y": r_ank[1], "right_ankle_z": r_ank[2], "right_ankle_visibility": r_ank[3],
            "left_wrist_x": l_wrist[0], "left_wrist_y": l_wrist[1], "left_wrist_z": l_wrist[2], "left_wrist_visibility": l_wrist[3],
            "right_wrist_x": r_wrist[0], "right_wrist_y": r_wrist[1], "right_wrist_z": r_wrist[2], "right_wrist_visibility": r_wrist[3],
            "trunk_lateral_inclination": trunk_lateral_inclination,
            "center_mass_x": cm_x, "center_mass_y": cm_y,
            "left_knee_angle": left_knee_angle, "right_knee_angle": right_knee_angle,
            "person_height": person_height,
            "center_mass_velocity_x": vx, "center_mass_velocity_y": vy,
            "center_mass_acceleration_x": ax, "center_mass_acceleration_y": ay,
            "body_orientation": body_orientation,
            "torso_leg_ratio": torso_leg_ratio,
            "frame": frame_count,
            "timestamp": timestamp
        }

        rows.append(row)
        frame_count += 1

    cap.release()
    pose.close()
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Saved {output_csv}")

if __name__ == "__main__":
    input_dir = "entrega1/data/raw_videos/"
    output_dir = "entrega1/data/processed/landmarks/"
    os.makedirs(output_dir, exist_ok=True)
    for fname in os.listdir(input_dir):
        if fname.endswith(".mp4"):
            process_video(os.path.join(input_dir, fname), os.path.join(output_dir, f"{os.path.splitext(fname)[0]}.csv"))
