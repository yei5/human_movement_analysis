import cv2
import mediapipe as mp
import numpy as np
import joblib
import math
import pandas as pd
import time
import warnings

# ==== CONFIG ====
print("starting")
MODEL_PATH = "entrega2/experiments/models/RandomForest_model.pkl"
ENCODER_PATH = "entrega2/experiments/models/scaler.pkl"
CONFIDENCE_THRESHOLD = 0.6

# Ignore sklearn feature name warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names")

# ==== LOAD MODEL ====
scaler = joblib.load(ENCODER_PATH)
model = joblib.load(MODEL_PATH)
print("✅ Model loaded successfully.")

# ==== MEDIAPIPE POSE ====
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.6, min_tracking_confidence=0.6)

# ==== FEATURE CALCULATIONS ====

def calc_angle(a, b, c):
    """Compute the angle between three points."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))


def compute_features(landmarks):
    """Extracts model features from mediapipe landmarks."""
    lm = landmarks.landmark

    # Extract key joints
    nose = [lm[0].x, lm[0].y, lm[0].z, lm[0].visibility]
    l_shoulder = [lm[11].x, lm[11].y, lm[11].z, lm[11].visibility]
    r_shoulder = [lm[12].x, lm[12].y, lm[12].z, lm[12].visibility]
    l_hip = [lm[23].x, lm[23].y, lm[23].z, lm[23].visibility]
    r_hip = [lm[24].x, lm[24].y, lm[24].z, lm[24].visibility]
    l_knee = [lm[25].x, lm[25].y, lm[25].z, lm[25].visibility]
    r_knee = [lm[26].x, lm[26].y, lm[26].z, lm[26].visibility]
    l_ankle = [lm[27].x, lm[27].y, lm[27].z, lm[27].visibility]
    r_ankle = [lm[28].x, lm[28].y, lm[28].z, lm[28].visibility]
    l_wrist = [lm[15].x, lm[15].y, lm[15].z, lm[15].visibility]
    r_wrist = [lm[16].x, lm[16].y, lm[16].z, lm[16].visibility]

    # Derived biomechanical features
    left_knee_angle = calc_angle(l_hip[:3], l_knee[:3], l_ankle[:3])
    right_knee_angle = calc_angle(r_hip[:3], r_knee[:3], r_ankle[:3])

    center_mass_x = (l_hip[0] + r_hip[0]) / 2
    center_mass_y = (l_hip[1] + r_hip[1]) / 2

    trunk_lateral_inclination = math.degrees(math.atan2(
        (r_shoulder[1] - l_shoulder[1]),
        (r_shoulder[0] - l_shoulder[0])
    ))

    # Height (nose to midpoint between ankles)
    person_height = np.linalg.norm(
        np.array(nose[:3]) - ((np.array(l_ankle[:3]) + np.array(r_ankle[:3])) / 2)
    )

    # Torso-leg ratio
    torso_len = np.linalg.norm(
        np.array([(l_shoulder[0]+r_shoulder[0])/2,
                  (l_shoulder[1]+r_shoulder[1])/2,
                  (l_shoulder[2]+r_shoulder[2])/2]) -
        np.array([(l_hip[0]+r_hip[0])/2,
                  (l_hip[1]+r_hip[1])/2,
                  (l_hip[2]+r_hip[2])/2])
    )
    leg_len = np.linalg.norm(
        np.array([(l_knee[0]+r_knee[0])/2,
                  (l_knee[1]+r_knee[1])/2,
                  (l_knee[2]+r_knee[2])/2]) -
        np.array([(l_ankle[0]+r_ankle[0])/2,
                  (l_ankle[1]+r_ankle[1])/2,
                  (l_ankle[2]+r_ankle[2])/2])
    )
    torso_leg_ratio = torso_len / (leg_len + 1e-6)

    # Dummy values for live prediction
    center_mass_velocity_x = 0
    center_mass_velocity_y = 0
    center_mass_acceleration_x = 0
    center_mass_acceleration_y = 0
    body_orientation = 0

    # Build feature vector (same order as training)
    features = [
        nose[0], nose[1], nose[2], nose[3],
        l_shoulder[0], l_shoulder[2], l_shoulder[3],
        r_shoulder[0], r_shoulder[2],
        l_hip[1], l_hip[2], l_hip[3],
        r_hip[2],
        l_knee[0], l_knee[2], l_knee[3],
        r_knee[0], r_knee[2], r_knee[3],
        l_ankle[2], l_ankle[3],
        r_ankle[2], r_ankle[3],
        l_wrist[1], l_wrist[2], l_wrist[3],
        r_wrist[0], r_wrist[1], r_wrist[2], r_wrist[3],
        trunk_lateral_inclination,
        center_mass_x,
        left_knee_angle,
        right_knee_angle,
        person_height,
        center_mass_velocity_x,
        center_mass_velocity_y,
        center_mass_acceleration_x,
        center_mass_acceleration_y,
        body_orientation,
        torso_leg_ratio
    ]

    feature_names = [
        "nose_x","nose_y","nose_z","nose_visibility",
        "left_shoulder_x","left_shoulder_z","left_shoulder_visibility",
        "right_shoulder_x","right_shoulder_z",
        "left_hip_y","left_hip_z","left_hip_visibility",
        "right_hip_z",
        "left_knee_x","left_knee_z","left_knee_visibility",
        "right_knee_x","right_knee_z","right_knee_visibility",
        "left_ankle_z","left_ankle_visibility",
        "right_ankle_z","right_ankle_visibility",
        "left_wrist_y","left_wrist_z","left_wrist_visibility",
        "right_wrist_x","right_wrist_y","right_wrist_z","right_wrist_visibility",
        "trunk_lateral_inclination","center_mass_x",
        "left_knee_angle","right_knee_angle","person_height",
        "center_mass_velocity_x","center_mass_velocity_y",
        "center_mass_acceleration_x","center_mass_acceleration_y",
        "body_orientation","torso_leg_ratio"
    ]

    df = pd.DataFrame([features], columns=feature_names)
    return df


# ==== LIVE CAPTURE ====
activities = {
    0: "Sit Down",
    1: "Stand Up",
    2: "Turn Around",
    3: "Walk Back",
    4: "Walk Forward"
}

cap = cv2.VideoCapture(0)
prev_time = 0

print("🎥 Starting live activity recognition (press 'q' to quit).")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if results.pose_landmarks:
        X_live = compute_features(results.pose_landmarks)
        X_live_scaled = scaler.transform(X_live)
        prediction = model.predict(X_live_scaled)[0]
        proba = np.max(model.predict_proba(X_live_scaled))

        label = activities.get(prediction, "Unknown")

        if proba >= CONFIDENCE_THRESHOLD:
            text = f"{label} ({proba:.2f})"
            color = (0, 255, 0)
        else:
            text = "Uncertain"
            color = (0, 0, 255)

        cv2.putText(frame, text, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # FPS display
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
    prev_time = curr_time
    cv2.putText(frame, f"FPS: {int(fps)}", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow("Human Activity Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
pose.close()
cv2.destroyAllWindows()
print("🔚 Finished.")
