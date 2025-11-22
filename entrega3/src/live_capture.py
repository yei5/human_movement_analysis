import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import joblib
import time
from math import atan2, degrees
from collections import deque, Counter

# ==== RUTAS DEL MODELO ====
MODEL_PATH = "entrega2/experiments/models/RandomForest_model.pkl"
SCALER_PATH = "entrega2/experiments/models/scaler.pkl"

# ==== CARGAR MODELO ====
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# ==== MEDIAPIPE ====
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(
    static_image_mode=False,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.75
)

# ==== VARIABLES GLOBALES ====
prev_time = 0
prev_cm = None
prev_vx = 0
prev_vy = 0

# ==== BUFFER PARA SUAVIZADO ====
WINDOW_SIZE = 7
pred_buffer = deque(maxlen=WINDOW_SIZE)
conf_buffer = deque(maxlen=WINDOW_SIZE)


def angle_3d(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba)*np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))


def compute_features(landmarks):
    global prev_time, prev_cm, prev_vx, prev_vy

    lm = landmarks.landmark

    # ==== LANDMARKS ====
    nose = [lm[0].x, lm[0].y, lm[0].z, lm[0].visibility]
    l_sh = [lm[11].x, lm[11].y, lm[11].z, lm[11].visibility]
    r_sh = [lm[12].x, lm[12].y, lm[12].z, lm[12].visibility]
    l_hip = [lm[23].x, lm[23].y, lm[23].z, lm[23].visibility]
    r_hip = [lm[24].x, lm[24].y, lm[24].z, lm[24].visibility]
    l_knee = [lm[25].x, lm[25].y, lm[25].z, lm[25].visibility]
    r_knee = [lm[26].x, lm[26].y, lm[26].z, lm[26].visibility]
    l_ank = [lm[27].x, lm[27].y, lm[27].z, lm[27].visibility]
    r_ank = [lm[28].x, lm[28].y, lm[28].z, lm[28].visibility]
    l_wrist = [lm[15].x, lm[15].y, lm[15].z, lm[15].visibility]
    r_wrist = [lm[16].x, lm[16].y, lm[16].z, lm[16].visibility]

    # ==== CENTER OF MASS ====
    cm_x = (l_hip[0] + r_hip[0]) / 2
    cm_y = (l_hip[1] + r_hip[1]) / 2

    # ==== VELOCIDAD / ACELERACIÓN ====
    now = time.time()
    dt = now - prev_time if prev_time else 1e-6

    vx = (cm_x - prev_cm[0]) / dt if prev_cm else 0
    vy = (cm_y - prev_cm[1]) / dt if prev_cm else 0
    ax = (vx - prev_vx) / dt
    ay = (vy - prev_vy) / dt

    prev_cm = (cm_x, cm_y)
    prev_time = now
    prev_vx = vx
    prev_vy = vy

    # ==== DERIVED FEATURES ====
    left_knee_angle = angle_3d(l_hip[:3], l_knee[:3], l_ank[:3])
    right_knee_angle = angle_3d(r_hip[:3], r_knee[:3], r_ank[:3])

    trunk_vec = np.array([(l_sh[0]+r_sh[0])/2 - (l_hip[0]+r_hip[0])/2,
                          (l_sh[1]+r_sh[1])/2 - (l_hip[1]+r_hip[1])/2])
    trunk_lateral_inclination = degrees(atan2(trunk_vec[0], trunk_vec[1]))

    body_orientation = degrees(atan2(
        r_sh[1] - l_sh[1],
        r_sh[0] - l_sh[0]
    ))

    person_height = np.linalg.norm(
        np.array(nose[:3]) - (np.array(l_ank[:3]) + np.array(r_ank[:3])) / 2
    )

    torso_len = np.linalg.norm(np.array([(l_sh[0]+r_sh[0])/2]) -
                               np.array([(l_hip[0]+r_hip[0])/2]))
    leg_len = np.linalg.norm(np.array([(l_knee[0]+r_knee[0])/2]) -
                             np.array([(l_ank[0]+r_ank[0])/2]))
    torso_leg_ratio = torso_len / (leg_len + 1e-6)

    features = [
        nose[0], nose[1], nose[2], nose[3],
        l_sh[0], l_sh[2], l_sh[3],
        r_sh[0], r_sh[2],
        l_hip[1], l_hip[2], l_hip[3],
        r_hip[2],
        l_knee[0], l_knee[2], l_knee[3],
        r_knee[0], r_knee[2], r_knee[3],
        l_ank[2], l_ank[3],
        r_ank[2], r_ank[3],
        l_wrist[1], l_wrist[2], l_wrist[3],
        r_wrist[0], r_wrist[1], r_wrist[2], r_wrist[3],
        trunk_lateral_inclination,
        cm_x,
        left_knee_angle,
        right_knee_angle,
        person_height,
        vx, vy, ax, ay, body_orientation, torso_leg_ratio
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

    return pd.DataFrame([features], columns=feature_names)


# ==== ACTIVIDADES ====
activities = {
    0: "Sit Down",
    1: "Stand Up",
    2: "Turn Around",
    3: "Walk Back",
    4: "Walk Forward"
}

# ==== LIVE CAPTURE ====
cap = cv2.VideoCapture(0)
print("Live recognition running... press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)

    if results.pose_landmarks:

        X = compute_features(results.pose_landmarks)
        X_scaled = scaler.transform(X)

        # === predicción instantánea ===
        probs = model.predict_proba(X_scaled)[0]
        pred = np.argmax(probs)
        conf = probs[pred] * 100

        # === añadir a buffer ===
        pred_buffer.append(pred)
        conf_buffer.append(conf)

        # === si la ventana está llena, aplicar votación ===
        if len(pred_buffer) == WINDOW_SIZE:
            most_common = Counter(pred_buffer).most_common(1)[0][0]
            final_pred = most_common
            final_conf = np.mean([c for p, c in zip(pred_buffer, conf_buffer) if p == most_common])
        else:
            final_pred = pred
            final_conf = conf

        label = f"{activities[final_pred]} ({final_conf:.1f}%)"

        cv2.putText(frame, label, (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 3)

        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    cv2.imshow("HAR Live (Smoothed)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
pose.close()
cv2.destroyAllWindows()
print("Done.")
