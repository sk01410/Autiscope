import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import joblib
import os

# --- Paths ---
test_video_path = 'dataset/test_videos/test1.mp4'
test_features_csv = 'features/test/test1_features.csv'

# --- Step 1: Extract features from test video (same as Step 3) ---
mp_face = mp.solutions.face_mesh
mp_pose = mp.solutions.pose

cap = cv2.VideoCapture(test_video_path)
face_mesh = mp_face.FaceMesh(static_image_mode=False, max_num_faces=1)
pose = mp_pose.Pose(static_image_mode=False)

all_features = []
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_results = face_mesh.process(frame_rgb)
    pose_results = pose.process(frame_rgb)

    frame_features = []
    if face_results.multi_face_landmarks:
        for landmark in face_results.multi_face_landmarks[0].landmark:
            frame_features.append(landmark.x)
            frame_features.append(landmark.y)
    else:
        frame_features.extend([0.0] * 468 * 2)

    if pose_results.pose_landmarks:
        for landmark in pose_results.pose_landmarks.landmark:
            frame_features.append(landmark.x)
            frame_features.append(landmark.y)
    else:
        frame_features.extend([0.0] * 33 * 2)

    all_features.append(frame_features)

df = pd.DataFrame(all_features)
os.makedirs(os.path.dirname(test_features_csv), exist_ok=True)
df.to_csv(test_features_csv, index=False)

print("✅ Test video features extracted.")

# --- Step 2: Load trained models ---
eye_model = joblib.load('models/eye_contact_model.pkl')
rep_model = joblib.load('models/repetitive_motion_model.pkl')
soc_model = joblib.load('models/social_deficit_model.pkl')

# --- Step 3: Predict using each model ---
features = pd.read_csv(test_features_csv)

# Use average per column as summary statistic
mean_features = features.mean(axis=0).values.reshape(1, -1)

eye_flag = eye_model.predict(mean_features)[0]
rep_flag = rep_model.predict(mean_features)[0]
soc_flag = soc_model.predict(mean_features)[0]

# --- Step 4: Print results ---
print("\n🎯 Detection Results:")
print(f"👁️  Eye Contact Issue:       {'YES' if eye_flag else 'No'}")
print(f"🔁  Repetitive Motion:       {'YES' if rep_flag else 'No'}")
print(f"🧍  Social Deficit:          {'YES' if soc_flag else 'No'}")

# Optional: Combine as autism detection signal
if eye_flag or rep_flag or soc_flag:
    print("\n⚠️  Autism-related behaviors detected!")
else:
    print("\n✅ No autism-related behavior detected.")
