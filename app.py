import streamlit as st
import os
import json
import pandas as pd
import cv2
import numpy as np
import joblib
import tempfile
from pathlib import Path
import mediapipe as mp

st.set_page_config(page_title="Autism Behavior Detector", layout="centered")
st.title("🧠 Autism Behavior Detection from Video")

st.markdown("Upload a video to detect early autism-related behavioral patterns (eye contact, repetitive motion, social deficit).")

# Upload section
uploaded_video = st.file_uploader("📤 Upload a video", type=["mp4", "mov", "avi"])
if uploaded_video:
    # Save the uploaded video temporarily
    tmp_dir = tempfile.mkdtemp()
    video_path = os.path.join(tmp_dir, uploaded_video.name)
    with open(video_path, 'wb') as f:
        f.write(uploaded_video.read())
    st.success("✅ Video uploaded!")

    # Button to start processing
    if st.button("🔍 Analyze Video"):
        with st.spinner("Extracting features and analyzing behavior..."):
            # --- Feature Extraction ---
            mp_face = mp.solutions.face_mesh
            mp_pose = mp.solutions.pose
            cap = cv2.VideoCapture(video_path)
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

            features_df = pd.DataFrame(all_features)

            # --- Load Models ---
            rep_model = joblib.load('models/repetitive_motion_model.pkl')
            soc_model = joblib.load('models/social_deficit_model.pkl')

            # --- Predict ---
            mean_features = features_df.mean(axis=0).values.reshape(1, -1)
            rep_flag = rep_model.predict(mean_features)[0]
            soc_flag = soc_model.predict(mean_features)[0]

            # --- Report ---
            report = {
                "repetitive_motion": bool(rep_flag),
                "social_deficit": bool(soc_flag)
            }

            risk_count = sum([rep_flag, soc_flag])
            if risk_count == 0:
                report["autism_risk_level"] = "Low"
            elif risk_count == 1:
                report["autism_risk_level"] = "Mild"
            elif risk_count == 2:
                report["autism_risk_level"] = "Moderate"
            else:
                report["autism_risk_level"] = "High"

            # Save report
            report_path = os.path.join("reports", Path(uploaded_video.name).stem + "_report.json")
            os.makedirs(os.path.dirname(report_path), exist_ok=True)
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=4)

        # --- Display Results ---
        st.subheader("📄 Report")
        st.json(report)

        st.success("✅ Analysis complete!")
