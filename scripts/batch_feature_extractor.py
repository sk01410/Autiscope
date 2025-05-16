import os
import cv2
import mediapipe as mp
import pandas as pd

# Initialize MediaPipe
mp_face = mp.solutions.face_mesh
mp_pose = mp.solutions.pose

# Paths
dataset_dir = 'dataset'
output_dir = 'features'

# Create output folders
os.makedirs(output_dir, exist_ok=True)

# Initialize MediaPipe models once
face_mesh = mp_face.FaceMesh(static_image_mode=False, max_num_faces=1)
pose = mp_pose.Pose(static_image_mode=False)

# Loop through each category
for behavior_folder in os.listdir(dataset_dir):
    input_folder = os.path.join(dataset_dir, behavior_folder)
    output_folder = os.path.join(output_dir, behavior_folder)
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if not filename.endswith(".mp4"):
            continue

        video_path = os.path.join(input_folder, filename)
        output_csv = os.path.join(output_folder, filename.replace('.mp4', '_features.csv'))

        print(f"⏳ Processing: {video_path}")

        cap = cv2.VideoCapture(video_path)
        all_features = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_results = face_mesh.process(frame_rgb)
            pose_results = pose.process(frame_rgb)

            frame_features = []

            # Face landmarks
            if face_results.multi_face_landmarks:
                for landmark in face_results.multi_face_landmarks[0].landmark:
                    frame_features.extend([landmark.x, landmark.y])
            else:
                frame_features.extend([0.0] * 468 * 2)

            # Pose landmarks
            if pose_results.pose_landmarks:
                for landmark in pose_results.pose_landmarks.landmark:
                    frame_features.extend([landmark.x, landmark.y])
            else:
                frame_features.extend([0.0] * 33 * 2)

            all_features.append(frame_features)

        cap.release()

        # Save to CSV
        df = pd.DataFrame(all_features)
        df.to_csv(output_csv, index=False)
        print(f"✅ Saved features to: {output_csv}")

print("\n🎉 All videos processed successfully!")
