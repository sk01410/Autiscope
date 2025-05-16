import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import joblib

# Mapping behavior names to folders
behavior_folders = {
    
    "repetitive_motion": "features/repetitive_motion",
    "social_deficit": "features/social_deficit",
    "normal": "features/normal"
}

# Folder to save models
os.makedirs("models", exist_ok=True)

# Loop through each behavior (excluding 'normal' which is used for negative examples)
for behavior, folder in behavior_folders.items():
    if behavior == "normal":
        continue  # Skip for now

    print(f"🔍 Training model for: {behavior}")

    X = []
    y = []

    # Load positive samples (label = 1)
    for filename in os.listdir(folder):
        filepath = os.path.join(folder, filename)
        df = pd.read_csv(filepath)
        flat_features = df.mean(axis=0).values  # Summary statistics (mean of features)
        X.append(flat_features)
        y.append(1)  # Label for present behavior

    # Load negative samples (label = 0) from normal videos
    normal_folder = behavior_folders["normal"]
    for filename in os.listdir(normal_folder):
        filepath = os.path.join(normal_folder, filename)
        df = pd.read_csv(filepath)
        flat_features = df.mean(axis=0).values
        X.append(flat_features)
        y.append(0)  # Label for normal

    # Convert to NumPy arrays
    X = np.array(X)
    y = np.array(y)

    # Preprocess: scale the data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the Random Forest model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

    # Save the model
    model_path = f"models/{behavior}_model.pkl"
    joblib.dump(model, model_path)
    print(f"✅ Saved model to {model_path}")
