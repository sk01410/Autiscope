### README.md

```markdown
# 🧠 Autism Behavior Detection from Videos using ML & Computer Vision

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-enabled-green)](https://opencv.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Google-red)](https://mediapipe.dev/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This project uses **computer vision** and **machine learning** to detect early signs of **autism spectrum behaviors** in toddlers through **video analysis**.

It identifies:
- 👁️ **Eye Contact Issues**
- 🔁 **Repetitive Motions**
- 🧍‍♂️ **Social Interaction Deficits**

---

## 🚀 Features

- Extracts facial, gaze, and body keypoints from videos using **MediaPipe**
- Trains separate models for each behavior category
- Detects multiple behaviors from new videos
- Generates structured reports and visualizations
- Modular pipeline — easy to improve or expand
- Streamlit-based interface for testing and visualization

---

## 📁 Project Structure


AUTISMDETECT/
├── app.py                      # Streamlit app interface
├── dataset/                    # Raw labeled videos
│   ├── repetitive\_motion/
│   └── social\_deficit/
├── features/                   # Extracted feature CSVs
│   ├── repetitive\_motion/
│   └── social\_deficit/
├── models/                     # Trained classifiers (.pkl)
├── reports/                    # Output JSON reports
├── scripts/                    # Main scripts
│   ├── batch\_feature\_extractor.py
│   ├── train\_classifiers.py
│   └── test\_video.py
├── venv/                       # Virtual environment (optional)
└── requirements.txt


---

## 📦 Setup Instructions

### 1. Clone the Repository or Download Project

```bash
cd path/to/project
````

### 2. Create and Activate Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate       # On Windows
# OR
source venv/bin/activate    # On Mac/Linux
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🧠 Full Pipeline

### ✅ Step 1: Feature Extraction

Run this to extract facial/pose keypoints and save to CSV:

```bash
python scripts/batch_feature_extractor.py
```

---

### ✅ Step 2: Train Behavior Classifiers

Train separate ML models for each category:

```bash
python scripts/train_classifiers.py
```

Models are saved in `models/`.

---

### ✅ Step 3: Test a Combined Behavior Video

Run detection for a new video:

```bash
python scripts/test_video.py --video_path dataset/repetitive_motion/handflapping.mp4
```

Outputs:

* 📄 JSON report → `reports/<video>_report.json`
* 🧠 (Optional) visuals → `visuals/<video>_annotated.mp4`

---

## ✅ Streamlit Interface (GUI)

A friendly interface for uploading a video and seeing results.

### ▶️ Launch it:

```bash
streamlit run app.py
```

### 🖼 Features:

* Upload `.mp4` video
* Extract features and run predictions
* View report in JSON
* View pose/gaze plots (optional)
* Risk level calculation

---

## 📄 Sample Report Output

```json
{
  "eye_contact_issue": true,
  "repetitive_motion": false,
  "social_deficit": true,
  "autism_risk_level": "Moderate"
}
```

---

## 📌 Future Improvements

* Add speech & emotion detection
* Real-time detection via webcam
* Deep learning models (CNN, LSTM)
* Auto-labeling with feedback loop

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).

---

## 🙌 Acknowledgements

* [MediaPipe by Google](https://mediapipe.dev/)
* [Scikit-learn](https://scikit-learn.org/)
* [OpenCV](https://opencv.org/)

---

## ✉️ Contact

**Project by:** Sukhad Kaur
📧 Email: [sukhadkaur141005@gmail.com]

```
