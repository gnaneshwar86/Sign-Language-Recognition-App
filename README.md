
---

# 🧠 Sign Language Recognition using Deep Learning (WLASL)

## 📌 Overview

This project implements a **Word-Level American Sign Language (ASL) Recognition System** using deep learning.
The system recognizes ASL gestures from video frames and predicts the corresponding word in real time using a webcam.

The project is built using the **WLASL (Word-Level American Sign Language) dataset** and a lightweight **MobileNetV2-based transfer learning model**, optimized for CPU-based training.

---

## 🎯 Problem Statement

Communication between hearing-impaired individuals and non-sign-language users can be challenging.
This project aims to bridge that gap by building an AI-powered system capable of recognizing ASL gestures and converting them into text.

---

## 📂 Dataset

* Dataset: **WLASL (Word-Level ASL Dataset)**
* Official Classes: 2000
* Available Classes (local subset): 190
* Selected Classes for Training: 20–30 (CPU optimized)

> ⚠ The dataset is **not included in this repository** due to licensing (C-UDA agreement) and size constraints.

### How to Download Dataset

1. Clone the official WLASL repository:

   ```
   https://github.com/dxli94/WLASL
   ```

2. Download raw videos:

   ```
   cd start_kit
   python video_downloader.py
   ```

3. Organize videos into:

   ```
   dataset/raw_videos/<class_name>/
   ```

---

## ⚙️ Project Pipeline

```
WLASL Videos
      ↓
Video–Annotation Matching
      ↓
Class Selection (Top N)
      ↓
Frame Extraction
      ↓
Transfer Learning (MobileNetV2)
      ↓
Model Training
      ↓
Real-Time Webcam Prediction
```

---

## 🧠 Model Architecture

* Base Model: **MobileNetV2**
* Transfer Learning Approach
* Global Average Pooling
* Dense Layer (ReLU)
* Softmax Output Layer
* Optimizer: Adam
* Loss: Sparse Categorical Crossentropy

The architecture is optimized to run efficiently on CPU systems.

---

## 🚀 Features

* Automatic dataset preparation
* Multi-class sign classification
* Real-time webcam prediction
* CPU-friendly training pipeline
* Modular project structure
* Clean GitHub-ready setup

---

## 🛠 Technologies Used

* Python
* TensorFlow / Keras
* OpenCV
* NumPy
* Scikit-learn
* WLASL Dataset

---

## 📁 Project Structure

```
SignLanguageRecognition/
│
├── dataset/              # Not pushed to GitHub
│   ├── raw_videos/
│   └── frames/
│
├── src/
│   ├── extract_frames.py
│   ├── train_model.py
│   └── predict_webcam.py
│
├── models/               # Not pushed to GitHub
│   └── asl_model.h5
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🔧 Installation

1. Clone the repository:

   ```
   git clone <your-repo-link>
   cd SignLanguageRecognition
   ```

2. Install dependencies:

   ```
   pip install -r requirements.txt
   ```

---

## ▶ Usage

### 1️⃣ Extract Frames

```
python src/extract_frames.py
```

### 2️⃣ Train Model

```
python src/train_model.py
```

### 3️⃣ Run Real-Time Prediction

```
python src/predict_webcam.py
```

Press **Q** to exit webcam.

---

## 📊 Performance

* Optimized for CPU training
* Works well with 20–30 classes
* Scalable to larger subsets with GPU

---

## ⚠️ Important Notes

* Dataset is excluded due to licensing restrictions.
* Some WLASL videos may be corrupted or missing.
* The preprocessing script automatically skips unreadable videos.

---

## 📈 Future Improvements

* Sequence-based video modeling (I3D / LSTM)
* Pose-based recognition using MediaPipe
* Web deployment (Streamlit)
* Mobile deployment (TensorFlow Lite)
* Sentence-level sign recognition

---

## 🎓 Academic Disclaimer

This project uses the WLASL dataset under the Computational Use of Data Agreement (C-UDA).
The dataset is intended for academic and research use only.

---

## 📚 References

Li et al.,
*Word-level Deep Sign Language Recognition from Video: A New Large-scale Dataset and Methods Comparison*, WACV 2020.

WLASL Repository:
[https://github.com/dxli94/WLASL](https://github.com/dxli94/WLASL)

---

## 👨‍💻 Author

Gnaneshwar R L
B.Tech / Computer Science
Project: Sign Language Recognition System

---
