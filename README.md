# NeuroAegis – AI-Based Support System for Myasthenia Gravis

NeuroAegis is an AI-assisted healthcare support system designed to help monitor symptoms associated with Myasthenia Gravis (MG). The system combines Machine Learning, voice analysis, and Computer Vision to assess fatigue, voice-related symptoms, and ptosis, while maintaining health assessment records through an interactive web interface.

> **Note:** NeuroAegis is an academic/project prototype intended for symptom monitoring and research purposes. It is not a medical diagnostic system and does not replace professional medical advice.

---

## Problem Statement

Myasthenia Gravis is a neuromuscular disorder in which symptoms such as muscle fatigue, speech difficulties, and eyelid drooping can fluctuate over time.

Traditional symptom tracking can be difficult because assessments may depend on manual observation and patient-reported information.

NeuroAegis aims to provide a centralized digital platform for AI-assisted symptom assessment and health monitoring.

---

## Solution

NeuroAegis integrates three major assessment modules:

- **Fatigue Detection** – Machine Learning-based fatigue assessment.
- **Voice Analysis** – Voice signal processing and Machine Learning for voice-related fatigue/dysarthria analysis.
- **Ptosis Detection** – Computer Vision-based eye analysis using facial landmarks.

The results are processed through a Python backend and presented through a web-based dashboard.

---

## Tech Stack

### Frontend
- HTML5
- CSS3
- JavaScript
- Fetch API

### Backend
- Python
- Flask
- Flask-CORS
- REST APIs

### Machine Learning
- Scikit-learn
- NumPy
- Pandas
- Joblib

### Voice Analysis
- Librosa
- SoundDevice
- SoundFile

### Computer Vision
- OpenCV
- MediaPipe Face Mesh

### Development Tools
- VS Code
- Git
- GitHub

---

## System Architecture

<img width="1536" height="1024" alt="image" src="https://github.com/user-attachments/assets/64c7518e-ed66-4020-b00a-3e1eefe9df82" />

