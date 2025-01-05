# Real-Time Pose Classification using LSTM & Mediapipe

This project implements **real-time human pose classification** using an **LSTM-based deep learning model** trained on **Mediapipe keypoints**. It allows you to detect and classify movements (e.g., *punch* vs. *neutral*) directly from a webcam feed.

**Why Keypoints Instead of Images?**  
By using **Mediapipe keypoints** instead of raw images, we make classification **faster, more efficient, and less dependent on lighting conditions**. The LSTM model learns **motion patterns** from sequences of keypoints, making it great for **gesture/action recognition**.

---

## Features
**Real-time video processing** with OpenCV.  
**Pose and hand tracking** using Mediapipe's Holistic model.  
**LSTM-based deep learning** for sequence classification.  
**Efficient keypoint-based training** (instead of raw images).  
**Real-time predictions displayed on video**.  

---

## Libraries Used
- `tensorflow` - Deep learning framework for training LSTM model  
- `mediapipe` - Pose & hand tracking model  
- `opencv-python` - Real-time video processing  
- `numpy` - Data handling and keypoint storage  

---

## How It Works
1. **Keypoint Extraction**: Uses **Mediapipe Holistic** to extract **33 pose landmarks** and **42 hand landmarks**.
2. **Data Collection**: Stores **30-frame sequences** of **pose + hand keypoints** and labels them (`punch`, `neutral`).
3. **LSTM Model Training**: A **recurrent neural network (RNN)** is trained on these keypoint sequences to classify movements.
4. **Real-Time Prediction**: Captures video, extracts keypoints, maintains a rolling buffer of 30 frames, and predicts actions in real-time.

---

## Accuracy Disclaimer
**This project is for educational purposes**, and accuracy may be **limited** due to the small dataset collected.  

To improve accuracy:
- **Capture more training data** using your own webcam.
- **Use action clips from UFC/MMA fights** for diverse movement samples.  
- **Fine-tune the LSTM model** with more training epochs and hyperparameter tuning.

---

## Installation & Setup

### Install Dependencies
```bash
pip install tensorflow opencv-python mediapipe numpy
