# 🧠 Real-Time Mask Detection & Face Recognition System

---

## 📌 About the Project

A high-performance AI system combining **Face Mask Detection** and **Face Recognition** in real-time, designed to demonstrate production-grade computer vision expertise. The project uses:

- ✅ MobileNetV2 (PyTorch) for Mask Classification
- ✅ MTCNN (facenet-pytorch) for Deep Learning–based Face Detection
- ✅ DeepFace (ArcFace / Facenet backend) for Face Recognition
- ✅ Streamlit for Web Interface
- ✅ OpenCV for Real-time Webcam Integration

This system can:

- Detect whether a person is **wearing a mask, not wearing a mask, or wearing incorrectly**.
- Recognize known faces using a pretrained face recognition model.
- Run as a local app (Streamlit) or be extended as a backend API (FastAPI recommended).

---

## 🚀 Features

- 🎥 Real-time Mask Detection via Webcam (OpenCV + PyTorch)
- 🖼️ Upload Image Interface via Streamlit
- 👤 Deep Learning–based Face Detection (MTCNN)
- 🧑‍💼 Face Recognition using Pretrained Models (DeepFace)
- ⚡ GPU Acceleration Support
- 📈 High Accuracy & Low Latency

---

## 📁 Project Structure

```
project/
├── mask_detector.pt             # Trained MobileNetV2 model
├── face_db/                     # Folder containing known faces (e.g. Alice.jpg)
├── app.py                       # Streamlit app for image-based usage
├── realtime.py                  # OpenCV real-time webcam application
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation
```

---

## ⚙️ Setup Instructions

1. **Clone the Repository:**

```bash
git clone <repository_url>
cd project
```

2. **Install Dependencies:**

```bash
pip install -r requirements.txt
```

3. **Place Known Faces:**

- Add clear face images in the `face_db/` folder.

4. **Run Streamlit Web App:**

```bash
streamlit run app.py
```

5. **Run Real-time Webcam Application:**

```bash
python realtime.py
```

---

## 🛠️ Requirements

- torch
- torchvision
- facenet-pytorch
- opencv-python
- opencv-python-headless
- numpy
- pillow
- matplotlib
- seaborn
- scikit-learn
- streamlit
- deepface
- tensorflow

---

## 🏆 Deployment Suggestions

- Use Streamlit for browser-based interface (local users).
- Wrap as FastAPI backend for production-ready API serving.
- Optionally Dockerize for easy deployment.

---

## 📖 How It Works

- **MTCNN** detects faces using a deep learning face detector.
- **MobileNetV2** classifies each detected face as Mask / No Mask / Incorrect.
- **DeepFace** compares detected faces against your face database (`face_db/`) to recognize known identities.
- Results are drawn on images or real-time webcam frames.

---

## 📷 Example Output

- ✅ Mask Detection: With\_Mask (97.4%)
- ✅ Face ID: Alice

---

## 📚 Future Enhancements

- Dockerfile for containerized deployment.
- FastAPI backend with REST endpoints.
- Hugging Face ViT backbone for further optimization.
- CI/CD Pipeline for automatic builds.

---

## 👤 Author

**Your Name**\
AI Engineer | Computer Vision Specialist

---

## 📄 License

This project is licensed for educational and demonstration purposes.

