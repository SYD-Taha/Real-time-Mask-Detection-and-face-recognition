import streamlit as st
import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np
from PIL import Image

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Classes
class_names = ["With_Mask", "Without_Mask", "Mask_Worn_Incorrect"]
model_path = r"D:\Ai-lab\portfolio Projects\Real-time Mask Detection and face recognition\model\mask_detector.pt"

# Load model
model = models.mobilenet_v2(pretrained=False)
model.classifier[1] = nn.Linear(model.last_channel, 2)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# Face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Streamlit UI
st.title("🧠 Mask Detection AI")

choice = st.radio("Choose input type:", ["Upload Image", "Use Webcam"])

def detect_and_predict(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
    for (x, y, w, h) in faces:
        face = image[y:y+h, x:x+w]
        face_pil = Image.fromarray(face)
        input_tensor = transform(face_pil).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(input_tensor)
            _, preds = torch.max(outputs, 1)
            confidence = torch.softmax(outputs, dim=1)[0][preds.item()].item()
            label = f"{class_names[preds.item()]} ({confidence*100:.1f}%)"

        color = (0, 255, 0) if preds.item() == 0 else (0, 0, 255)
        cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)
        cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return image

if choice == "Upload Image":
    uploaded = st.file_uploader("Upload an image", type=['jpg', 'png', 'jpeg'])
    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        img_np = np.array(img)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        result_img = detect_and_predict(img_bgr)
        st.image(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB), caption="Prediction Result")

else:
    st.warning("Webcam in Streamlit is limited. Use the desktop OpenCV version (Day 3) for real-time.")
