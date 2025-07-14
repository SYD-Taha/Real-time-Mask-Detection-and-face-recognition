import torch
import torch.nn as nn
from torchvision import transforms, models
from facenet_pytorch import MTCNN
import cv2
import numpy as np
from PIL import Image

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = r"D:\Ai-lab\portfolio Projects\Real-time Mask Detection and face recognition\model\mask_detector.pt"

# Load model
model = models.mobilenet_v2(pretrained=False)
model.classifier[1] = nn.Linear(model.last_channel, 2)  
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval().to(device)

# Class names
class_names = ["With_Mask", "Without_Mask"]

# Image Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# MTCNN face detector
mtcnn = MTCNN(keep_all=True, device=device)

# Webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect faces
    boxes, _ = mtcnn.detect(frame)

    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            face = frame[y1:y2, x1:x2]

            if face.size == 0:
                continue  # Skip invalid crops

            try:
                face_pil = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
                input_tensor = transform(face_pil).unsqueeze(0).to(device)

                with torch.no_grad():
                    outputs = model(input_tensor)
                    _, preds = torch.max(outputs, 1)
                    confidence = torch.softmax(outputs, dim=1)[0][preds.item()].item() * 100
                    label = f"{class_names[preds.item()]}: {confidence:.1f}%"
            except Exception as e:
                label = "Error"
                confidence = 0

            color = (0, 255, 0) if "With_Mask" in label else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("Mask Detector (MTCNN)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
