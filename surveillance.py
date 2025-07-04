import cv2
import numpy as np
import joblib
import os
import csv
import time
from datetime import datetime
from keras_facenet import FaceNet
from ultralytics import YOLO
from sklearn.preprocessing import Normalizer

# Load models
yolo_model = YOLO("yolov8n-face-lindevs.pt")
embedder = FaceNet()
classifier = joblib.load("face_classifier.pkl")
labels = joblib.load("labels.pkl")
l2_normalizer = Normalizer("l2")

# Prepare log
os.makedirs("logs", exist_ok=True)
log_file = "logs/attendance_log.csv"
if not os.path.exists(log_file):
    with open(log_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Name", "Entry Time"])

logged_entries = set()

# Start camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Error: Could not open webcam.")
    exit()

print("📷 Surveillance started. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Error: Could not read from webcam.")
        break

    results = yolo_model(frame)
    frame_names = []

    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        if len(boxes) == 0:
            print("🔍 No faces detected.")
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            face = frame[y1:y2, x1:x2]
            try:
                face = cv2.resize(face, (160, 160))
            except:
                continue

            embedding = embedder.embeddings([face])[0]
            embedding = l2_normalizer.transform([embedding])
            pred = classifier.predict(embedding)[0]

            try:
                name = labels[pred]
            except:
                name = "Unknown"

            print(f"✅ Detected: {name}")
            frame_names.append(name)

            if name != "Unknown":
                if name not in logged_entries:
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    with open(log_file, "a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow([name, timestamp])
                    print(f"[ENTRY LOGGED] {name} at {timestamp}")
                    logged_entries.add(name)

            # Draw the box and name
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, name, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Display frame
    cv2.imshow("Surveillance Debug View", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("🛑 Surveillance stopped.")
