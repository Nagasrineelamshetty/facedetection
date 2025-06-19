from ultralytics import YOLO
from keras_facenet import FaceNet
from sklearn.neighbors import KNeighborsClassifier
import cv2
import numpy as np
import joblib

# Load models
detector = YOLO('yolov8n-face-lindevs.pt')
embedder = FaceNet()
classifier = joblib.load('face_classifier.pkl')

def recognize_face(image_path):
    img = cv2.imread(image_path)
    results = detector(img)[0]

    if len(results.boxes) == 0:
        print("No face detected.")
        return

    for i, box in enumerate(results.boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        face_crop = img[y1:y2, x1:x2]
        face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)

        # Get 128D embedding
        embedding = embedder.embeddings([face_rgb])[0]
        name = classifier.predict([embedding])[0]

        # Draw and label
        cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(img, name, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        print(f"✅ Prediction: {name}")

    # Show result
    cv2.imshow("Prediction", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_image = "test.jpg"  # Replace with a path to any test image
    recognize_face(test_image)
