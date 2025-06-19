# Face Detection and Recognition System

This project is a complete face detection and recognition pipeline using YOLOv8 for detection and FaceNet for recognition. It is designed to accurately identify individuals even when they wear masks or caps.

### Features:

Face detection using YOLOv8n-face model

Face recognition using FaceNet embeddings

Classification using KNN or SVM

Trained on images with variations: normal face, masked face, and face with a cap

Achieved 100% accuracy on a small test dataset

### Tech Stack:

Programming Language: Python 3.10

Face Detection: YOLOv8n-face (Ultralytics)

Face Embedding: FaceNet (via keras-facenet)

Classifier: KNN / SVM (scikit-learn)

Image Processing: OpenCV

Model Storage: Joblib and NumPy

Evaluation Metric: Accuracy

User Interface: Streamlit

### Folder Structure:

detect_faces_yolo.py – Detects and crops faces using YOLOv8

facenet_embeddings.py – Extracts 128-dimensional embeddings using FaceNet

train_classifier.py – Trains the face recognition classifier

predict.py – Predicts the identity of a test image

app.py – Streamlit interface for deployment

yolov8n-face-lindevs.pt – Pretrained face detection model

face_classifier.pkl – Saved trained classifier

embeddings.npy – NumPy array containing FaceNet embeddings

dataset/ – Contains training images (normal, masked, and capped)

cropped_faces/ – YOLO-detected cropped face images

uploads/ – Folder for uploaded images during Streamlit app testing

requirements.txt – Python package dependencies

README.md – Project documentation
