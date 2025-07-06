# Face Detection and Recognition System

This project is a complete face detection and recognition pipeline using YOLOv8 for detection and FaceNet for recognition. It is designed to accurately identify individuals even when they wear masks or caps and store the entry time logs in a CSV file.


### Features:

Face detection using YOLOv8n-face model

Face recognition using FaceNet embeddings

Classification using KNN or SVM

Trained on images with variations: normal face, masked face, and face with a cap

Achieved 90% accuracy on the dataset

Records entry logs into a CSV file

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

.
├── app.py                      # Streamlit interface for face recognition deployment
├── detect_faces_yolo.py       # Detects and crops faces using YOLOv8
├── facenet_embeddings.py      # Extracts 128D embeddings from cropped faces using FaceNet
├── train_classifier.py        # Trains the face recognition classifier using embeddings
├── predict.py                 # Predicts the identity of a test image using trained classifier
├── surveillance.py            # Logs entry/exit timestamps from live webcam feed
├── yolov8n-face-lindevs.pt    # Pretrained YOLOv8 face detection model
├── face_classifier.pkl        # Trained face recognition classifier (saved model)
├── embeddings.npy             # NumPy array of FaceNet embeddings
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation
│
├── dataset/                   # Original training images (normal, masked, capped faces)
├── cropped_faces/             # YOLOv8-detected cropped face images
├── uploads/       
