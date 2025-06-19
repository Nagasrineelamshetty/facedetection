# Face detection and recognition:
A face detection and recognition system that accurately identifies individuals—even with masks or caps—using YOLOv8 and FaceNet.

### Tech Stack:
Programming Language : Python 3.10
Face Detection       : YOLOv8n-face (Ultralytics)
Face Embedding       : FaceNet (via keras-facenet)
Classifier           : KNN / SVM (scikit-learn)
Image Processing     : OpenCV
Model Storage        : Joblib and NumPy
Evaluation Metric    : Accuracy

### Folder Structure:
facedetection/
│
├── detect_faces_yolo.py       # YOLOv8-based face detection script
├── facenet_embeddings.py      # Generate FaceNet embeddings from cropped faces
├── train_classifier.py        # Train the SVM/KNN face recognition model
├── predict.py                 # Run predictions on test images
│
├── yolov8n-face-lindevs.pt    # Pretrained YOLOv8 face detection model
├── face_classifier.pkl        # Trained face recognition classifier
├── embeddings.npy             # Stored FaceNet embeddings
│
├── dataset/                   # Folder for raw images (normal, masked, capped)
├── cropped_faces/             # Folder for storing cropped face images
│
├── requirements.txt           # List of all required Python packages
└── README.md                  # Project overview and instructions

