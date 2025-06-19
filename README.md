Here is your complete and clean `README.md` file:

---

```markdown
# Face Detection and Recognition System

This project is a complete face detection and recognition pipeline using **YOLOv8** for detection and **FaceNet** for recognition. It is designed to identify individuals even when they wear masks or caps. This was developed as part of an internship assignment by **Nagasri**.

---

## Features

- Face detection using YOLOv8n-face model
- Face recognition using FaceNet embeddings
- Classification using KNN / SVM
- Trained with variations: normal face, masked face, and capped face
- 100% accuracy on a controlled test dataset
- Streamlit app for easy image upload and face recognition

---

## Tech Stack

```

Programming Language : Python 3.10
Face Detection       : YOLOv8n-face (Ultralytics)
Face Embedding       : FaceNet (via keras-facenet)
Classifier           : KNN / SVM (scikit-learn)
Image Processing     : OpenCV
Model Storage        : Joblib and NumPy
Evaluation Metric    : Accuracy
User Interface       : Streamlit (for deployment)

```

---

## Folder Structure

```

facedetection/
│
├── detect\_faces\_yolo.py       # YOLOv8-based face detection script
├── facenet\_embeddings.py      # Generate FaceNet embeddings from cropped faces
├── train\_classifier.py        # Train the SVM/KNN face recognition model
├── predict.py                 # Run predictions on test images
├── app.py                     # Streamlit app for deployment
│
├── yolov8n-face-lindevs.pt    # Pretrained YOLOv8 face detection model
├── face\_classifier.pkl        # Trained face recognition classifier
├── embeddings.npy             # Stored FaceNet embeddings
│
├── dataset/                   # Raw images (normal, masked, capped)
├── cropped\_faces/             # Detected and cropped faces
├── uploads/                   # Uploaded images via Streamlit
│
├── requirements.txt           # Project dependencies
└── README.md                  # Project documentation

````

---

## Setup Instructions

1. **Clone the repository**  
   ```bash
   git clone <your-repo-url>
   cd facedetection
````

2. **Create and activate virtual environment**

   ```bash
   python -m venv venv
   venv\Scripts\activate  # For Windows
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the system step-by-step**

   * Detect faces

     ```bash
     python detect_faces_yolo.py
     ```

   * Generate embeddings

     ```bash
     python facenet_embeddings.py
     ```

   * Train the classifier

     ```bash
     python train_classifier.py
     ```

   * Predict (optional test)

     ```bash
     python predict.py
     ```

   * Run the Streamlit app

     ```bash
     streamlit run app.py
     ```

---

## Notes

* Ensure the YOLOv8 model file `yolov8n-face-lindevs.pt` is placed in the root folder.
* The system works best when trained on at least 3 variations of each person (normal, masked, and capped).
* Accuracy may vary based on dataset quality and variation handling.

---

## Contact

For any questions or improvements, feel free to reach out to **Nagasri**.

```

---

Let me know if you’d like me to generate the actual file or include it in your repo structure!
```
