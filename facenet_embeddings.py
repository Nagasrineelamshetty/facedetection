from keras_facenet import FaceNet
import numpy as np
import os

def preprocess_face(img_path):
    # If your images are already cropped faces, we can just read them using cv2
    import cv2
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img  # returns HxWx3 RGB image array

def generate_embeddings(embedder, input_dir, output_path="embeddings.npy"):
    data = []
    labels = []

    for person in os.listdir(input_dir):
        person_dir = os.path.join(input_dir, person)
        for img_name in os.listdir(person_dir):
            img_path = os.path.join(person_dir, img_name)
            img = preprocess_face(img_path)
            # Get 128‑D embedding
            embeddings = embedder.embeddings([img])
            embedding = embeddings[0]
            data.append(embedding)
            labels.append(person)

    np.save(output_path, {'embeddings': np.array(data), 'labels': np.array(labels)})
    print(f"Saved embeddings to {output_path}")

if __name__ == "__main__":
    embedder = FaceNet()
    generate_embeddings(embedder, input_dir='cropped_faces')
