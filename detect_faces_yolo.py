from ultralytics import YOLO
import cv2
import os

def detect_and_crop_faces(input_folder, output_folder, model_path='yolov8n-face-lindevs.pt'):
    model = YOLO(model_path)

    for person_name in os.listdir(input_folder):
        person_path = os.path.join(input_folder, person_name)
        output_person_folder = os.path.join(output_folder, person_name)
        os.makedirs(output_person_folder, exist_ok=True)

        for img_name in os.listdir(person_path):
            img_path = os.path.join(person_path, img_name)
            img = cv2.imread(img_path)

            results = model(img)[0]

            for i, det in enumerate(results.boxes):
                x1, y1, x2, y2 = map(int, det.xyxy[0])
                face_crop = img[y1:y2, x1:x2]
                save_path = os.path.join(output_person_folder, img_name)
                cv2.imwrite(save_path, face_crop)
                print(f"Saved cropped face: {save_path}")

if __name__ == "__main__":
    input_dir = "dataset"
    output_dir = "cropped_faces"
    detect_and_crop_faces(input_dir, output_dir)
