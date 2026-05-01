import cv2
import os
import numpy as np

dataset_path = "dataset"

faces = []
labels = []
current_label = 0

for person_name in sorted(os.listdir(dataset_path)):
    person_folder = os.path.join(dataset_path, person_name)

    if os.path.isdir(person_folder):
        for image_name in os.listdir(person_folder):
            img_path = os.path.join(person_folder, image_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (200, 200))

            faces.append(img)
            labels.append(current_label)

        current_label += 1

if len(faces) == 0:
    print("No training data found.")
    exit()

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.train(faces, np.array(labels))

recognizer.save("model.yml")

print("Training completed successfully.")