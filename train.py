import cv2
import os
import numpy as np
import pickle

dataset_path = "dataset"

if not os.path.exists(dataset_path):
    print("Dataset folder not found")
    exit()

faces = []
labels = []
label_map = {}
current_label = 0

for person_name in sorted(os.listdir(dataset_path)):
    person_folder = os.path.join(dataset_path, person_name)

    if os.path.isdir(person_folder):
        label_map[current_label] = person_name

        for image_name in os.listdir(person_folder):
            img_path = os.path.join(person_folder, image_name)

            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if img is None:
                continue

            try:
                img = cv2.resize(img, (200, 200))
            except:
                continue

            faces.append(img)
            labels.append(current_label)

        current_label += 1

if len(faces) == 0:
    print("No training data found")
    exit()

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.train(faces, np.array(labels))

recognizer.save("model.yml")

with open("labels.pickle", "wb") as f:
    pickle.dump(label_map, f)

print("Training completed successfully")