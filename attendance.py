import cv2
import os
import pandas as pd
from datetime import datetime

# Load trained model
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("model.yml")

# Face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Rebuild label map
dataset_path = "dataset"
label_map = {}
current_label = 0

for person_name in sorted(os.listdir(dataset_path)):
    if os.path.isdir(os.path.join(dataset_path, person_name)):
        label_map[current_label] = person_name
        current_label += 1

attendance_file = "attendance.xlsx"

if not os.path.exists(attendance_file):
    df = pd.DataFrame(columns=["Name", "Date", "Time"])
    df.to_excel(attendance_file, index=False)

# Unknown handling setup
unknown_folder = "unknown_images"

if not os.path.exists(unknown_folder):
    os.makedirs(unknown_folder)

unknown_counter = 1
last_unknown_time = None
cooldown_seconds = 20

cap = cv2.VideoCapture(0)

print("Press ESC to exit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (200, 200))

        label, confidence = recognizer.predict(face)

        now = datetime.now()
        date = now.strftime("%Y-%m-%d")
        time_now = now.strftime("%H:%M:%S")

        df = pd.read_excel(attendance_file)

        if confidence < 80:
            name = label_map[label]

            new_entry = pd.DataFrame([[name, date, time_now]],
                                     columns=["Name", "Date", "Time"])
            df = pd.concat([df, new_entry], ignore_index=True)
            df.to_excel(attendance_file, index=False)

            cv2.putText(frame, name, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        else:
            current_time = datetime.now()

            if last_unknown_time is None or \
               (current_time - last_unknown_time).seconds > cooldown_seconds:

                unknown_name = f"Unknown_Person{unknown_counter}"

                new_entry = pd.DataFrame([[unknown_name, date, time_now]],
                                         columns=["Name", "Date", "Time"])
                df = pd.concat([df, new_entry], ignore_index=True)
                df.to_excel(attendance_file, index=False)

                image_path = os.path.join(unknown_folder, f"{unknown_name}.jpg")
                cv2.imwrite(image_path, frame)

                cv2.putText(frame, unknown_name, (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

                unknown_counter += 1
                last_unknown_time = current_time

            else:
                cv2.putText(frame, "Invalid Person", (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)

    cv2.imshow("Smart Attendance System", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()