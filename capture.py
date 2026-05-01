import cv2
import os

student_name = input("Enter student name: ").strip()

dataset_path = "dataset"
person_path = os.path.join(dataset_path, student_name)

if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)

if not os.path.exists(person_path):
    os.makedirs(person_path)

face_cascade = cv2.CascadeClassifier("face_model.xml")

if face_cascade.empty():
    print("Error loading face_model.xml")
    exit()

cap = cv2.VideoCapture(0)

count = 0
max_images = 20

print("Look at the camera. Capturing images...")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to access camera")
        break

    frame = cv2.resize(frame, (640, 480))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (200, 200))

        count += 1
        file_path = os.path.join(person_path, f"{count}.jpg")
        cv2.imwrite(file_path, face)

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)

    if count >= max_images:
        break

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()

print(f"Done! {count} images saved for {student_name}.")