import face_recognition
import numpy as np
import os
import pickle
from datetime import datetime
from deepface import DeepFace
from scipy.spatial.distance import cosine
import cv2

# Path to the main directory containing subdirectories for each person
path = r'C:\Users\Davi\PycharmProjects\pythonProject\Material'
classNames = []
encodeListKnown = []

# Function to find encodings using DeepFace
def findDeepFaceEncodings(images, model_name='VGG-Face'):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = DeepFace.represent(img, model_name=model_name, enforce_detection=False)[0]["embedding"]
        encodeList.append(encode)
    return encodeList

# Load or compute encodings
for person_name in os.listdir(path):
    person_folder = os.path.join(path, person_name)
    if os.path.isdir(person_folder):
        classNames.append(person_name)
        person_images = []
        for image_name in os.listdir(person_folder):
            image_path = os.path.join(person_folder, image_name)
            img = cv2.imread(image_path)
            if img is not None:
                person_images.append(img)

        if person_images:
            encodings = findDeepFaceEncodings(person_images)
            encodeListKnown.extend(encodings)

        print(f"Encodings for {person_name} completed.")

print("All encodings retrieved successfully.")

# Save encodings
with open('deepface_encodings.pkl', 'wb') as file:
    pickle.dump(encodeListKnown, file)

# Load encodings
with open('deepface_encodings.pkl', 'rb') as file:
    encodeListKnown = pickle.load(file)

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    if not success:
        break

    # Flip the image horizontally
    img = cv2.flip(img, 1)  # 1 for horizontal flip

    # Resize the frame for faster processing
    small_frame = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(rgb_small_frame)
    encodesCurFrame = DeepFace.represent(rgb_small_frame, model_name='VGG-Face', enforce_detection=False)

    for faceLoc, encodeFace in zip(facesCurFrame, encodesCurFrame):
        encodeFace = encodeFace["embedding"]
        print(encodeFace)
        print(len(encodeFace))
        matches = [cosine(encodeFace, enc) < 0.688 for enc in encodeListKnown]
        print(matches)
        faceDis = [cosine(encodeFace, enc) for enc in encodeListKnown]
        print(faceDis)
        matchIndex = np.argmin(faceDis)
        print(matchIndex)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
