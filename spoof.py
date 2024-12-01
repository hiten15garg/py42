from deepface import DeepFace
from scipy.spatial.distance import cosine
import face_recognition
import cv2
import os
import pickle
import numpy as np

# Path to the main directory containing subdirectories for each person
path = r'C:\Users\Lenovo\OneDrive\Desktop\py42\assets'
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
    if not os.path.isdir(person_folder):
        continue
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

cap = cv2.VideoCapture(1)

while True:
    success, img = cap.read()
    if not success:
        break

    # Flip the image horizontally
    img = cv2.flip(img, 1)

    # Resize the frame for faster processing
    small_frame = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(rgb_small_frame)

    for faceLoc in facesCurFrame:
        # Extract face region for liveness check
        y1, x2, y2, x1 = faceLoc
        y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
        face_img = img[y1:y2, x1:x2]

        try:
            # Perform liveness detection using DeepFace
            verification = DeepFace.verify(face_img, face_img, enforce_detection=False, detector_backend='opencv')
            if not verification["verified"] or verification.get("distance") > 0.4:
                print("Spoof detected or liveness check failed.")
                continue
        except Exception as e:
            print(f"Error in liveness detection: {e}")
            continue

        # If liveness check passes, proceed with recognition
        encodeFace = DeepFace.represent(face_img, model_name='VGG-Face', enforce_detection=False)[0]["embedding"]
        faceDis = [cosine(encodeFace, enc) for enc in encodeListKnown]
        matches = [cosine(encodeFace, enc) < 0.688 for enc in encodeListKnown]

        if not matches or not any(matches):
            print("No match found for the detected face.")
            continue

        matchIndex = np.argmin(faceDis)
        if matchIndex >= len(classNames):
            print(f"Invalid matchIndex {matchIndex}. Skipping.")
            continue

        name = classNames[matchIndex].upper()
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
