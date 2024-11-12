import cv2
import numpy as np
import face_recognition
import os
import pickle
from datetime import datetime
from deepface import DeepFace
from scipy.spatial.distance import cosine

# Path to images
path = r'C:\Users\Davi\PycharmProjects\pythonProject\Material'
images = []
classNames = []
myList = os.listdir(path)
print(myList)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)


# Function to find encodings using face_recognition
# def findEncodings(images):
#     encodeList = []
#     for img in images:
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         encode = face_recognition.face_encodings(img)[0]
#         encodeList.append(encode)
#     return encodeList


# Function to find encodings using DeepFace
def findDeepFaceEncodings(images, model_name='Facenet512'):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = DeepFace.represent(img, model_name=model_name, enforce_detection=False)[0]["embedding"]
        encodeList.append(encode)
    return encodeList


# Load or compute encodings
try:
    with open('deepface_encodings.pkl', 'rb') as file:
        encodeListKnown = pickle.load(file)
        print("Existing DeepFace embeddings loaded successfully.")
except FileNotFoundError:
    encodeListKnown = findDeepFaceEncodings(images)
    with open('deepface_encodings.pkl', 'wb') as file:
        pickle.dump(encodeListKnown, file)
        print("DeepFace embeddings computed and saved.")

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    if not success:
        break

    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = DeepFace.represent(imgS, model_name='Facenet512', enforce_detection=False)

    for faceLoc, encodeFace in zip(facesCurFrame, encodesCurFrame):
        encodeFace = encodeFace["embedding"]
        matches = [cosine(encodeFace, enc) < 0.688 for enc in encodeListKnown]
        faceDis = [cosine(encodeFace, enc) for enc in encodeListKnown]
        # matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        # faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)

        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            # markAttendance(name)  # Uncomment to enable attendance marking

    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
