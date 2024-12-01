# import face_recognition
# import numpy as np
# import cv2
# from deepface import DeepFace
# from pymongo import MongoClient
# from dotenv import load_dotenv
# from scipy.spatial.distance import cosine
# import requests
# from io import BytesIO
# from PIL import Image

# # Load the .env file
# load_dotenv()

# # MongoDB setup
# # mongodb_url = os.getenv("mongodb_url")
# # client = MongoClient("mongodb_url", serverSelectionTimeoutMS=5000)
# client = MongoClient("mongodb+srv://jas0113_kaur:J0111K0111Jj@cluster0.uzaq8.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
# db = client["NewDB"]
# collection = db["NewCollection"]

# try:
#     client.server_info()  # Force connection to test
#     print("Connected to MongoDB successfully!")
# except Exception as e:
#     print(f"Failed to connect to MongoDB: {e}")

# # Helper function to fetch and decode images from URL
# def fetch_image_from_url(url):
#     response = requests.get(url)
#     if response.status_code == 200:
#         image = Image.open(BytesIO(response.content))
#         return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
#     else:
#         raise ValueError(f"Failed to fetch image from URL: {url}")

# # Function to find encodings using DeepFace
# def findDeepFaceEncodings(images, model_name='Facenet'):
#     encodeList = []
#     for img in images:
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         encode = DeepFace.represent(img, model_name=model_name, enforce_detection=False)[0]["embedding"]
#         encodeList.append(encode)
#     return encodeList

# # Load images and compute encodings from MongoDB
# classNames = []
# encodeListKnown = []

# # Fetch data from MongoDB
# documents = collection.find({})
# for doc in documents:
#     name = doc["Name"]
#     image_url = doc["photo"]  # Assume this field contains the image URL

#     try:
#         # Fetch the image from URL
#         img = fetch_image_from_url(image_url)
#         encodings = findDeepFaceEncodings([img])

#         # Append results
#         classNames.append(name)
#         encodeListKnown.extend(encodings)
#         print(f"Encodings for {name} completed.")
#     except Exception as error:
#         print(f"Error processing {name}: {error}")

# print("All encodings retrieved successfully.")

# # Webcam live face recognition
# capture = cv2.VideoCapture(1)

# while True:
#     isTrue, frame = capture.read()
#     if not isTrue:
#         break

#     # Flip the image horizontally
#     frames = cv2.flip(frame, 1)

#     # Resize the frame for faster processing
#     small_frame = cv2.resize(frames, (0, 0), fx=0.25, fy=0.25)
#     rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

#     facesCurFrame = face_recognition.face_locations(rgb_small_frame)
#     encodesCurFrame = DeepFace.represent(rgb_small_frame, model_name='Facenet', enforce_detection=False)

#     for faceLoc, encodeFace in zip(facesCurFrame, encodesCurFrame):
#         encodeFace = encodeFace["embedding"]
#         faceDis = [cosine(encodeFace, enc) for enc in encodeListKnown]
#         matches = [cosine(encodeFace, enc) < 0.688 for enc in encodeListKnown]

#         if not matches or not any(matches):
#             print("No match found for the detected face.")
#             continue

#         matchIndex = np.argmin(faceDis)
#         if matchIndex >= len(classNames):
#             print(f"Invalid matchIndex {matchIndex}. Skipping.")
#             continue

#        # Extract the name and resize face location coordinates
#         name = classNames[matchIndex].upper()
#         (top, right, bottom, left) = faceLoc
#         (top, right, bottom, left) = top * 4, right * 4, bottom * 4, left * 4

#         # Draw a rectangle around the detected face
#         cv2.rectangle(frames, (left, top), (right, bottom), (0, 255, 0), thickness=2)

#         # Draw a filled rectangle below the face for the name label
#         cv2.rectangle(frames, (left, bottom - 35), (right, bottom), (0, 255, 0), thickness=cv2.FILLED)

#         # Put the name text on the filled rectangle
#         cv2.putText(frames, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_COMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)

#     cv2.imshow('Webcam', frames)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# capture.release()
# cv2.destroyAllWindows() 


import face_recognition
import numpy as np
import cv2
from deepface import DeepFace
from pymongo import MongoClient
from dotenv import load_dotenv
from scipy.spatial.distance import cosine
import requests
from io import BytesIO
from PIL import Image

# Load the .env file
load_dotenv()

# MongoDB setup
# mongodb_url = os.getenv("mongodb_url")
# client = MongoClient("mongodb_url", serverSelectionTimeoutMS=5000)
client = MongoClient("mongodb+srv://jas0113_kaur:J0111K0111Jj@cluster0.uzaq8.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
db = client["NewDB"]
collection = db["NewCollection"]

try:
    client.server_info()  # Force connection to test
    print("Connected to MongoDB successfully!")
except Exception as e:
    print(f"Failed to connect to MongoDB: {e}")

# Helper function to fetch and decode images from URL
def fetch_image_from_url(url):
    response = requests.get(url)
    if response.status_code == 200:
        image = Image.open(BytesIO(response.content))
        return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    else:
        raise ValueError(f"Failed to fetch image from URL: {url}")

# Function to find encodings using DeepFace
def findDeepFaceEncodings(images, model_name='Facenet'):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = DeepFace.represent(img, model_name=model_name, enforce_detection=False)[0]["embedding"]
        encodeList.append(encode)
    return encodeList

# Load images and compute encodings from MongoDB
classNames = []
encodeListKnown = []

# Fetch data from MongoDB
documents = collection.find({})
for doc in documents:
    name = doc["Name"]
    image_url = doc["photo"]  # Assume this field contains the image URL

    try:
        # Fetch the image from URL
        img = fetch_image_from_url(image_url)
        encodings = findDeepFaceEncodings([img])

        # Append results
        classNames.append(name)
        encodeListKnown.extend(encodings)
        print(f"Encodings for {name} completed.")
    except Exception as error:
        print(f"Error processing {name}: {error}")

print("All encodings retrieved successfully.")

# Webcam live face recognition
capture = cv2.VideoCapture(1)

while True:
    isTrue, frame = capture.read()
    if not isTrue:
        break

    # Flip the image horizontally
    frames = cv2.flip(frame, 1)

    # Resize the frame for faster processing
    small_frame = cv2.resize(frames, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(rgb_small_frame)
    encodesCurFrame = [
        DeepFace.represent(rgb_small_frame, model_name='Facenet', enforce_detection=False)[0]["embedding"]
        for _ in facesCurFrame
    ]

    for faceLoc, encodeFace in zip(facesCurFrame, encodesCurFrame):
        faceDis = [cosine(encodeFace, enc) for enc in encodeListKnown]
        matches = [dis < 0.688 for dis in faceDis]

        if not matches or not any(matches):
            continue

        matchIndex = np.argmin(faceDis)
        name = classNames[matchIndex].upper()

        (top, right, bottom, left) = faceLoc
        top, right, bottom, left = top * 4, right * 4, bottom * 4, left * 4

        # Draw a rectangle around the detected face
        cv2.rectangle(frames, (left, top), (right, bottom), (0, 255, 0), thickness=2)

        # Draw a filled rectangle below the face for the name label
        cv2.rectangle(frames, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)

        # Put the name text on the filled rectangle
        cv2.putText(frames, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_COMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)

    cv2.imshow('Webcam', frames)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
