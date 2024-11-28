# import cv2
# import numpy as np
# import face_recognition
# import os
# import pickle
# from deepface import DeepFace
# from scipy.spatial.distance import cosine

# # Path to images
# IMAGE_PATH = r'C:\Users\Lenovo\OneDrive\Desktop\py42\assets'

# # Load images and extract class names
# def load_images(path):
#     images = []
#     classNames = []
#     for file_name in os.listdir(path):
#         if file_name.endswith(('.jpg', '.png')):
#             image = cv2.imread(os.path.join(path, file_name))
#             images.append(image)
#             classNames.append(os.path.splitext(file_name)[0])
#     return images, classNames

# # Generate DeepFace encodings
# def find_deepface_encodings(images, model_name='Facenet512'):
#     encodings = []
#     for img in images:
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         embedding = DeepFace.represent(img, model_name=model_name, enforce_detection=False)[0]["embedding"]
#         encodings.append(embedding)
#     return encodings

# # Load or compute DeepFace encodings
# def get_known_encodings(images, model_name='Facenet512', file_name='deepface_encodings.pkl'):
#     try:
#         with open(file_name, 'rb') as file:
#             encodings = pickle.load(file)
#             print("Loaded existing encodings.")
#     except FileNotFoundError:
#         encodings = find_deepface_encodings(images, model_name=model_name)
#         with open(file_name, 'wb') as file:
#             pickle.dump(encodings, file)
#             print("Computed and saved new encodings.")
#     return encodings

# # Initialize variables
# images, classNames = load_images(IMAGE_PATH)
# known_encodings = get_known_encodings(images, model_name='Facenet512')

# # Real-time face recognition
# def recognize_faces():
#     cap = cv2.VideoCapture(0)
#     while True:
#         success, frame = cap.read()
#         if not success:
#             break

#         # Resize and convert frame for faster processing
#         small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
#         rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

#         # Detect faces and generate embeddings
#         face_locations = face_recognition.face_locations(rgb_small_frame)
#         current_encodings = DeepFace.represent(rgb_small_frame, model_name='Facenet512', enforce_detection=False)

#         for face_loc, face_embedding in zip(face_locations, current_encodings):
#             embedding = face_embedding["embedding"]
#             distances = [cosine(embedding, known_enc) for known_enc in known_encodings]
#             min_distance = min(distances)
#             match_index = np.argmin(distances)

#             if min_distance < 0.688:  # Threshold for DeepFace cosine similarity
#                 name = classNames[match_index].upper()
#             else:
#                 name = "Unknown"

#             # Scale face locations back to the original frame size
#             top, right, bottom, left = face_loc
#             top, right, bottom, left = top * 4, right * 4, bottom * 4, left * 4

#             # Draw a box around the face and label it
#             cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
#             cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
#             cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

#         cv2.imshow('Face Recognition', frame)

#         # Break the loop on 'q' key press
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()

# # Start real-time recognition
# recognize_faces()

import cv2
import numpy as np
import face_recognition
import os
import pickle
from deepface import DeepFace
from scipy.spatial.distance import cosine

# Path to images
IMAGE_PATH = r'C:\Users\Lenovo\OneDrive\Desktop\py42\assets'

# Load images and extract class names
def load_images(path) :
    images = []
    classNames = []
    for file_name in os.listdir(path):
        if file_name.endswith(('.jpg', '.png')):
            image = cv2.imread(os.path.join(path, file_name))
            images.append(image)
            classNames.append(os.path.splitext(file_name)[0])
    return images, classNames

# Generate DeepFace encodings
def find_deepface_encodings(images, model_name='VGG-Face') :  # Use a lighter model
    encodings = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        embedding = DeepFace.represent(img, model_name=model_name, enforce_detection=False)[0]["embedding"]
        encodings.append(embedding)
    return encodings

# Load or compute DeepFace encodings
def get_known_encodings(images, model_name='VGG-Face', file_name='deepface_encodings.pkl'):
    try:
        with open(file_name, 'rb') as file:
            encodings = pickle.load(file)
            print("Loaded existing encodings.")
    except FileNotFoundError:
        encodings = find_deepface_encodings(images, model_name=model_name)
        with open(file_name, 'wb') as file:
            pickle.dump(encodings, file)
            print("Computed and saved new encodings.")
    return encodings

# Initialize variables
images, classNames = load_images(IMAGE_PATH)
known_encodings = get_known_encodings(images, model_name='VGG-Face')

# Real-time face recognition
def recognize_faces():
    cap = cv2.VideoCapture(1)
    frame_skip = 3  # Process every 3rd frame
    frame_count = 0

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame_count += 1
        if frame_count % frame_skip != 0:
            continue

        # Resize and convert frame for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Detect faces and generate embeddings
        face_locations = face_recognition.face_locations(rgb_small_frame, model="hog")  # Faster but less accurate
        current_encodings = DeepFace.represent(rgb_small_frame, model_name='VGG-Face', enforce_detection=False)

        for face_loc, face_embedding in zip(face_locations, current_encodings):
            embedding = face_embedding["embedding"]
            distances = [cosine(embedding, known_enc) for known_enc in known_encodings]
            min_distance = min(distances)
            match_index = np.argmin(distances)

            if min_distance < 0.324:  # Threshold for DeepFace cosine similarity
                name = classNames[match_index].upper()
            else:
                name = "Unknown"

            # Scale face locations back to the original frame size
            top, right, bottom, left = face_loc
            top, right, bottom, left = top * 4, right * 4, bottom * 4, left * 4

            # Draw a box around the face and label it
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow('Face Recognition', frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Start real-time recognition
recognize_faces()
