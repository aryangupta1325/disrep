import os
import cv2
import numpy as np
import pickle
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder

# Load face detector and facial embeddings model
proto_path = "face_detection_model/deploy.prototxt"
model_path = "face_detection_model/res10_300x300_ssd_iter_140000.caffemodel"
embedder_path = "face_detection_model/openface_nn4.small2.v1.t7"
detector = cv2.dnn.readNetFromCaffe(proto_path, model_path)
embedder = cv2.dnn.readNetFromTorch(embedder_path)

# Directory containing user datasets
dataset_dir = "./dataset"

# Dictionary to store embeddings for each user
user_embeddings = {}

# Function to extract embeddings for a given image
def extract_embeddings(image):
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=False, crop=False)
    detector.setInput(blob)
    detections = detector.forward()
    
    # Loop over the detections
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        # Filter out weak detections
        if confidence > 0.5:
            # Compute the (x, y)-coordinates of the bounding box
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            face = image[startY:endY, startX:endX]

            # Ensure the face width and height are sufficiently large
            if face.shape[0] >= 20 and face.shape[1] >= 20:
                # Preprocess the face and extract embeddings
                face_blob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
                embedder.setInput(face_blob)
                vec = embedder.forward()
                return vec.flatten()

# Iterate through each user's dataset folder
for user_dir in os.listdir(dataset_dir):
    user_name = user_dir
    user_embeddings[user_name] = []

    # Get image paths for the current user
    user_images_dir = os.path.join(dataset_dir, user_dir)
    
    # Ensure the user_images_dir is a directory
    if not os.path.isdir(user_images_dir):
        continue
    
    # Extract embeddings for each image in the user's folder
    for image_file in os.listdir(user_images_dir):
        image_path = os.path.join(user_images_dir, image_file)
        
        # Skip non-image files
        if not image_file.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp')):
            continue
        
        image = cv2.imread(image_path)
        if image is not None:
            embedding = extract_embeddings(image)
            if embedding is not None:
                user_embeddings[user_name].append(embedding)

# Save the embeddings to a pickle file
with open("user_embeddings.pickle", "wb") as f:
    pickle.dump({"embeddings": user_embeddings}, f)

# Load user embeddings
with open("user_embeddings.pickle", "rb") as f:
    data = pickle.load(f)

# Extract embeddings and labels from the user embeddings dictionary
embeddings = []
labels = []
for user_name, user_data in data["embeddings"].items():
    embeddings.extend(user_data)
    labels.extend([user_name] * len(user_data))

# Convert labels to numerical form using LabelEncoder
le = LabelEncoder()
labels = le.fit_transform(labels)

# Train the classifier
classifier = SVC(C=1.0, kernel="linear", probability=True)
classifier.fit(embeddings, labels)

# Save the trained classifier and label encoder
with open("classifier.pickle", "wb") as f:
    pickle.dump(classifier, f)

with open("label.pickle", "wb") as f:
    pickle.dump(le, f)
