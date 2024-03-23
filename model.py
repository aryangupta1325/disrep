import pickle 
from sklearn.svm import SVC 
from sklearn.preprocessing import LabelEncoder

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

with open("label_encoder.pickle", "wb") as f:
    pickle.dump(le, f)
