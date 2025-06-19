# train_classifier.py

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import joblib

# Load embeddings
data = np.load('embeddings.npy', allow_pickle=True).item()
X = np.array(data['embeddings'])
y = np.array(data['labels'])

# Split into train/test (even though tiny data, helps test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Train classifier
clf = KNeighborsClassifier(n_neighbors=1, metric='euclidean')
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f" Accuracy: {acc * 100:.2f}%")

# Save classifier
joblib.dump(clf, 'face_classifier.pkl')
print(" Saved model as face_classifier.pkl")
