import joblib

labels = {
    0: "person1",
    1: "person2",
}

joblib.dump(labels, "labels.pkl")
print("✅ labels.pkl saved.")
