import pickle
from sklearn.preprocessing import LabelEncoder

# Your actual labels used during training
labels = ["left", "right", "straight"]

# Initialize and fit the label encoder
label_encoder = LabelEncoder()
label_encoder.fit(labels)

# Save the fitted label encoder
with open("head_pose_label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

print("✅ Label Encoder saved as 'head_pose_label_encoder.pkl'")
print("Classes:", label_encoder.classes_)
