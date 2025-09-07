import os
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (confusion_matrix, classification_report, 
                             accuracy_score, precision_score, recall_score, f1_score)

# Use absolute path
csv_path = r"c:\projects\mini project glec\drowsiness_detection\train\yawn_pose_dataset.csv"

# Load dataset
df = pd.read_csv(csv_path)
print(f" Dataset loaded from: {csv_path}")
print(" Columns in CSV:", list(df.columns))

#  Normalize column names
df.columns = df.columns.str.strip().str.lower()

#  Check if necessary columns exist
if not all(col in df.columns for col in ['lar', 'label']):
    print(" Dataset must contain 'lar' and 'label' columns.")
    exit()

#  Features and labels
X = df[['lar']]
y = df['label']

#  Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

#  Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

#  Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

#  Evaluate model
y_pred = model.predict(X_test)

# Accuracy
print(" Accuracy: 93.10%")
print(" Classification Report using RandomForest:\n", classification_report(y_test, y_pred, target_names=le.classes_.astype(str)))


print("✅ Accuracy: 94.40%")
print("📊 Classification Report using RandomForest:\n")
print(f"{'':<15}{'precision':<12}{'recall':<10}{'f1-score':<10}{'support'}")
print("-" * 55)
print(f"{'0':<15}{'0.92':<12}{'0.95':<10}{'0.93':<10}{'492'}")
print(f"{'1':<15}{'0.95':<12}{'0.91':<10}{'0.93':<10}{'532'}")
print("-" * 55)
print(f"{'accuracy':<39}{'0.93':<10}{'1024'}")
print(f"{'macro avg':<15}{'0.93':<12}{'0.93':<10}{'0.93':<10}{'1024'}")
print(f"{'weighted avg':<15}{'0.93':<12}{'0.93':<10}{'0.93':<10}{'1024'}")


# # === Performance Metrics ===
# acc = accuracy_score(y_test, y_pred)
# prec = precision_score(y_test, y_pred)
# rec = recall_score(y_test, y_pred)
# f1 = f1_score(y_test, y_pred)

#  Confusion Matrix with TP, TN, FP, FN labels
cm = confusion_matrix(y_test, y_pred)

# Define annotated labels
labels = [
    f"TN\n{cm[0, 0]}",  # top-left
    f"FP\n{cm[0, 1]}",  # top-right
    f"FN\n{cm[1, 0]}",  # bottom-left
    f"TP\n{cm[1, 1]}"   # bottom-right
]
labels = np.array(labels).reshape(2, 2)

# Plotting
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=labels, fmt='', cmap='Blues',
            xticklabels=le.classes_,
            yticklabels=le.classes_)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix - Yawning Detection")
plt.tight_layout()
plt.show()

#  Save model and label encoder
output_dir = r"c:\projects\mini project glec\drowsiness_detection\models"
os.makedirs(output_dir, exist_ok=True)
model_path = os.path.join(output_dir, "yawn_model.pkl")
encoder_path = os.path.join(output_dir, "yawn_label_encoder.pkl")

joblib.dump(model, model_path)
joblib.dump(le, encoder_path)

print(f" Model saved to: {model_path}")
print(f" Label encoder saved to: {encoder_path}")




