import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (confusion_matrix, classification_report, 
                             accuracy_score, precision_score, recall_score, f1_score)
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

# === Load Data ===
csv_path = r'C:\projects\mini project glec\drowsiness_detection\train\yawn_pose_dataset.csv'
df = pd.read_csv(csv_path)
print(f" Dataset loaded from: {csv_path}")
print(" Columns:", df.columns.tolist())

# === Preprocessing ===
X = df.drop('label', axis=1)
y = df['label']

le = LabelEncoder()
y_encoded = le.fit_transform(y)

# === Train-Test Split ===
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# === Model Training ===
model = XGBClassifier(eval_metric='logloss')
model.fit(X_train, y_train)

# === Save Model and Encoder ===
joblib.dump(model, r'C:\projects\mini project glec\drowsiness_detection\models\yawn_model_xgb.pkl')
joblib.dump(le, r'C:\projects\mini project glec\drowsiness_detection\models\yawn_label_encoder.pkl')
print(" Model and label encoder saved.")

# === Predictions ===
y_pred = model.predict(X_test)

# === Performance Metrics ===
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"🔹 Accuracy       : {acc * 100:.2f}%")

# === Confusion Matrix ===
cm = confusion_matrix(y_test, y_pred)
labels = le.classes_

# Annotated Confusion Matrix for Binary Classification
if cm.shape == (2, 2):
    group_names = ['True Negative (TN)', 'False Positive (FP)', 'False Negative (FN)', 'True Positive (TP)']
    group_counts = [f"{value}" for value in cm.flatten()]
    labels_matrix = [f"{name}\n{count}" for name, count in zip(group_names, group_counts)]
    labels_matrix = np.asarray(labels_matrix).reshape(2, 2)
else:
    labels_matrix = cm

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=labels_matrix, fmt='', cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title(" Confusion Matrix - Yawning Detection (XGBoost)")
plt.tight_layout()
plt.show()

# === Classification Report ===
print("\n Classification Report:\n")
print(classification_report(y_test, y_pred, target_names=[str(cls) for cls in labels]))



