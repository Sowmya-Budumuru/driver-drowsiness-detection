import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os
import sys

# === Absolute Paths ===
BASE_DIR = r"C:\projects\mini project glec\drowsiness_detection"
csv_path = os.path.join(BASE_DIR, "train", "head_pose_dataset.csv")
model_path = os.path.join(BASE_DIR, "models", "head_pose_model.pkl")
label_encoder_path = os.path.join(BASE_DIR, "models", "head_pose_label_encoder.pkl")

# === Check file existence ===
if not os.path.exists(csv_path):
    print(f"❌ Dataset not found at: {csv_path}")
    sys.exit(1)

# === Load Data ===
df = pd.read_csv(csv_path)
df.dropna(inplace=True)

if df.empty:
    print("❌ Dataset is empty after dropping NaNs.")
    sys.exit(1)

# === Encode labels ===
le = LabelEncoder()
df["label_encoded"] = le.fit_transform(df["label"])

# === Features and labels ===
X = df[["eye_dist", "nose_chin_dist"]]
y = df["label_encoded"]

# === Train-Test Split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Train Random Forest ===
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# === Evaluate ===
y_pred = clf.predict(X_test)
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))
print(f"✅ Accuracy: {accuracy_score(y_test, y_pred):.2f}")

# === Save model and label encoder ===
os.makedirs(os.path.dirname(model_path), exist_ok=True)
joblib.dump(clf, model_path)
joblib.dump(le, label_encoder_path)

print(f"\n💾 Model saved to: {model_path}")
print(f"💾 Label Encoder saved to: {label_encoder_path}")
