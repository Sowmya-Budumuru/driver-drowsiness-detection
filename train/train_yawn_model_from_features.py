import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report

# ✅ Absolute path to your labeled dataset (CSV file)
csv_path = "C:/projects/mini project glec/drowsiness_detection/train/yawn_features_dataset.csv"

# ✅ Load your dataset
df = pd.read_csv(csv_path)

# ✅ Feature columns - must match exactly how you're extracting in main.py
feature_columns = ['mar', 'mouth_width', 'mouth_height', 'lip_distance']
target_column = 'label'

# ✅ Split features and labels
X = df[feature_columns]
y = df[target_column]

# ✅ Train-test split (80-20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Train Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ✅ Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\n✅ Accuracy: {acc*100:.2f}%")
print("📊 Classification Report:\n", classification_report(y_test, y_pred))

# ✅ Cross-validation
cv_scores = cross_val_score(model, X, y, cv=5)
print(f"🔁 5-Fold CV Accuracy: {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")

# ✅ Save model to models/ directory
model_path = "C:/projects/mini project glec/drowsiness_detection/models/yawn_model.pkl"
joblib.dump(model, model_path)
print(f"✅ Model saved to: {model_path}")
