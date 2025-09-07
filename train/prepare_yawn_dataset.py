import os
import cv2
import pandas as pd

# === Setup Paths ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
YAWN_DIR = os.path.join(BASE_DIR, "..", "datasets", "yawning")
OUTPUT_CSV = os.path.join(BASE_DIR, "yawn_pose_dataset.csv")

# === Data Preparation ===
rows = []

for label_name in ["yawn", "no yawn"]:
    folder_path = os.path.join(YAWN_DIR, label_name)
    label = 1 if label_name == "yawn" else 0

    if not os.path.exists(folder_path):
        print(f"[SKIP] Missing folder: {folder_path}")
        continue

    for file in os.listdir(folder_path):
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            img_path = os.path.join(folder_path, file)
            img = cv2.imread(img_path)

            if img is None:
                print(f"[SKIP] Cannot read: {img_path}")
                continue

            h, w = img.shape[:2]
            lar = h / w  # Pseudo-LAR: height-to-width ratio
            rows.append([lar, label])

# === Save CSV ===
df = pd.DataFrame(rows, columns=["LAR", "label"])
df.to_csv(OUTPUT_CSV, index=False)

print(f"✅ Saved dataset to: {OUTPUT_CSV}")
print("Label distribution:")
print(df["label"].value_counts())
