# extract_yawn_features.py

import os
import pandas as pd
import xml.etree.ElementTree as ET

# === Paths ===
ANNOT_DIR = 'datasets/yawn/train/labels'  # Path to Roboflow annotation XMLs

# === Process Annotations ===
data = []

for file in os.listdir(ANNOT_DIR):
    if not file.endswith('.xml'):
        continue
    
    path = os.path.join(ANNOT_DIR, file)
    tree = ET.parse(path)
    root = tree.getroot()
    
    # Dummy logic: assume files with 'yawn' in name = label 1
    label = 1 if 'yawn' in file.lower() else 0

    # Use dummy feature for now; proper LAR requires facial landmarks
    lar = 0.75 if label == 1 else 0.45
    data.append({'lar': lar, 'label': label})

# === Save CSV ===
df = pd.DataFrame(data)
df.to_csv("yawn_pose_dataset.csv", index=False)
print("[✅] Saved yawn_pose_dataset.csv")



# import cv2
# import os
# import pandas as pd
# import numpy as np

# # Updated paths
# base_dir = r"c:\projects\mini project glec\drowsiness_detection\datasets\yawning"
# output_csv = r"c:\projects\mini project glec\drowsiness_detection\train\yawn_pose_dataset.csv"

# processed, skipped = 0, 0
# data = []

# def extract_features(image_path, label):
#     image = cv2.imread(image_path)
#     if image is None:
#         print(f"⚠️ Could not read: {image_path}")
#         return None

#     try:
#         image = cv2.resize(image, (100, 60))
#         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#         _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
#         contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#         if not contours:
#             print(f"❌ No contours detected: {image_path}")
#             return None

#         largest = max(contours, key=cv2.contourArea)
#         x, y, w, h = cv2.boundingRect(largest)

#         lar = h / w if w else 0
#         mouth_area = cv2.contourArea(largest)
#         mean_intensity = np.mean(gray)
#         aspect_ratio = image.shape[1] / image.shape[0]
#         mouth_height = h

#         return {
#             "lar": lar,
#             "mouth_area": mouth_area,
#             "mean_intensity": mean_intensity,
#             "aspect_ratio": aspect_ratio,
#             "mouth_height": mouth_height,
#             "label": label
#         }
#     except Exception as e:
#         print(f"❌ Error: {image_path}: {str(e)}")
#         return None

# # ✅ Loop through yawn and noyawn subfolders
# for label_folder in ['yawn', 'noyawn']:
#     full_path = os.path.join(base_dir, label_folder)
#     if not os.path.exists(full_path):
#         print(f"🚫 Folder not found: {full_path}")
#         continue

#     for fname in os.listdir(full_path):
#         if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
#             img_path = os.path.join(full_path, fname)
#             feat = extract_features(img_path, label_folder)
#             if feat:
#                 data.append(feat)
#                 processed += 1
#             else:
#                 skipped += 1

# # ✅ Save
# if data:
#     df = pd.DataFrame(data)
#     df.to_csv(output_csv, index=False)
#     print(f"\n✅ Feature CSV saved to: {output_csv}")
#     print(f"📊 Extracted: {processed} | Skipped: {skipped}")
# else:
#     print("🚫 No features extracted.")
