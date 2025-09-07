import os
import cv2
import dlib
import pandas as pd
from imutils import face_utils
from scipy.spatial import distance as dist

# === Setup Paths ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIRS = [
    os.path.normpath(os.path.join(BASE_DIR, "..", "datasets", "head_pose", "head_pose_masks")),
    os.path.normpath(os.path.join(BASE_DIR, "..", "datasets", "head_pose", "faces_0")),
]
PREDICTOR_PATH = os.path.normpath(os.path.join(BASE_DIR, "..", "models", "shape_predictor_68_face_landmarks.dat"))
OUTPUT_CSV = os.path.normpath(os.path.join(BASE_DIR, "head_pose_dataset.csv"))

# === Load Dlib ===
if not os.path.exists(PREDICTOR_PATH):
    raise FileNotFoundError(f"Facial landmark model not found at: {PREDICTOR_PATH}")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)

# === Helper Function ===
def extract_features(shape):
    eye_left = shape[36]
    eye_right = shape[45]
    nose = shape[30]
    chin = shape[8]

    eye_dist = dist.euclidean(eye_left, eye_right)
    nose_chin_dist = dist.euclidean(nose, chin)

    # Simple logic: if nose deviates too far horizontally from the center, label as tilted
    center_x = (eye_left[0] + eye_right[0]) / 2
    is_tilted = 1 if abs(nose[0] - center_x) > 10 else 0

    return [eye_dist, nose_chin_dist, is_tilted]

# === Process Images ===
rows = []
count = 0

for dir_path in IMAGE_DIRS:
    if not os.path.exists(dir_path):
        print(f"[⚠️] Skipping missing folder: {dir_path}")
        continue

    for fname in os.listdir(dir_path):
        if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        img_path = os.path.join(dir_path, fname)
        img = cv2.imread(img_path)
        if img is None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 0)
        if len(rects) == 0:
            continue

        shape = predictor(gray, rects[0])
        shape = face_utils.shape_to_np(shape)
        features = extract_features(shape)
        rows.append(features)
        count += 1

# === Save to CSV ===
df = pd.DataFrame(rows, columns=["eye_dist", "nose_chin_dist", "label"])
df.to_csv(OUTPUT_CSV, index=False)

# === Summary ===
print(f"✅ Saved to: {OUTPUT_CSV}")
print(f"📸 Total images processed: {count}")
print("📊 Label Distribution:")
print(df["label"].value_counts())
