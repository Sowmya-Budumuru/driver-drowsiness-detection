import cv2
import dlib
import os
import csv
from scipy.spatial import distance as dist
from imutils import face_utils

# === Setup Paths ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
CSV_FILE = os.path.join(BASE_DIR, "yawn_pose_dataset.csv")

predictor_path = os.path.join(MODELS_DIR, "shape_predictor_68_face_landmarks.dat")
if not os.path.exists(predictor_path):
    raise FileNotFoundError("Landmark predictor missing!")

# === Init Dlib ===
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

# === Feature extractor ===
def compute_lar(mouth):
    A = dist.euclidean(mouth[13], mouth[19])
    B = dist.euclidean(mouth[14], mouth[18])
    C = dist.euclidean(mouth[15], mouth[17])
    D = dist.euclidean(mouth[12], mouth[16])
    return (A + B + C) / (3.0 * D)

# === CSV Write Setup ===
header = ["LAR", "label"]  # label: 1 (yawn), 0 (not yawn)
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

# === Capture ===
cap = cv2.VideoCapture(0)
print("[INFO] Press 'y' to record yawn, 'n' for no yawn, 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray)

    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        mouth = shape[mStart:mEnd]

        lar = compute_lar(mouth)

        cv2.drawContours(frame, [cv2.convexHull(mouth)], -1, (255, 0, 0), 1)
        cv2.putText(frame, f"LAR: {lar:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow("Yawn Data Collector", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("y"):
        with open(CSV_FILE, "a", newline="") as f:
            csv.writer(f).writerow([lar, 1])
        print("[+] Yawn data saved")
    elif key == ord("n"):
        with open(CSV_FILE, "a", newline="") as f:
            csv.writer(f).writerow([lar, 0])
        print("[+] No-yawn data saved")
    elif key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
