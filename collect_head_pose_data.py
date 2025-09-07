import cv2
import dlib
import csv
import os
from imutils import face_utils
from scipy.spatial import distance as dist

# === Paths ===
predictor_path = r"c:/Users/janu/OneDrive/Desktop/mini project glec/drowsiness_detection/models/shape_predictor_68_face_landmarks.dat"
csv_path = "head_pose_dataset.csv"

# === Initialize ===
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

# === Feature extractor ===
def extract_head_features(shape):
    nose = shape[33]
    chin = shape[8]
    left_eye = shape[36:42].mean(axis=0)
    right_eye = shape[42:48].mean(axis=0)

    eye_dist = dist.euclidean(left_eye, right_eye)
    nose_chin_dist = dist.euclidean(nose, chin)

    return [eye_dist, nose_chin_dist]

# === Setup CSV file ===
if not os.path.exists(csv_path):
    with open(csv_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["eye_dist", "nose_chin_dist", "label"])

# === Start webcam ===
cap = cv2.VideoCapture(0)
print("[INFO] Press 'f'=Forward, 'd'=Down, 'l'=Left, 'r'=Right, 'q'=Quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray)

    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        for (x, y) in shape:
            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

        features = extract_head_features(shape)
        cv2.putText(frame, "Press f/d/l/r to label, q to quit", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow("Head Pose Data Collector", frame)
    key = cv2.waitKey(1) & 0xFF

    label = None
    if key == ord("f"):
        label = "Forward"
    elif key == ord("d"):
        label = "Down"
    elif key == ord("l"):
        label = "Left"
    elif key == ord("r"):
        label = "Right"
    elif key == ord("q"):
        break

    if label and rects:
        with open(csv_path, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(features + [label])
        print(f"[INFO] Saved label: {label}")

cap.release()
cv2.destroyAllWindows()
