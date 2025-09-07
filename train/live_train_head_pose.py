import cv2
import dlib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)).replace("\\train", ""))

# Load dlib detector and predictor
predictor_path = "C:/projects/mini project glec/drowsiness_detection/models/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

# 3D model reference points
model_points = np.array([
    (0.0, 0.0, 0.0),             # Nose tip
    (0.0, -330.0, -65.0),        # Chin
    (-225.0, 170.0, -135.0),     # Left eye left corner
    (225.0, 170.0, -135.0),      # Right eye right corner
    (-150.0, -150.0, -125.0),    # Left Mouth corner
    (150.0, -150.0, -125.0)      # Right mouth corner
], dtype=np.float64)

# Extract head pose angles from landmarks
def get_head_pose(shape, size):
    image_points = np.array([
        (shape.part(30).x, shape.part(30).y),  # Nose tip
        (shape.part(8).x, shape.part(8).y),    # Chin
        (shape.part(36).x, shape.part(36).y),  # Left eye
        (shape.part(45).x, shape.part(45).y),  # Right eye
        (shape.part(48).x, shape.part(48).y),  # Left mouth
        (shape.part(54).x, shape.part(54).y)   # Right mouth
    ], dtype=np.float64)

    focal_length = size[1]
    center = (size[1] / 2, size[0] / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float64)

    dist_coeffs = np.zeros((4, 1))  # No lens distortion
    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs)

    rotation_mat, _ = cv2.Rodrigues(rotation_vector)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rotation_mat)

    return [angles[0], angles[1], angles[2]]  # pitch, yaw, roll

# Label mapping
label_map = {
    ord('c'): "center",
    ord('l'): "left",
    ord('r'): "right",
    ord('u'): "up",
    ord('d'): "down"
}

X, y = [], []
cap = cv2.VideoCapture(0)
print("👉 Press keys to label: [c]enter, [l]eft, [r]ight, [u]p, [d]own. Press [q] to train & quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Flip for selfie view
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        shape = predictor(gray, face)
        pose = get_head_pose(shape, gray.shape)

        # Draw face rectangle
        cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 2)

        # Show pitch/yaw/roll
        cv2.putText(frame, f"Pitch: {pose[0]:.2f}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, f"Yaw: {pose[1]:.2f}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, f"Roll: {pose[2]:.2f}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Instruction
        cv2.putText(frame, "Label: Press [c/l/r/u/d], [q] to quit", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        key = cv2.waitKey(1) & 0xFF
        if key in label_map:
            X.append(pose)
            y.append(label_map[key])
            print(f"📸 Captured: {label_map[key]} - {pose}")

        elif key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break

    cv2.imshow("Head Pose Labeling", frame)

cap.release()
cv2.destroyAllWindows()

# Train the classifier
print("🧠 Training Random Forest...")
clf = RandomForestClassifier(n_estimators=100, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"✅ Accuracy: {acc * 100:.2f}%")

# Save model
model_path = os.path.join(os.path.dirname(__file__), "..", "models", "head_pose_model.pkl")
joblib.dump(clf, os.path.abspath(model_path))

print("✅ Model saved to: drowsiness_detection/models/head_pose_model.pkl")
