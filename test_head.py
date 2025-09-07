# # import joblib

# # try:
# #     model = joblib.load("models/head_pose_model.pkl")
# #     print("✅ Head pose model loaded successfully.")
# # except Exception as e:
# #     print("❌ Failed to load head pose model:")
# #     print(e)

# # import dlib
# # import os

# # path = os.path.abspath("drowsiness_detection/models/shape_predictor_68_face_landmarks.dat")
# # print("Loading from:", path)

# # predictor = dlib.shape_predictor(path)
# # print("Loaded successfully!")


# # import dlib

# # path = r"C:\projects\mini project glec\drowsiness_detection\shape_predictor_68_face_landmarks.dat"
# # predictor = dlib.shape_predictor(path)
# # print("✅ Predictor loaded successfully.")


# import pandas as pd

# df = pd.read_csv(r"c:\projects\mini project glec\drowsiness_detection\train\head_pose_dataset.csv")

# print("🔢 Shape before dropping NaNs:", df.shape)
# print("🕳️ Missing values:\n", df.isnull().sum())

# df_clean = df.dropna()
# print("✅ Shape after dropping NaNs:", df_clean.shape)


import cv2
import dlib
import numpy as np
import os
import pandas as pd

# === Initialize ===
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("c:/projects/mini project glec/drowsiness_detection/models/shape_predictor_68_face_landmarks.dat")

# === CSV Path ===
csv_path = os.path.join(os.path.dirname(__file__), "head_pose_dataset.csv")

# === Open Webcam ===
cap = cv2.VideoCapture(0)
print("📷 Starting webcam. Press 'l', 'r', or 's' to label head pose.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)

        # Eye corners: 36 (left), 45 (right)
        x1, y1 = landmarks.part(36).x, landmarks.part(36).y
        x2, y2 = landmarks.part(45).x, landmarks.part(45).y
        eye_dist = np.linalg.norm(np.array([x1, y1]) - np.array([x2, y2]))

        # Nose (33), Chin (8)
        x3, y3 = landmarks.part(33).x, landmarks.part(33).y
        x4, y4 = landmarks.part(8).x, landmarks.part(8).y
        nose_chin_dist = np.linalg.norm(np.array([x3, y3]) - np.array([x4, y4]))

        # Draw landmarks (optional)
        cv2.circle(frame, (x1, y1), 2, (255, 0, 0), -1)
        cv2.circle(frame, (x2, y2), 2, (255, 0, 0), -1)
        cv2.circle(frame, (x3, y3), 2, (0, 255, 0), -1)
        cv2.circle(frame, (x4, y4), 2, (0, 255, 0), -1)

        cv2.putText(frame, "Press l/r/s to label", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow("Head Pose Labeling", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('l'):
            label = "left"
        elif key == ord('r'):
            label = "right"
        elif key == ord('s'):
            label = "straight"
        elif key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            print("👋 Exiting...")
            exit(0)
        else:
            continue  # skip if no valid key

        # === Save to CSV ===
        row = pd.DataFrame([[eye_dist, nose_chin_dist, label]], columns=["eye_dist", "nose_chin_dist", "label"])
        row.to_csv(csv_path, mode='a', header=not os.path.exists(csv_path), index=False)
        print(f"✅ Saved: {eye_dist:.2f}, {nose_chin_dist:.2f}, {label}")

    cv2.imshow("Head Pose Labeling", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# === Cleanup ===
cap.release()
cv2.destroyAllWindows()
