from scipy.spatial import distance
import imutils
from imutils import face_utils
import dlib
import cv2
from pygame import mixer
import joblib
import numpy as np
import time
import pyttsx3
from extract_head_pose_features import extract_head_features

# Initialize alert sound
mixer.init()
sound = mixer.Sound(r'C:\projects\mini project glec\drowsiness_detection\sounds\beep.wav')

engine = pyttsx3.init()

def speak_alert(text):
    engine.say(text)
    engine.runAndWait()

# Eye Aspect Ratio calculation
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# Lip Aspect Ratio calculation for yawning
def lip_aspect_ratio(mouth):
    A = distance.euclidean(mouth[2], mouth[10])  # 51, 59
    B = distance.euclidean(mouth[4], mouth[8])   # 53, 57
    C = distance.euclidean(mouth[0], mouth[6])   # 49, 55
    return (A + B) / (2.0 * C)

# Thresholds and counters
EYE_THRESH = 0.25
MOUTH_THRESH = 0.7
FRAME_CHECK = 20
flag = 0
last_alert_time = 0

# ALERT DISPLAY TIMERS
eye_alert_start = None
mouth_alert_start = None
head_alert_start = None
alert_display_time = 3  # seconds for which alert text stays

# Load models
print("🔁 Loading models...")
detect = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("c:/projects/mini project glec/drowsiness_detection/models/shape_predictor_68_face_landmarks.dat")
yawn_model = joblib.load("c:/projects/mini project glec/drowsiness_detection/models/yawn_model.pkl")
label_encoder = joblib.load("c:/projects/mini project glec/drowsiness_detection/models/yawn_label_encoder.pkl")
head_pose_model = joblib.load("c:/projects/mini project glec/drowsiness_detection/models/head_pose_model.pkl")

# Landmark indices
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["mouth"]

# Start video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break 
    frame = imutils.resize(frame, width=500)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    subjects = detect(gray, 0)

    for subject in subjects:
        shape = predictor(gray, subject)
        shape_np = face_utils.shape_to_np(shape)

        # Eyes
        leftEye = shape_np[lStart:lEnd]
        rightEye = shape_np[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0

        # Draw eye contours
        cv2.drawContours(frame, [cv2.convexHull(leftEye)], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [cv2.convexHull(rightEye)], -1, (0, 255, 0), 1)

        # Mouth
        mouth = shape_np[mStart:mEnd]
        lar = lip_aspect_ratio(mouth)
        cv2.drawContours(frame, [cv2.convexHull(mouth)], -1, (255, 0, 0), 1)

        # Head pose
        try:
            angle_x, angle_y, angle_z = extract_head_features(shape_np)
            head_features = [[angle_x, angle_y, angle_z]]
            head_pose = head_pose_model.predict(head_features)[0]
        except Exception as e:
            print("[WARN] Head pose detection failed:", str(e))
            head_pose = 0  # default no head tilt

        # Current time for alert timing
        current_time = time.time()

        # EYE alert
        if ear < EYE_THRESH:
            flag += 1
            if flag >= FRAME_CHECK:
                if eye_alert_start is None:
                    eye_alert_start = current_time
                if current_time - eye_alert_start < alert_display_time: 
                    cv2.putText(frame, "DROWSINESS DETECTED!", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    if current_time - last_alert_time > 5:
                        sound.play()
                        speak_alert("Warning! You seem drowsy. Please stay alert.")
                        last_alert_time = current_time
                else:
                    eye_alert_start = None
                    flag = 0
        else:
            flag = 0
            eye_alert_start = None

        # MOUTH alert (yawning)
        if lar > MOUTH_THRESH:
            if mouth_alert_start is None:
                mouth_alert_start = current_time
            if current_time - mouth_alert_start < alert_display_time:
                cv2.putText(frame, "YAWNING DETECTED!", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
                if current_time - last_alert_time > 5:
                    sound.play()
                    speak_alert("You seem tired. Please take a break.")
                    last_alert_time = current_time
        else:
            mouth_alert_start = None


        # Head pose detection
        # head_features = extract_head_features(shape)
        # head_pose = head_model.predict([head_features])[0]
        
        # HEAD TILT alert
        if head_pose == 1:
            if head_alert_start is None:
                head_alert_start = current_time
            if current_time - head_alert_start < alert_display_time:
                cv2.putText(frame, "HEAD TILT DETECTED!", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                if current_time - last_alert_time > 5:
                    sound.play()
                    speak_alert("Head tilted! Stay alert.")
                    last_alert_time = current_time
        else:
            head_alert_start = None

    # Show frame
    cv2.imshow("Driver Drowsiness Detection", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
