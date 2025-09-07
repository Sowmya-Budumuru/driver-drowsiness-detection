# **Driver Drowsiness Detection System**



A real-time Driver Drowsiness Detection System built using Python, OpenCV, and Machine Learning.

The system monitors the driver’s eyes, mouth, and head movements to detect signs of fatigue and trigger timely alerts, helping reduce road accidents caused by drowsy driving.



### **🚀 Features**



Eye Closure Detection → Uses Eye Aspect Ratio (EAR) to detect if eyes are closed.



Yawning Detection (ML-based) → Detects yawns using Lip Aspect Ratio (LAR) and ML models.



Head Pose Detection → Detects head tilting/nodding with ML-based classification.



Adaptive Alerts System:



First alert → Beep sound 🔊



Second alert → Flashing lights ⚡



Third alert → (Future extension) Automatic speed reduction via IoT 🚗



Voice Assistant Alerts → Speaks warnings like “You seem tired, please take a break!”



### **🛠️ Tech Stack**



Language: Python



Libraries: OpenCV, dlib, NumPy, imutils, scikit-learn, XGBoost, pyttsx3



Models: Random Forest \& XGBoost for yawning and head pose detection



📂 Project Structure

drowsiness\_detection/

│── main.py                 # Main application  

│── yawn\_model.pkl          # ML model for yawning detection  

│── head\_pose\_model.pkl     # ML model for head pose detection  

│── requirements.txt        # Dependencies  

│── README.md               # Project documentation  

│── datasets/               # Training datasets  

└── utils/                  # Helper functions  



### **⚡ Installation**



Clone the repo:



git clone https://github.com/sowmya-budumuru/driver-drowsiness-detection.git

cd driver-drowsiness-detection





Install dependencies:



pip install -r requirements.txt





Run the app:



python main.py



### **🎯 Future Improvements**



Integrate IoT system for speed control via OBD-II.



Improve ML models with larger datasets.



Add mobile app integration for notifications.



### **📜 License**



This project is licensed under the MIT License – feel free to use and modify it.

