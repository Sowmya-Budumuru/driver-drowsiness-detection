# import os
# import joblib

# # Get base directory of the script
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# MODEL_PATH = os.path.join(BASE_DIR, "models", "yawn_model.pkl")

# try:
#     yawn_model = joblib.load(MODEL_PATH)
#     print("✅ Yawn model loaded successfully.")
# except Exception as e:
#     print("❌ Failed to load yawn model:", e)


import pickle

with open("C:/projects/mini project glec/drowsiness_detection/models/yawn_model.pkl", "rb") as f:
    try:
        model = pickle.load(f)
        print("[✅] Yawn model loaded successfully.")
    except Exception as e:
        print("[❌] Error loading model:", e)
