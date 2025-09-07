import bz2

source = r"C:\projects\shape_predictor_68_face_landmarks.dat.bz2"
destination = r"c:\projects\mini project glec\drowsiness_detection\models\shape_predictor_68_face_landmarks.dat"

with bz2.BZ2File(source, "rb") as fr, open(destination, "wb") as fw:
    fw.write(fr.read())

print("✅ Extracted successfully.")
