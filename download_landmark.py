import urllib.request
import bz2
import os

url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
bz2_path = "shape_predictor_68_face_landmarks.dat.bz2"
dat_path = "shape_predictor_68_face_landmarks.dat"

print("[INFO] Downloading...")
urllib.request.urlretrieve(url, bz2_path)

print("[INFO] Extracting...")
with bz2.BZ2File(bz2_path) as fr, open(dat_path, 'wb') as fw:
    fw.write(fr.read())

print("[✅] Done. Move this .dat file to your 'models' folder.")
