import os
import xml.etree.ElementTree as ET

ANNOT_DIR = r"drowsiness_detection/datasets/yawn/train"

labels = set()

for file in os.listdir(ANNOT_DIR):
    if file.endswith(".xml"):
        tree = ET.parse(os.path.join(ANNOT_DIR, file))
        root = tree.getroot()
        for obj in root.findall("object"):
            label = obj.find("name").text.strip().lower()
            labels.add(label)

print("✅ Unique labels found in your dataset:", labels)
