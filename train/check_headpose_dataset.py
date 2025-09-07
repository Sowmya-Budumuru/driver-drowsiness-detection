import pandas as pd
import os

csv_path = os.path.join(os.path.dirname(__file__), "head_pose_dataset.csv")

if not os.path.exists(csv_path):
    print("❌ Dataset not found:", csv_path)
else:
    df = pd.read_csv(csv_path)
    print("✅ Dataset found:", csv_path)
    print("📊 Head Pose Label Distribution:")
    print(df["label"].value_counts())
    print("\n🧪 Sample Rows:")
    print(df.head())
