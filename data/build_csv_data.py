import os
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

label_path = BASE_DIR / "MVSA" / "labels_master.csv"
df = pd.read_csv(label_path)

data_dir = BASE_DIR / "MVSA" / "data"

df["text_path"] = df["id"].apply(lambda x: os.path.join(data_dir, f"{x}.txt"))
df["image_path"] = df["id"].apply(lambda x: os.path.join(data_dir, f"{x}.jpg"))

df.to_csv("MVSA.csv", index=False)