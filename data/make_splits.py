import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

mvsa_path = ROOT / "data" / "MVSA" / "MVSA.csv"
labels_path = ROOT / "data" / "MVSA" / "labels_master.csv"

# load full MVSA metadata; keep only paths to avoid duplicate label columns
mvsa = pd.read_csv(mvsa_path)[["id", "text_path", "image_path"]]

# load curated labels (id, text_label, image_label, combined_label)
labels = pd.read_csv(labels_path)

# keep only rows whose id appears in labels_master and attach all three labels
df = mvsa.merge(
    labels[["id", "text_label", "image_label", "combined_label"]],
    on="id",
    how="inner",
)

# drop rows with missing files (just in case)
df = df[df["text_path"].notna() & df["image_path"].notna()]

# choose the target for this split: "text_label", "image_label", or "combined_label"
TARGET = "image_label"



train_df, temp_df = train_test_split(df, test_size=0.2, stratify=df[TARGET], random_state=42)
valid_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df[TARGET], random_state=42)

out = ROOT / "data" / "MVSA" / "splits"
out.mkdir(parents=True, exist_ok=True)
train_df.to_csv(out / "train.csv", index=False)
valid_df.to_csv(out / "valid.csv", index=False)
test_df.to_csv(out / "test.csv", index=False)
print("Splits written to", out)
