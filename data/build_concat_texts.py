# build_concat_texts.py
"""
Build combined text files for training by concatenating tweet text, image caption,
and optional OCR text. Rows without a caption are skipped.

Example:
    python data/build_concat_texts.py \
        --base_dir data/MVSA/splits \
        --splits train,valid,test \
        --captions_csv data/descriptions/captions.csv \
        --ocr_csv data/descriptions/ocr.csv \
        --out_dir data/MVSA/combined_text

    python data/build_concat_texts.py \
        --base_dir data/MVSA/splits \
        --splits train,valid,test \
        --captions_csv data/descriptions/captions.csv \
        --ocr_csv data/descriptions/ocr.csv \
        --max_caption_chars 150 \
        --max_ocr_chars 150 \
        --out_dir data/MVSA/combined_text

"""

import argparse
from pathlib import Path
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description="Concatenate tweet text, caption, and OCR into new text files/CSVs.")
    p.add_argument(
        "--base_dir",
        default=str(Path("data") / "MVSA" / "splits"),
        help="Directory containing split CSVs (train.csv, valid.csv, test.csv).",
    )
    p.add_argument(
        "--splits",
        default="train,valid,test",
        help="Comma-separated split names to process (files must be <split>.csv under base_dir).",
    )
    p.add_argument("--captions_csv", required=True, help="Master captions CSV with columns: id, caption.")
    p.add_argument("--ocr_csv", help="Optional master OCR CSV with columns: id, ocr_text.")
    p.add_argument(
        "--out_dir",
        default=str(Path("data") / "MVSA" / "combined_text"),
        help="Directory to write combined text files and output CSVs.",
    )
    p.add_argument("--joiner", default=" [SEP] ", help="String used between parts of the combined text.")
    p.add_argument("--max_caption_chars", type=int, default=None, help="Optional max characters for caption.")
    p.add_argument("--max_ocr_chars", type=int, default=None, help="Optional max characters for OCR text.")
    return p.parse_args()


def load_map(csv_path: str, key: str, value: str) -> dict:
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=[key, value])
    return dict(zip(df[key], df[value]))


def main():
    args = parse_args()
    cap_map = load_map(args.captions_csv, "id", "caption")
    ocr_map = load_map(args.ocr_csv, "id", "ocr_text") if args.ocr_csv else {}

    out_root = Path(args.out_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    results = {}

    for split in splits:
        base_path = Path(args.base_dir) / f"{split}.csv"
        if not base_path.exists():
            print(f"[WARN] Base CSV not found for split '{split}': {base_path}")
            continue

        base_df = pd.read_csv(base_path)
        out_text_dir = out_root / "texts" / split
        out_text_dir.mkdir(parents=True, exist_ok=True)

        # Optionally drop rows where text/image labels disagree
        if "text_label" in base_df.columns and "image_label" in base_df.columns:
            before = len(base_df)
            base_df = base_df[base_df["text_label"] == base_df["image_label"]].copy()
            after = len(base_df)
            if before != after:
                print(f"[{split}] filtered mismatched text/image labels: {before} -> {after}")
        else:
            print(f"[{split}] [WARN] text_label/image_label columns not found; skipping agreement filter.")

        records = []
        skipped = 0
        for _, row in base_df.iterrows():
            img_id = row.get("id")
            caption = cap_map.get(img_id)
            if not caption:
                skipped += 1
                continue
            if args.max_caption_chars is not None:
                caption = caption[: args.max_caption_chars]

            try:
                tweet_text = Path(row["text_path"]).read_text(encoding="utf-8").strip()
            except Exception as e:
                skipped += 1
                print(f"[WARN] Skipping id={img_id} due to text read error: {e}")
                continue

            combined = f"{tweet_text}{args.joiner}{caption}"
            ocr_text = ocr_map.get(img_id, "")
            if args.max_ocr_chars is not None:
                ocr_text = ocr_text[: args.max_ocr_chars]
            if ocr_text:
                combined = f"{combined}{args.joiner}{ocr_text}"

            out_path = out_text_dir / f"{img_id}.txt"
            out_path.write_text(combined, encoding="utf-8")

            rec = row.to_dict()
            rec["text_path"] = str(out_path.resolve())
            records.append(rec)

        out_csv = out_root / f"{split}_combined.csv"
        pd.DataFrame(records).to_csv(out_csv, index=False)

        results[split] = {"rows": len(records), "out_csv": str(out_csv), "skipped": skipped}
        print(f"[{split}] wrote {len(records)} rows to {out_csv}")
        print(f"[{split}] combined text files in {out_text_dir}")
        if skipped:
            print(f"[{split}] [WARN] Skipped {skipped} rows (missing caption or read errors)")

    if not results:
        print("No splits processed. Check paths and split names.")
    else:
        print("Done. " \
        # "Summary:", results
        )


if __name__ == "__main__":
    main()
