# generate_captions.py
"""
Scan a directory of images, generate BLIP captions and optional OCR text, and write two CSV files:

    captions.csv    with columns: id, caption
    ocr.csv         with columns: id, ocr_text  (if OCR is enabled)

Typical usage:

    # Generate BOTH captions and OCR
    python data/generate_captions.py \
        --images_dir /path/to/images \
        --out_dir /path/to/out

    # Caption ONLY (no OCR)
    python data/generate_captions.py \
        --images_dir /path/to/images \
        --out_dir /path/to/out \
        --no_ocr
    
    python data/generate_captions.py \
        --no_ocr

    # OCR ONLY (skip captioning)
    python data/generate_captions.py \
        --images_dir /path/to/images \
        --out_dir /path/to/out \
        --skip_captions

    python data/generate_captions.py \
        --skip_captions
"""

import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import BlipProcessor, BlipForConditionalGeneration

# Silence MPS pin_memory warning from torch DataLoader (raised inside EasyOCR)
warnings.filterwarnings(
    "ignore",
    message=".*pin_memory.*not supported on MPS.*",
    category=UserWarning,
    module="torch.utils.data.dataloader",
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "MVSA" / "data"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate BLIP captions (and optional OCR text) into separate CSVs."
    )
    parser.add_argument(
        "--images_dir",
        default=str(DATA_DIR),
        help="Directory containing images to caption/OCR.",
    )
    parser.add_argument(
        "--caption_model",
        default="Salesforce/blip-image-captioning-base",
        help="Image captioning model name or path (HF).",
    )
    parser.add_argument(
        "--out_dir",
        default=str(BASE_DIR / "MVSA" / "descriptions"),
        help="Directory where captions.csv and ocr.csv will be written.",
    )
    parser.add_argument("--max_new_tokens", type=int, default=40, help="Max tokens for caption generation.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for caption generation.")
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.1,
        help="Repetition penalty for caption generation.",
    )
    parser.add_argument(
        "--no_repeat_ngram_size",
        type=int,
        default=2,
        help="Size of n-grams that should not be repeated (set 0 to disable).",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.95,
        help="Top-p (nucleus) sampling parameter for caption generation.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Softmax temperature for caption generation.",
    )
    parser.add_argument(
        "--skip_captions",
        action="store_true",
        help="Skip BLIP caption generation (only run OCR if enabled).",
    )
    parser.add_argument(
        "--no_ocr",
        action="store_true",
        help="Disable OCR (PaddleOCR). Only generate captions.",
    )
    parser.add_argument(
        "--ocr_langs",
        default="en",
        help="Comma-separated language codes for OCR (default: 'en').",
    )
    parser.add_argument(
        "--ocr_engine",
        choices=["easyocr", "paddle"],
        default="easyocr",
        help="OCR engine to use (default: easyocr, which supports MPS).",
    )
    return parser.parse_args()


def load_caption_model(name: str):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    processor = BlipProcessor.from_pretrained(name)
    model = BlipForConditionalGeneration.from_pretrained(name).to(device)
    model.eval()
    return processor, model, device


def load_ocr_reader(langs: List[str], engine: str = "easyocr"):
    """Load an OCR reader (easyocr or paddle). Returns None if unavailable."""
    lang = langs[0] if langs else "en"
    if engine == "easyocr":
        try:
            import easyocr
        except ImportError:
            print("[WARN] easyocr not installed. OCR will be disabled.")
            return None
        use_mps = torch.backends.mps.is_available() and not torch.cuda.is_available()
        use_gpu = torch.cuda.is_available()
        gpu_flag = True if (use_gpu or use_mps) else False
        device_str = "cuda" if use_gpu else ("mps" if use_mps else "cpu")
        print(f"[INFO] Initializing EasyOCR with lang='{lang}' (device={device_str}, gpu_flag={gpu_flag})")
        try:
            reader = easyocr.Reader([lang], gpu=gpu_flag)
        except Exception as e:
            print(f"[WARN] EasyOCR initialization failed ({e}); OCR disabled.")
            return None
        return reader
    else:
        try:
            from paddleocr import PaddleOCR
        except ImportError:
            print("[WARN] PaddleOCR not installed. OCR will be disabled.")
            return None
        print(f"[INFO] Initializing PaddleOCR with lang='{lang}'")
        try:
            reader = PaddleOCR(lang=lang)
        except Exception as e:
            print(f"[WARN] PaddleOCR initialization failed ({e}); OCR disabled.")
            return None
        return reader


def run_ocr(path: str, reader) -> str:
    """Run OCR on the image at `path` and return a single concatenated string."""
    if reader is None:
        return ""

    try:
        if hasattr(reader, "ocr"):
            results = reader.ocr(path, cls=True)  # PaddleOCR
        else:
            # EasyOCR returns list of [bbox, text, conf]
            results = reader.readtext(path, detail=1, paragraph=False)
    except Exception as e:
        print(f"[WARN] OCR failed for {path}: {e}")
        return ""

    if not results:
        return ""

    texts = []
    # Handle both PaddleOCR (nested pages) and EasyOCR (flat list)
    if hasattr(reader, "ocr"):
        for page in results:
            for line in page:
                texts.append(line[1][0])
    else:
        for _, text, _ in results:
            texts.append(text)

    text = " ".join(texts).strip()
    return text


def caption_batch(
    paths: List[str],
    processor,
    model,
    device,
    max_new_tokens: int,
    repetition_penalty: float,
    no_repeat_ngram_size: int,
    top_p: float,
    temperature: float,
) -> Tuple[List[Optional[str]], List[Tuple[int, Exception]]]:
    """
    Caption a batch of image paths.

    Returns
    -------
    captions : list of str | None
        Caption for each input path; None if loading or captioning failed.
    errors : list of (idx, Exception)
        Any image-load errors with the local index in `paths`.
    """
    images: List[Image.Image] = []
    ok_indices: List[int] = []
    errors: List[Tuple[int, Exception]] = []

    for idx, p in enumerate(paths):
        try:
            images.append(Image.open(p).convert("RGB"))
            ok_indices.append(idx)
        except Exception as e:
            errors.append((idx, e))

    captions: List[Optional[str]] = [None] * len(paths)
    if images:
        inputs = processor(
            images=images,
            return_tensors="pt",
            padding=True,
        ).to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                do_sample=True,
                top_p=top_p,
                temperature=temperature,
            )
        decoded = processor.batch_decode(out, skip_special_tokens=True)
        for local_idx, caption in enumerate(decoded):
            captions[ok_indices[local_idx]] = caption.strip()

    return captions, errors


def find_images(root: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
    return sorted(
        p for p in root.rglob("*")
        if p.suffix.lower() in exts and p.is_file()
    )


def process_images(
    image_paths: List[Path],
    out_root: Path,
    processor,
    model,
    device,
    max_new_tokens: int,
    ocr_reader,
    generate_captions: bool,
    generate_ocr: bool,
    batch_size: int,
    repetition_penalty: float,
    no_repeat_ngram_size: int,
    top_p: float,
    temperature: float,
) -> None:
    caption_rows: List[Dict[str, str]] = []
    ocr_rows: List[Dict[str, str]] = []
    skipped = 0

    for start in tqdm(range(0, len(image_paths), batch_size), desc="Images"):
        batch_paths = image_paths[start : start + batch_size]
        batch_paths_str = [str(p) for p in batch_paths]

        if generate_captions:
            captions, errors = caption_batch(
                batch_paths_str,
                processor,
                model,
                device,
                max_new_tokens,
                repetition_penalty,
                no_repeat_ngram_size,
                top_p,
                temperature,
            )
            if errors:
                for idx, e in errors:
                    skipped += 1
                    print(f"[WARN] Skipping caption for id={batch_paths[idx].stem} due to image load error: {e}")
            error_indices = {idx for idx, _ in errors}
        else:
            captions = [None] * len(batch_paths)
            error_indices = set()

        for local_idx, img_path in enumerate(batch_paths):
            img_id = img_path.stem

            if generate_captions:
                if local_idx not in error_indices:
                    caption_text = captions[local_idx]
                    if caption_text is not None:
                        caption_rows.append({"id": img_id, "caption": caption_text})
                    else:
                        print(f"[WARN] No caption produced for id={img_id}")

            if generate_ocr:
                ocr_text = run_ocr(str(img_path), ocr_reader)
                ocr_rows.append({"id": img_id, "ocr_text": ocr_text})

    if generate_captions:
        captions_path = out_root / "captions.csv"
        pd.DataFrame(caption_rows, columns=["id", "caption"]).to_csv(captions_path, index=False)

    if generate_ocr:
        ocr_path = out_root / "ocr.csv"
        pd.DataFrame(ocr_rows, columns=["id", "ocr_text"]).to_csv(ocr_path, index=False)

    if skipped:
        print(f"[WARN] Skipped {skipped} caption(s) due to image load errors")


def main():
    args = parse_args()

    images_dir = Path(args.images_dir).resolve()
    out_root = Path(args.out_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    image_paths = find_images(images_dir)
    if not image_paths:
        print(f"[WARN] No images found in {images_dir}")
        return

    # Set up BLIP captioning model unless we're skipping captions entirely
    if args.skip_captions:
        processor = model = device = None
        print("[INFO] Skipping BLIP caption generation (--skip_captions).")
    else:
        processor, model, device = load_caption_model(args.caption_model)

    # Set up OCR reader unless OCR is disabled
    if args.no_ocr:
        ocr_reader = None
        print("[INFO] OCR disabled via --no_ocr.")
    else:
        langs = [lang.strip() for lang in args.ocr_langs.split(",") if lang.strip()]
        ocr_reader = load_ocr_reader(langs, engine=args.ocr_engine)

    generate_captions = not args.skip_captions
    generate_ocr = ocr_reader is not None and not args.no_ocr

    process_images(
        image_paths=image_paths,
        out_root=out_root,
        processor=processor,
        model=model,
        device=device,
        max_new_tokens=args.max_new_tokens,
        ocr_reader=ocr_reader,
        generate_captions=generate_captions,
        generate_ocr=generate_ocr,
        batch_size=args.batch_size,
        repetition_penalty=args.repetition_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        top_p=args.top_p,
        temperature=args.temperature,
    )


if __name__ == "__main__":
    main()
