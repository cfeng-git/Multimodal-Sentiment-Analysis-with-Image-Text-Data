import csv
from pathlib import Path
from collections import Counter

BASE_DIR = Path(__file__).resolve().parent
input_file = BASE_DIR / "MVSA" / "labelResultAll.txt"
output_file = BASE_DIR / "MVSA" / "labels_master.csv"

def majority_with_tie_rule(labels):
    # handle missing
    labels = [l for l in labels if l]
    c = Counter(labels)
    # If we have all three unique labels, return neutral
    if len(c) == 3:
        return "neutral"
    # Otherwise, return majority (break ties by frequency)
    return c.most_common(1)[0][0]

with open(input_file, "r", encoding="utf-8") as f_in, \
     open(output_file, "w", newline="", encoding="utf-8") as f_out:

    reader = csv.reader(f_in, delimiter="\t")
    header = next(reader)  # ['ID', 'text,image', 'text,image', 'text,image']

    writer = csv.DictWriter(f_out, fieldnames=["id", "text_label", "image_label", "combined_label"])
    writer.writeheader()

    total, dropped = 0, 0
    for row in reader:
        total += 1
        ex_id = row[0]
        annot_cells = row[1:]  # three annotator columns

        text_labels = []
        image_labels = []

        for cell in annot_cells:
            if not cell:
                continue
            parts = cell.split(",")
            if len(parts) != 2:
                continue
            t_lab, i_lab = parts
            text_labels.append(t_lab.strip())
            image_labels.append(i_lab.strip())

        final_text = majority_with_tie_rule(text_labels)
        final_image = majority_with_tie_rule(image_labels)

        # Discard examples where majority text and image labels disagree
        if final_text != final_image:
            dropped += 1
            continue

        writer.writerow({
            "id": ex_id,
            "text_label": final_text,
            "image_label": final_image,
            "combined_label": final_text,  # same as text_label
        })

print(f"Wrote {total - dropped} examples to {output_file} (dropped {dropped} due to text/image disagreement)")
