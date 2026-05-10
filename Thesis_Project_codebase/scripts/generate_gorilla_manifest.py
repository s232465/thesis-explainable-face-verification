"""
generate_gorilla_manifest.py

Generates manifest.csv for the Gorilla curated 41-pair set from the
ground-truth table used in the user study.

Filename convention in resized_512_crop/:
    easy_NN_A/B   -> Easy (2N-1).jpg / Easy (2N).jpg
    medium_NN_A/B -> medium (2N-1).jpg / medium (2N).jpg
    hard_21..27   -> Hard_same (1)..(14).jpg  (mated pairs)
    hard_01..20   -> Hard (1)..(40).jpg       (non-mated pairs)

Usage:
    python scripts/generate_gorilla_manifest.py \
        --img_dir  "dataset_for_gorilla/resized_512_crop" \
        --out      "assets/gorilla/manifest.csv"
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

# Ground-truth table (matches the study spreadsheet)

GT_ROWS = [
    ("PAIR_21", "easy_01_A",    "easy_01_B",    "SAME"),
    ("PAIR_22", "easy_02_A",    "easy_02_B",    "SAME"),
    ("PAIR_23", "easy_03_A",    "easy_03_B",    "SAME"),
    ("PAIR_24", "easy_04_A",    "easy_04_B",    "DIFFERENT"),
    ("PAIR_25", "easy_05_A",    "easy_05_B",    "DIFFERENT"),
    ("PAIR_26", "easy_06_A",    "easy_06_B",    "DIFFERENT"),
    ("PAIR_27", "medium_01_A",  "medium_01_B",  "SAME"),
    ("PAIR_28", "medium_02_A",  "medium_02_B",  "SAME"),
    ("PAIR_29", "medium_03_A",  "medium_03_B",  "SAME"),
    ("PAIR_30", "medium_04_A",  "medium_04_B",  "SAME"),
    ("PAIR_31", "medium_05_A",  "medium_05_B",  "DIFFERENT"),
    ("PAIR_32", "medium_06_A",  "medium_06_B",  "DIFFERENT"),
    ("PAIR_33", "medium_07_A",  "medium_07_B",  "DIFFERENT"),
    ("PAIR_34", "medium_08_A",  "medium_08_B",  "DIFFERENT"),
    ("PAIR_35", "hard_21_A",    "hard_21_B",    "SAME"),
    ("PAIR_36", "hard_22_A",    "hard_22_B",    "SAME"),
    ("PAIR_37", "hard_23_A",    "hard_23_B",    "SAME"),
    ("PAIR_38", "hard_24_A",    "hard_24_B",    "SAME"),
    ("PAIR_39", "hard_25_A",    "hard_25_B",    "SAME"),
    ("PAIR_40", "hard_26_A",    "hard_26_B",    "SAME"),
    ("PAIR_41", "hard_27_A",    "hard_27_B",    "SAME"),
    ("PAIR_1",  "hard_01_A",    "hard_01_B",    "DIFFERENT"),
    ("PAIR_2",  "hard_02_A",    "hard_02_B",    "DIFFERENT"),
    ("PAIR_3",  "hard_03_A",    "hard_03_B",    "DIFFERENT"),
    ("PAIR_4",  "hard_04_A",    "hard_04_B",    "DIFFERENT"),
    ("PAIR_5",  "hard_05_A",    "hard_05_B",    "DIFFERENT"),
    ("PAIR_6",  "hard_06_A",    "hard_06_B",    "DIFFERENT"),
    ("PAIR_7",  "hard_07_A",    "hard_07_B",    "DIFFERENT"),
    ("PAIR_8",  "hard_08_A",    "hard_08_B",    "DIFFERENT"),
    ("PAIR_9",  "hard_09_A",    "hard_09_B",    "DIFFERENT"),
    ("PAIR_10", "hard_10_A",    "hard_10_B",    "DIFFERENT"),
    ("PAIR_11", "hard_11_A",    "hard_11_B",    "DIFFERENT"),
    ("PAIR_12", "hard_12_A",    "hard_12_B",    "DIFFERENT"),
    ("PAIR_13", "hard_13_A",    "hard_13_B",    "DIFFERENT"),
    ("PAIR_14", "hard_14_A",    "hard_14_B",    "DIFFERENT"),
    ("PAIR_15", "hard_15_A",    "hard_15_B",    "DIFFERENT"),
    ("PAIR_16", "hard_16_A",    "hard_16_B",    "DIFFERENT"),
    ("PAIR_17", "hard_17_A",    "hard_17_B",    "DIFFERENT"),
    ("PAIR_18", "hard_18_A",    "hard_18_B",    "DIFFERENT"),
    ("PAIR_19", "hard_19_A",    "hard_19_B",    "DIFFERENT"),
    ("PAIR_20", "hard_20_A",    "hard_20_B",    "DIFFERENT"),
]


def _gt_name_to_filename(name: str) -> str:
    """Map a ground-truth image name to the actual filename on disk."""
    m = re.match(r"(easy|medium|hard)_(\d+)_(A|B)$", name)
    if not m:
        raise ValueError(f"Cannot parse GT name: {name!r}")
    category = m.group(1)
    num      = int(m.group(2))
    side     = m.group(3)
    offset   = 1 if side == "A" else 2

    if category == "easy":
        idx = (num - 1) * 2 + offset
        return f"Easy ({idx}).jpg"
    elif category == "medium":
        idx = (num - 1) * 2 + offset
        return f"medium ({idx}).jpg"
    elif category == "hard" and num >= 21:
        local = num - 20
        idx   = (local - 1) * 2 + offset
        return f"Hard_same ({idx}).jpg"
    else:
        idx = (num - 1) * 2 + offset
        return f"Hard ({idx}).jpg"


def _category_label(gt_name_a: str, gt_label: str) -> str:
    m = re.match(r"(easy|medium|hard)_", gt_name_a)
    prefix = m.group(1) if m else "unknown"
    suffix = "match" if gt_label == "SAME" else "nonmatch"
    return f"{prefix}_{suffix}"


def generate(img_dir: Path, out_path: Path) -> None:
    img_dir  = img_dir.resolve()
    out_path = out_path.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows  = []
    errors = []

    for pair_id, name_a, name_b, gt_label in GT_ROWS:
        fname_a = _gt_name_to_filename(name_a)
        fname_b = _gt_name_to_filename(name_b)
        path_a  = img_dir / fname_a
        path_b  = img_dir / fname_b

        ok = True
        if not path_a.exists():
            errors.append(f"MISSING: {path_a}")
            ok = False
        if not path_b.exists():
            errors.append(f"MISSING: {path_b}")
            ok = False

        gt_norm   = "match" if gt_label == "SAME" else "nonmatch"
        category  = _category_label(name_a, gt_label)
        pair_id_c = pair_id.lower().replace("pair_", "gorilla_pair_")

        rows.append({
            "pair_id":      pair_id_c,
            "category":     category,
            "ground_truth": gt_norm,
            "img_a":        str(path_a).replace("\\", "/"),
            "img_b":        str(path_b).replace("\\", "/"),
            "notes":        "" if ok else "FILE_MISSING",
        })

    fieldnames = ["pair_id", "category", "ground_truth", "img_a", "img_b", "notes"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Written {len(rows)} pairs -> {out_path}")

    from collections import Counter
    cats = Counter(r["category"] for r in rows)
    for cat, n in sorted(cats.items()):
        print(f"  {cat:<25} {n}")

    if errors:
        print(f"\n{len(errors)} missing file(s):")
        for e in errors:
            print(f"  {e}")
        sys.exit(1)
    else:
        print("\nAll files verified.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", required=True,
                        help="Path to resized_512_crop folder")
    parser.add_argument("--out", required=True,
                        help="Output manifest CSV path")
    args = parser.parse_args()
    generate(Path(args.img_dir), Path(args.out))


if __name__ == "__main__":
    main()
