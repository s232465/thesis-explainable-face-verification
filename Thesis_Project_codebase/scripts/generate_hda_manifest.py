"""
generate_hda_manifest.py

Generates manifest.csv for the HDA Doppelgänger impostor set.

Folder structure:
    HDA/
        Male/    M_001_Original.jpg  M_001_Lookalike.jpg  ...
        Female/  F_001_Original.jpg  F_001_Lookalike.jpg  ...

All pairs are non-mated (impostor) by definition. imgA = Original
(reference), imgB = Lookalike (probe). Extensions are mixed
(.jpg/.png/.jpeg) and resolved automatically.

Usage:
    python scripts/generate_hda_manifest.py \
        --hda_dir "dataset_for_gorilla/HDA" \
        --out     "assets/hda/manifest.csv"
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import List, Optional


VALID_EXTS = {".jpg", ".jpeg", ".png"}


def _find_file(folder: Path, stem: str) -> Optional[Path]:
    """Find a file matching stem with any image extension."""
    for ext in [".jpg", ".jpeg", ".png"]:
        p = folder / f"{stem}{ext}"
        if p.exists():
            return p
    # Case-insensitive fallback
    stem_lower = stem.lower()
    for p in folder.iterdir():
        if p.suffix.lower() in VALID_EXTS and p.stem.lower() == stem_lower:
            return p
    return None


def _parse_numbers(folder: Path, prefix: str) -> List[int]:
    """Find all pair numbers present in folder for a given prefix (M or F)."""
    numbers = set()
    pattern = re.compile(rf"^{re.escape(prefix)}_(\d{{3}})_Original$", re.IGNORECASE)
    for p in folder.iterdir():
        if p.suffix.lower() not in VALID_EXTS:
            continue
        m = pattern.match(p.stem)
        if m:
            numbers.add(int(m.group(1)))
    return sorted(numbers)


def generate(hda_dir: Path, out_path: Path) -> None:
    hda_dir  = hda_dir.resolve()
    out_path = out_path.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    male_dir   = hda_dir / "Male"
    female_dir = hda_dir / "Female"

    for d in [male_dir, female_dir]:
        if not d.exists():
            print(f"ERROR: folder not found: {d}")
            sys.exit(1)

    rows   = []
    errors = []

    for folder, prefix, category in [
        (male_dir,   "M", "male_impostor"),
        (female_dir, "F", "female_impostor"),
    ]:
        numbers = _parse_numbers(folder, prefix)
        print(f"[{category}] Found {len(numbers)} pairs in {folder.name}/")

        for num in numbers:
            num_str    = f"{num:03d}"
            orig_stem  = f"{prefix}_{num_str}_Original"
            look_stem  = f"{prefix}_{num_str}_Lookalike"
            pair_id    = f"hda_{prefix.lower()}_{num_str}"

            path_a = _find_file(folder, orig_stem)
            path_b = _find_file(folder, look_stem)

            ok = True
            if path_a is None:
                errors.append(f"MISSING: {folder}/{orig_stem}.*")
                ok = False
            if path_b is None:
                errors.append(f"MISSING: {folder}/{look_stem}.*")
                ok = False

            rows.append({
                "pair_id":      pair_id,
                "category":     category,
                "ground_truth": "nonmatch",
                "img_a":        str(path_a).replace("\\", "/") if path_a else "MISSING",
                "img_b":        str(path_b).replace("\\", "/") if path_b else "MISSING",
                "notes":        "" if ok else "FILE_MISSING",
            })

    fieldnames = ["pair_id", "category", "ground_truth", "img_a", "img_b", "notes"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    n_male   = sum(1 for r in rows if r["category"] == "male_impostor")
    n_female = sum(1 for r in rows if r["category"] == "female_impostor")
    print(f"\nWritten {len(rows)} pairs -> {out_path}")
    print(f"  male_impostor    {n_male}")
    print(f"  female_impostor  {n_female}")

    if errors:
        print(f"\n{len(errors)} missing file(s):")
        for e in errors:
            print(f"  {e}")
        sys.exit(1)
    else:
        print("\nAll files verified.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hda_dir", required=True,
                        help="Path to HDA folder containing Male/ and Female/")
    parser.add_argument("--out", required=True,
                        help="Output manifest CSV path")
    args = parser.parse_args()
    generate(Path(args.hda_dir), Path(args.out))


if __name__ == "__main__":
    main()
