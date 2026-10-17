"""
split_dataset.py
----------------
Splits the PlantVillage tomato dataset into train / val / test subsets.

Split ratio  : 70% train | 15% val | 15% test  (configurable below)
Random seed  : 42  (fixed for reproducibility)
Method       : COPY (originals in data/plantvillage/train are untouched)
Output       : data/plantvillage/split/{train,val,test}/<ClassName>/
"""

import random
import shutil
import sys
from pathlib import Path

# ── Configuration (edit these if needed) ─────────────────────────────────────

RANDOM_SEED  = 42
TRAIN_RATIO  = 0.70
VAL_RATIO    = 0.15
# TEST_RATIO is derived: 1.0 - TRAIN_RATIO - VAL_RATIO  (= 0.15)

SOURCE_DIR = Path(__file__).resolve().parent.parent / "data" / "plantvillage" / "train"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "plantvillage" / "split"

CLASS_NAMES = [
    "Tomato_Early_blight",
    "Tomato_healthy",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

SPLITS = ["train", "val", "test"]

# ── Safety check: are output folders already populated? ───────────────────────

def count_existing_output_images() -> int:
    """Return the total number of image files already in the output directory."""
    if not OUTPUT_DIR.exists():
        return 0
    return sum(
        1
        for f in OUTPUT_DIR.rglob("*")
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    )


def ask_overwrite_permission(existing_count: int) -> bool:
    """Ask the user whether to proceed when output already has files."""
    print(f"\n[WARNING] The output folder already contains {existing_count} image file(s):")
    print(f"          {OUTPUT_DIR}")
    print()
    print("  Options:")
    print("    y  - Delete existing output and re-split (safe — originals untouched)")
    print("    n  - Abort (keep existing output as-is)")
    print()
    while True:
        choice = input("  Your choice [y/n]: ").strip().lower()
        if choice in ("y", "yes"):
            return True
        if choice in ("n", "no"):
            return False
        print("  Please type 'y' or 'n'.")


# ── Core split logic ──────────────────────────────────────────────────────────

def split_class(class_name: str, rng: random.Random):
    """
    Shuffle and partition image files for one class, then copy them
    into the appropriate split sub-folders.

    Returns a dict: { 'train': count, 'val': count, 'test': count }
    """
    src_class_dir = SOURCE_DIR / class_name
    if not src_class_dir.exists():
        print(f"  [ERROR] Source folder not found: {src_class_dir}")
        sys.exit(1)

    # Collect all image paths
    images = sorted([
        f for f in src_class_dir.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    ])

    if not images:
        print(f"  [ERROR] No images found in {src_class_dir}")
        sys.exit(1)

    # Shuffle with fixed seed for reproducibility
    rng.shuffle(images)

    n_total = len(images)
    n_train = round(n_total * TRAIN_RATIO)
    n_val   = round(n_total * VAL_RATIO)
    # test gets the remainder (avoids any rounding gaps)
    n_test  = n_total - n_train - n_val

    partitions = {
        "train": images[:n_train],
        "val":   images[n_train : n_train + n_val],
        "test":  images[n_train + n_val :],
    }

    counts = {}
    for split_name, file_list in partitions.items():
        dest_dir = OUTPUT_DIR / split_name / class_name
        dest_dir.mkdir(parents=True, exist_ok=True)

        for src_path in file_list:
            dest_path = dest_dir / src_path.name
            shutil.copy2(src_path, dest_path)   # copy2 preserves metadata

        counts[split_name] = len(file_list)

    return counts


# ── Pretty printing ───────────────────────────────────────────────────────────

def print_summary(results: dict):
    """Print a formatted table of per-class and total counts."""
    col_w = 25

    print()
    print("=" * 68)
    print("  Dataset Split Summary")
    print("=" * 68)
    print(f"  {'Class':<{col_w}} {'Train':>7}  {'Val':>6}  {'Test':>6}  {'Total':>7}")
    print("  " + "-" * 60)

    totals = {"train": 0, "val": 0, "test": 0}

    for class_name, counts in results.items():
        t = counts["train"]
        v = counts["val"]
        te = counts["test"]
        row_total = t + v + te
        print(f"  {class_name:<{col_w}} {t:>7}  {v:>6}  {te:>6}  {row_total:>7}")
        totals["train"] += t
        totals["val"]   += v
        totals["test"]  += te

    print("  " + "-" * 60)
    grand = totals["train"] + totals["val"] + totals["test"]
    print(f"  {'TOTAL':<{col_w}} {totals['train']:>7}  {totals['val']:>6}  {totals['test']:>6}  {grand:>7}")
    print("=" * 68)

    # Ratio check
    print()
    print(f"  Effective split ratios (approx):")
    print(f"    Train : {totals['train'] / grand * 100:.1f}%")
    print(f"    Val   : {totals['val']   / grand * 100:.1f}%")
    print(f"    Test  : {totals['test']  / grand * 100:.1f}%")
    print()
    print(f"  Output saved to : {OUTPUT_DIR}")
    print()
    print("[OK]   Splitting complete. Originals in data/plantvillage/train/ are untouched.")
    print()


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    print()
    print("=" * 68)
    print("  PlantVillage Dataset Splitter")
    print(f"  Ratio  : {int(TRAIN_RATIO*100)}% train / "
          f"{int(VAL_RATIO*100)}% val / "
          f"{int((1-TRAIN_RATIO-VAL_RATIO)*100)}% test")
    print(f"  Seed   : {RANDOM_SEED}")
    print("=" * 68)

    # ── Safety: check for existing output ────────────────────────────────────
    existing = count_existing_output_images()
    if existing > 0:
        proceed = ask_overwrite_permission(existing)
        if not proceed:
            print("\n[ABORTED] No files were changed.")
            sys.exit(0)
        # Remove existing output directory entirely before re-splitting
        print(f"\n  Removing old output directory...")
        shutil.rmtree(OUTPUT_DIR)
        print(f"  Done. Starting fresh split...\n")
    else:
        print(f"\n  No existing output found. Proceeding with fresh split...\n")

    # ── Perform split ─────────────────────────────────────────────────────────
    rng = random.Random(RANDOM_SEED)
    results = {}

    for class_name in CLASS_NAMES:
        print(f"  Splitting  {class_name} ...", end="  ", flush=True)
        counts = split_class(class_name, rng)
        print(f"train={counts['train']}  val={counts['val']}  test={counts['test']}")
        results[class_name] = counts

    # ── Print summary ─────────────────────────────────────────────────────────
    print_summary(results)


if __name__ == "__main__":
    main()
