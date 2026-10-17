"""
verify_dataset.py
-----------------
Verifies the PlantVillage tomato dataset:
  - Checks that each class folder exists
  - Counts valid image files per class
  - Detects and reports unreadable / corrupt images
  - Prints a summary table
"""

import os
import sys
from pathlib import Path
from PIL import Image, UnidentifiedImageError

# ── Configuration ────────────────────────────────────────────────────────────

TRAIN_DIR = Path(__file__).resolve().parent.parent / "data" / "plantvillage" / "train"

CLASS_NAMES = [
    "Tomato_Early_blight",
    "Tomato_healthy",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
]

# Common image extensions to recognise
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".gif", ".webp"}

# ── Helpers ───────────────────────────────────────────────────────────────────

def check_image(path: Path) -> str:
    """
    Try to open an image with Pillow.
    Returns 'ok', 'corrupt', or 'unreadable'.
    """
    try:
        with Image.open(path) as img:
            img.verify()          # checks file integrity without decoding
        return "ok"
    except UnidentifiedImageError:
        return "corrupt"
    except Exception:
        return "unreadable"


def verify_class(class_dir: Path):
    """
    Walk a single class directory and return statistics.
    """
    valid_count   = 0
    bad_files     = []   # list of (filename, reason)

    if not class_dir.exists():
        return None, None, None   # folder missing

    all_files = list(class_dir.iterdir())
    image_files = [f for f in all_files if f.suffix.lower() in IMAGE_EXTENSIONS]

    for img_path in image_files:
        status = check_image(img_path)
        if status == "ok":
            valid_count += 1
        else:
            bad_files.append((img_path.name, status))

    non_image = [f.name for f in all_files if f.suffix.lower() not in IMAGE_EXTENSIONS]

    return valid_count, bad_files, non_image


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 60)
    print("  PlantVillage Dataset Verification Report")
    print("=" * 60)
    print(f"  Dataset root : {TRAIN_DIR}\n")

    if not TRAIN_DIR.exists():
        print(f"[ERROR] Train directory not found:\n  {TRAIN_DIR}")
        print("  Make sure your images are inside data/plantvillage/train/")
        sys.exit(1)

    grand_total_valid = 0
    grand_total_bad   = 0
    missing_classes   = []

    col_w = 25   # column width for pretty printing

    header = f"  {'Class':<{col_w}} {'Valid':>7}  {'Corrupt/Bad':>11}  {'Status'}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for class_name in CLASS_NAMES:
        class_dir = TRAIN_DIR / class_name
        valid, bad_files, non_image = verify_class(class_dir)

        if valid is None:
            status = "FOLDER MISSING"
            missing_classes.append(class_name)
            print(f"  {class_name:<{col_w}} {'--':>7}  {'--':>11}  {status}")
            continue

        bad_count = len(bad_files)
        grand_total_valid += valid
        grand_total_bad   += bad_count

        if bad_count == 0 and valid > 0:
            status = "OK"
        elif valid == 0:
            status = "WARNING: EMPTY FOLDER"
        else:
            status = f"WARNING: {bad_count} bad image(s)"

        print(f"  {class_name:<{col_w}} {valid:>7}  {bad_count:>11}  {status}")

        # Report bad files in detail (up to 10)
        if bad_files:
            for fname, reason in bad_files[:10]:
                print(f"       -> {fname}  [{reason}]")
            if len(bad_files) > 10:
                print(f"       -> ... and {len(bad_files) - 10} more")

        # Warn about non-image files
        if non_image:
            print(f"       ! {len(non_image)} non-image file(s) found (e.g. {non_image[0]})")

    print("  " + "-" * (len(header) - 2))
    print(f"  {'TOTAL':<{col_w}} {grand_total_valid:>7}  {grand_total_bad:>11}")
    print("=" * 60)

    # Final verdict
    print()
    if missing_classes:
        print("[FAIL] Missing folders:")
        for mc in missing_classes:
            print(f"       {TRAIN_DIR / mc}")
        print("       -> Copy your images into the paths above and re-run.")
    elif grand_total_valid == 0:
        print("[FAIL] No valid images found. Check that images are inside the class folders.")
    elif grand_total_bad > 0:
        print(f"[WARN] {grand_total_bad} corrupt / unreadable image(s) found.")
        print("       Consider removing them before training.")
        print(f"\n[OK]   {grand_total_valid} clean images are ready.")
    else:
        print(f"[OK]   All {grand_total_valid} images passed validation.")
        print("       Dataset is ready for the next step (train/val/test split).")

    print()


if __name__ == "__main__":
    main()
