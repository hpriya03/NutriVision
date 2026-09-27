"""
NutriVision - Step 1: Create train/test splits.

Uses official Nutrition5K RGB splits to preserve incremental scan
grouping (prevents data leakage), then filters to only overhead
dishes that have both images and metadata.

228 overhead dishes not present in official splits are added to
the training set (safe - they have no scan partners in test).

Final split: ~2983 train / 507 test (85/15).

Usage:
    python src/data_prep/create_splits.py
"""

import json
import os
from pathlib import Path

# ── Paths ──
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
OFFICIAL_DIR = PROJECT_ROOT / "data" / "official_splits"
IMAGE_DIR = RAW_DIR / "realsense_overhead"
METADATA_DIR = RAW_DIR / "metadata"


def get_dishes_with_images():
    """Find dish IDs that have rgb.png in realsense_overhead/."""
    dishes = set()
    for folder in IMAGE_DIR.iterdir():
        if folder.is_dir() and (folder / "rgb.png").exists():
            dishes.add(folder.name)
    return dishes


def get_dishes_with_metadata():
    """Find dish IDs that appear in the metadata CSVs."""
    dishes = set()
    for csv_file in METADATA_DIR.glob("dish_metadata_cafe*.csv"):
        with open(csv_file, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    dishes.add(line.split(",")[0])
    return dishes


def load_official_split(filename):
    """Load dish IDs from an official split file (one ID per line)."""
    path = OFFICIAL_DIR / filename
    with open(path, "r") as f:
        return set(line.strip() for line in f if line.strip())


def main():
    print("=" * 55)
    print("NutriVision — Creating Train/Test Splits")
    print("  Using official Nutrition5K splits + overhead filter")
    print("=" * 55)

    # Step 1: Find usable overhead dishes (have BOTH image + metadata)
    image_dishes = get_dishes_with_images()
    meta_dishes = get_dishes_with_metadata()
    overhead = image_dishes & meta_dishes

    print(f"\nDishes with overhead rgb.png:  {len(image_dishes)}")
    print(f"Dishes with metadata:          {len(meta_dishes)}")
    print(f"Dishes with BOTH (usable):     {len(overhead)}")

    img_only = image_dishes - meta_dishes
    if img_only:
        print(f"  WARNING: {len(img_only)} have images but no metadata")

    # Step 2: Load official splits
    official_train = load_official_split("rgb_train_ids.txt")
    official_test = load_official_split("rgb_test_ids.txt")

    print(f"\nOfficial RGB train IDs:        {len(official_train)}")
    print(f"Official RGB test IDs:         {len(official_test)}")

    # Step 3: Filter to our overhead dishes
    train_ids = overhead & official_train
    test_ids = overhead & official_test
    not_in_official = overhead - official_train - official_test

    print(f"\nAfter filtering to overhead:")
    print(f"  Train (from official):       {len(train_ids)}")
    print(f"  Test (from official):        {len(test_ids)}")
    print(f"  Not in any official split:   {len(not_in_official)}")

   # Step 4: Exclude dishes not in official splits
    print(f"  Excluded (not in official splits): {len(not_in_official)}")

    # Step 5: Safety check - no overlap
    overlap = train_ids & test_ids
    assert len(overlap) == 0, f"DATA LEAK: {len(overlap)} dishes in both train and test!"

    # Sort for reproducibility
    train_ids = sorted(train_ids)
    test_ids = sorted(test_ids)

    print(f"\nFinal split:")
    print(f"  Train: {len(train_ids)} dishes ({len(train_ids)*100/(len(train_ids)+len(test_ids)):.1f}%)")
    print(f"  Test:  {len(test_ids)} dishes ({len(test_ids)*100/(len(train_ids)+len(test_ids)):.1f}%)")
    print(f"  Total: {len(train_ids) + len(test_ids)}")

    # Step 6: Save
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    with open(SPLITS_DIR / "train_ids.json", "w") as f:
        json.dump(train_ids, f, indent=2)

    with open(SPLITS_DIR / "test_ids.json", "w") as f:
        json.dump(test_ids, f, indent=2)

    print(f"\nSaved to:")
    print(f"  {SPLITS_DIR / 'train_ids.json'}")
    print(f"  {SPLITS_DIR / 'test_ids.json'}")
    print("\nDone!")


if __name__ == "__main__":
    main()