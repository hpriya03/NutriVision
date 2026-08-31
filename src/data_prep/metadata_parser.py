"""
NutriVision — Metadata Parser

Reads Nutrition5K dish metadata CSVs and returns structured data.
Used by every other script (ground truth, training data, evaluation).

CSV format (no header, per official Nutrition5K docs):
  dish_id, total_calories, total_mass, total_fat, total_carb, total_protein,
  [ingr_id, ingr_name, ingr_grams, ingr_cal, ingr_fat, ingr_carb, ingr_protein] × N

Usage:
    from src.data_prep.metadata_parser import load_metadata
    metadata = load_metadata()
    dish = metadata["dish_1561662216"]
    print(dish["calories"])       # 300.8
    print(dish["ingredients"])    # [{"name": "soy sauce", "grams": 3.4, ...}, ...]
"""

import csv
from pathlib import Path

# Default path — works when called from ~/nutrivision/
DEFAULT_METADATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "metadata"


def load_metadata(metadata_dir=None):
    """
    Load all dish metadata from both cafe CSVs.

    Returns:
        dict: {dish_id: {calories, mass, fat, carb, protein, ingredients: [...]}}
    """
    if metadata_dir is None:
        metadata_dir = DEFAULT_METADATA_DIR
    else:
        metadata_dir = Path(metadata_dir)

    metadata = {}

    for csv_file in sorted(metadata_dir.glob("dish_metadata_cafe*.csv")):
        with open(csv_file, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                if not row or not row[0].startswith("dish_"):
                    continue

                dish_id = row[0]
                total_cal = float(row[1])
                total_mass = float(row[2])
                total_fat = float(row[3])
                total_carb = float(row[4])
                total_prot = float(row[5])

                # Parse ingredients (groups of 7 fields each)
                ingredients = []
                i = 6
                while i + 6 < len(row):
                    try:
                        ingredients.append({
                            "id": row[i],
                            "name": row[i + 1].strip(),
                            "grams": float(row[i + 2]),
                            "calories": float(row[i + 3]),
                            "fat": float(row[i + 4]),
                            "carb": float(row[i + 5]),
                            "protein": float(row[i + 6]),
                        })
                    except (ValueError, IndexError):
                        break
                    i += 7

                metadata[dish_id] = {
                    "calories": total_cal,
                    "mass": total_mass,
                    "fat": total_fat,
                    "carb": total_carb,
                    "protein": total_prot,
                    "ingredients": ingredients,
                    "source_file": csv_file.name,
                }

    return metadata


def get_ingredient_names(dish_data):
    """Return sorted list of ingredient names for a dish."""
    return sorted(set(ing["name"] for ing in dish_data["ingredients"]))


def get_top_ingredients(dish_data, n=5):
    """Return top N ingredients by mass (the main components of the dish)."""
    sorted_ingr = sorted(dish_data["ingredients"], key=lambda x: x["grams"], reverse=True)
    return [ing["name"] for ing in sorted_ingr[:n]]


if __name__ == "__main__":
    # Quick self-test
    metadata = load_metadata()
    print(f"Loaded {len(metadata)} dishes")

    # Test with first dish
    test_dish = list(metadata.keys())[0]
    d = metadata[test_dish]
    print(f"\nSample dish: {test_dish}")
    print(f"  Calories: {d['calories']:.1f}")
    print(f"  Mass: {d['mass']:.1f}g")
    print(f"  Fat: {d['fat']:.1f}g, Carb: {d['carb']:.1f}g, Protein: {d['protein']:.1f}g")
    print(f"  Ingredients ({len(d['ingredients'])}): {', '.join(get_ingredient_names(d))}")
    print(f"  Top 3 by mass: {get_top_ingredients(d, 3)}")