"""
NutriVision — Ground Truth Generator (Phase B4, v2)

Builds the answer key for evaluating model outputs against real
Nutrition5K measurements.

What it does:
    1. Loads metadata for all 507 test dishes (actual calories, macros, ingredients)
    2. Defines VQA thresholds based on established dietary guidelines
       (FDA Daily Values, American Diabetes Association, Dietary Guidelines
       for Americans) — NOT dataset-specific percentiles
    3. For each test dish, creates the correct answer for every task:
       - Task 2 (ingredients): list of ingredient names from metadata
       - Task 3 (calories): actual measured calories
       - Task 4 (macros): actual fat, carb, protein in grams
       - Task 5 (VQA): correct yes/no answers using guideline-based thresholds
         — 12 SEEN questions (used in training) + 8 UNSEEN (held-out)
       - Task 1 (captioning): skipped — no reference captions in Nutrition5K
    4. Saves to results/v2_ground_truth.json

v2 changes:
    - Added get_vqa_sentence_answer() for sentence-level training answers
    - Added 8 held-out question types with threshold functions
    - Ground truth now covers all 20 questions (12 seen + 8 unseen)
    - Output file: v2_ground_truth.json (preserves old ground_truth.json)

Usage:
    python -m src.eval.ground_truth 2>&1 | tee logs/07_ground_truth_v2.log
"""

import json

from src import config
from src.data_prep.metadata_parser import load_metadata, get_ingredient_names


# ── VQA Thresholds Based on Dietary Guidelines ──
# These are per-meal thresholds derived from established nutrition standards.
# Each threshold is traceable to a published dietary guideline.
#
# Sources:
#   - FDA Daily Values (2,000 cal/day reference diet)
#     https://www.fda.gov/food/nutrition-facts-label/daily-value-nutrition-and-supplement-facts-labels
#   - American Diabetes Association: 45-60g carbs per meal
#     https://diabetes.org/food-nutrition/understanding-carbs
#   - Dietary Guidelines for Americans 2020-2025
#     https://www.dietaryguidelines.gov/
#   - ISSN position stand on protein and exercise (Jäger et al., 2017)
#     https://doi.org/10.1186/s12970-017-0177-8

VQA_THRESHOLDS = {
    # ── Calorie thresholds ──
    # FDA DV: 2,000 cal/day → ~667 cal/meal
    # "High calorie" = above 600 cal (a calorie-dense meal)
    # "Good for calorie watching" = below 500 cal (common weight management target)
    "high_calorie": 600,          # cal per meal
    "calorie_watching": 500,      # cal per meal

    # ── Protein threshold ──
    # FDA DV: 50g/day → ~17g/meal
    # "High protein" = above 25g per meal (delivers ≥50% of daily value)
    # Widely used in nutrition labeling and dietary advice
    "high_protein": 25,           # grams per meal

    # ── Fat threshold ──
    # FDA DV: 78g/day → ~26g/meal
    # "High fat" = above 25g per meal
    "high_fat": 25,               # grams per meal

    # ── Carb thresholds ──
    # Low-carb diets: typically <100g/day → <30g/meal
    # ADA diabetic guideline: 45-60g carbs/meal (we use conservative 45g)
    "low_carb": 30,               # grams per meal
    "diabetic_carb_limit": 45,    # grams per meal (ADA guideline)

    # ── Balanced meal ──
    # Dietary Guidelines for Americans: protein 10-35%, carbs 45-65%, fat 20-35%
    # We define "balanced" as each macro contributing at least 20% of total calories
    # Calorie conversions: protein=4 cal/g, carbs=4 cal/g, fat=9 cal/g
    "balanced_min_pct": 20,       # minimum % of calories from each macro

    # ── "Healthy" (simplified proxy) ──
    # Not too many calories AND not too much fat
    "healthy_max_cal": 600,       # cal per meal
    "healthy_max_fat": 25,        # grams per meal

    # ══ Held-out question thresholds (v2) ══

    # ── Low fat ──
    # FDA DV: 78g/day → 26g/meal.  "Low fat" = below 15g per meal
    # (~20% of daily value — aligns with FDA "low fat" labeling: ≤3g per serving,
    # scaled to a full meal).
    "low_fat": 15,                # grams per meal

    # ── Light meal ──
    # Common usage: a meal under 400 calories.  More conservative than
    # "calorie watching" (500), reflecting what most people mean by "light."
    "light_meal": 400,            # cal per meal

    # ── High carbohydrates ──
    # FDA DV: 275g/day → ~92g/meal.  "High carb" = above 50g per meal
    # (exceeds ~55% of a 600-cal meal from carbs alone).
    "high_carb": 50,              # grams per meal

    # ── Weight loss diet ──
    # Moderate caloric restriction + limited fat is the standard advice.
    # We combine a calorie cap (450 cal) with a fat cap (20g).
    "weight_loss_cal": 450,       # cal per meal
    "weight_loss_fat": 20,        # grams per meal

    # ── Post-workout meal ──
    # ISSN position stand: 20-40g protein + adequate carbs for glycogen
    # replenishment after resistance exercise (Jäger et al., 2017).
    # We use 20g protein + 30g carbs as minimum thresholds.
    "post_workout_protein": 20,   # grams per meal
    "post_workout_carb": 30,      # grams per meal
}

# ── Non-vegetarian ingredient keywords ──
# Used to determine if a dish is vegetarian.  Covers common meat, poultry,
# and seafood terms found in Nutrition5K ingredient names.
MEAT_FISH_KEYWORDS = {
    "chicken", "beef", "pork", "turkey", "lamb", "steak", "bacon",
    "ham", "sausage", "salami", "pepperoni", "prosciutto", "veal",
    "duck", "venison", "bison", "chorizo", "pastrami", "jerky",
    "fish", "salmon", "tuna", "shrimp", "crab", "lobster", "clam",
    "mussel", "oyster", "squid", "anchovy", "sardine", "cod",
    "tilapia", "trout", "catfish", "halibut", "mahi", "scallop",
    "prawn", "calamari", "swordfish", "mackerel", "herring",
    "meat", "hot dog", "meatball", "wing", "ribs",
}


# ═══════════════════════════════════════════════════════════
# THRESHOLD FUNCTIONS — original 12 questions
# ═══════════════════════════════════════════════════════════

def is_high_protein(dish: dict) -> bool:
    """Above 25g protein per meal (≥50% of FDA daily value of 50g)."""
    return dish["protein"] > VQA_THRESHOLDS["high_protein"]


def is_high_fat(dish: dict) -> bool:
    """Above 25g fat per meal (based on FDA DV 78g/day ÷ 3)."""
    return dish["fat"] > VQA_THRESHOLDS["high_fat"]


def is_low_carb(dish: dict) -> bool:
    """Below 30g carbs per meal (standard low-carb diet threshold)."""
    return dish["carb"] < VQA_THRESHOLDS["low_carb"]


def is_high_calorie(dish: dict) -> bool:
    """Above 600 calories per meal."""
    return dish["calories"] > VQA_THRESHOLDS["high_calorie"]


def is_balanced_meal(dish: dict) -> bool:
    """
    Each macronutrient contributes at least 20% of total calories.
    Aligns with Dietary Guidelines for Americans recommended ranges.

    Calorie conversions:
      protein: 4 cal/g
      carbs:   4 cal/g
      fat:     9 cal/g
    """
    if dish["calories"] <= 0:
        return False

    min_pct = VQA_THRESHOLDS["balanced_min_pct"]
    total_cal = dish["calories"]

    protein_pct = (dish["protein"] * 4 / total_cal) * 100
    carb_pct = (dish["carb"] * 4 / total_cal) * 100
    fat_pct = (dish["fat"] * 9 / total_cal) * 100

    return (protein_pct >= min_pct and
            carb_pct >= min_pct and
            fat_pct >= min_pct)


def is_healthy(dish: dict) -> bool:
    """
    Simplified proxy: not too many calories AND not too much fat.
    Below 600 cal and below 25g fat.
    """
    return (dish["calories"] <= VQA_THRESHOLDS["healthy_max_cal"] and
            dish["fat"] <= VQA_THRESHOLDS["healthy_max_fat"])


def is_good_for_calorie_watching(dish: dict) -> bool:
    """Below 500 cal per meal (common weight management target)."""
    return dish["calories"] < VQA_THRESHOLDS["calorie_watching"]


def is_appropriate_for_diabetic(dish: dict) -> bool:
    """Below 45g carbs per meal (American Diabetes Association guideline)."""
    return dish["carb"] < VQA_THRESHOLDS["diabetic_carb_limit"]


# ═══════════════════════════════════════════════════════════
# THRESHOLD FUNCTIONS — 8 held-out questions (v2)
# ═══════════════════════════════════════════════════════════

def is_low_fat(dish: dict) -> bool:
    """Below 15g fat per meal (FDA 'low fat' threshold scaled to meal)."""
    return dish["fat"] < VQA_THRESHOLDS["low_fat"]


def has_more_than_500_cal(dish: dict) -> bool:
    """Above 500 calories per meal."""
    return dish["calories"] > 500


def is_light_meal(dish: dict) -> bool:
    """Below 400 calories — common meaning of 'light meal'."""
    return dish["calories"] < VQA_THRESHOLDS["light_meal"]


def has_more_fat_than_protein(dish: dict) -> bool:
    """Fat (in grams) exceeds protein (in grams)."""
    return dish["fat"] > dish["protein"]


def is_high_carb(dish: dict) -> bool:
    """Above 50g carbs per meal."""
    return dish["carb"] > VQA_THRESHOLDS["high_carb"]


def is_good_for_weight_loss(dish: dict) -> bool:
    """Below 450 cal AND below 20g fat — moderate restriction."""
    return (dish["calories"] < VQA_THRESHOLDS["weight_loss_cal"] and
            dish["fat"] < VQA_THRESHOLDS["weight_loss_fat"])


def is_vegetarian(dish: dict) -> bool:
    """
    No meat, poultry, or fish/seafood ingredients.

    Checks each ingredient name against a keyword set of common
    non-vegetarian foods.  Errs on the side of 'not vegetarian'
    if any keyword matches (conservative for dietary advice).
    """
    ingredients = get_ingredient_names(dish)
    if not ingredients or ingredients == ["plate only"]:
        return True  # empty plate is technically vegetarian

    for ingr in ingredients:
        ingr_lower = ingr.lower()
        for keyword in MEAT_FISH_KEYWORDS:
            if keyword in ingr_lower:
                return False
    return True


def is_good_post_workout(dish: dict) -> bool:
    """
    Good post-workout meal: adequate protein (>20g) for muscle protein
    synthesis and adequate carbs (>30g) for glycogen replenishment.
    Based on ISSN position stand (Jäger et al., 2017).
    """
    return (dish["protein"] > VQA_THRESHOLDS["post_workout_protein"] and
            dish["carb"] > VQA_THRESHOLDS["post_workout_carb"])


# ═══════════════════════════════════════════════════════════
# VQA ANSWER GENERATORS
# ═══════════════════════════════════════════════════════════

def get_vqa_answer(question: str, dish: dict) -> str:
    """
    Generate the correct SHORT VQA answer for evaluation ground truth.

    Numeric questions → return the actual measured value as a string.
    Yes/no questions → apply guideline-based thresholds, return "yes"/"no".
    Ingredient questions → return the actual ingredient list.

    Covers ALL 20 questions (12 seen + 8 held-out).
    """
    q = question.lower()

    # ── Numeric questions — return the real measured value ──
    if "how many calories" in q:
        return str(round(dish["calories"], 1))

    if "how much protein" in q:
        return str(round(dish["protein"], 1))

    if "how much fat" in q:
        return str(round(dish["fat"], 1))

    if "carbohydrate content" in q or "how many carb" in q:
        return str(round(dish["carb"], 1))

    # ── Ingredient questions — return the real ingredients ──
    if "main ingredients" in q or "ingredients in this" in q:
        ingrs = get_ingredient_names(dish)
        return ", ".join(ingrs)

    # ── Yes/no questions — original 12 ──
    if "high in protein" in q:
        return "yes" if is_high_protein(dish) else "no"

    if "low-carb" in q or "low carb" in q:
        return "yes" if is_low_carb(dish) else "no"

    if "balanced meal" in q:
        return "yes" if is_balanced_meal(dish) else "no"

    if "healthy" in q:
        return "yes" if is_healthy(dish) else "no"

    if "watching their calorie" in q or "calorie intake" in q:
        return "yes" if is_good_for_calorie_watching(dish) else "no"

    if "diabetic" in q:
        return "yes" if is_appropriate_for_diabetic(dish) else "no"

    if "food groups" in q:
        ingrs = get_ingredient_names(dish)
        return ", ".join(ingrs)

    # ── Yes/no questions — 8 held-out (v2) ──
    if "low in fat" in q:
        return "yes" if is_low_fat(dish) else "no"

    if "more than 500 calories" in q:
        return "yes" if has_more_than_500_cal(dish) else "no"

    if "light meal" in q:
        return "yes" if is_light_meal(dish) else "no"

    if "more fat than protein" in q:
        return "yes" if has_more_fat_than_protein(dish) else "no"

    if "high in carbohydrates" in q:
        return "yes" if is_high_carb(dish) else "no"

    if "weight loss" in q:
        return "yes" if is_good_for_weight_loss(dish) else "no"

    if "vegetarian" in q:
        return "yes" if is_vegetarian(dish) else "no"

    if "post-workout" in q or "post workout" in q:
        return "yes" if is_good_post_workout(dish) else "no"

    # Fallback
    return "UNKNOWN_QUESTION"


def get_vqa_sentence_answer(question: str, dish: dict) -> str:
    """
    Generate a SENTENCE-LEVEL VQA answer for training data.

    Unlike get_vqa_answer() which returns short answers ("yes", "no", "350.5")
    for evaluation, this function returns natural language sentences that
    exploit VLM capabilities.  The model learns to produce descriptive,
    evidence-based responses rather than single-word outputs.

    Only used for the 12 TRAINING questions — the 8 held-out questions
    never appear in training data.

    Example pairs:
        Q: "How many calories does this dish have?"
        Short (old):    "350.5"
        Sentence (new): "This dish contains approximately 351 calories."

        Q: "Is this dish high in protein?"
        Short (old):    "yes"
        Sentence (new): "Yes, this dish is high in protein with 32.5 grams,
                         which exceeds the 25 gram per meal threshold."
    """
    q = question.lower()
    ingredients = get_ingredient_names(dish)

    # ── Numeric questions — embed value in a natural sentence ──
    if "how many calories" in q:
        cal = round(dish["calories"])
        return f"This dish contains approximately {cal} calories."

    if "how much protein" in q:
        protein = round(dish["protein"], 1)
        if is_high_protein(dish):
            return (f"This dish provides {protein} grams of protein, "
                    f"which is considered high in protein.")
        return f"This dish provides {protein} grams of protein."

    if "how much fat" in q:
        fat = round(dish["fat"], 1)
        if is_high_fat(dish):
            return (f"This dish contains {fat} grams of fat, "
                    f"which is relatively high.")
        return f"This dish contains {fat} grams of fat."

    if "carbohydrate content" in q or "how many carb" in q:
        carb = round(dish["carb"], 1)
        return (f"The approximate carbohydrate content of this dish "
                f"is {carb} grams.")

    # ── Ingredient questions — list in a natural sentence ──
    if "main ingredients" in q or "ingredients in this" in q:
        if not ingredients or ingredients == ["plate only"]:
            return ("This appears to be an empty plate with no visible "
                    "food ingredients.")
        if len(ingredients) == 1:
            ingr_text = ingredients[0]
        else:
            ingr_text = (", ".join(ingredients[:-1]) +
                         f" and {ingredients[-1]}")
        return f"The main ingredients in this dish are {ingr_text}."

    # ── Yes/no questions — start with Yes/No, then give evidence ──
    if "high in protein" in q:
        protein = round(dish["protein"], 1)
        if is_high_protein(dish):
            return (f"Yes, this dish is high in protein with {protein} "
                    f"grams, which exceeds the "
                    f"{VQA_THRESHOLDS['high_protein']} gram per meal "
                    f"threshold.")
        return (f"No, this dish is not particularly high in protein, "
                f"containing {protein} grams.")

    if "low-carb" in q or "low carb" in q:
        carb = round(dish["carb"], 1)
        if is_low_carb(dish):
            return (f"Yes, this dish is suitable for a low-carb diet "
                    f"with only {carb} grams of carbohydrates.")
        return (f"No, this dish is not low-carb as it contains "
                f"{carb} grams of carbohydrates.")

    if "balanced meal" in q:
        if is_balanced_meal(dish):
            return ("Yes, this would be considered a balanced meal as "
                    "each macronutrient contributes a meaningful "
                    "proportion of the total calories.")
        return ("No, this would not be considered a balanced meal as "
                "the macronutrient proportions are not evenly "
                "distributed.")

    if "healthy" in q:
        cal = round(dish["calories"])
        fat = round(dish["fat"], 1)
        if is_healthy(dish):
            return (f"Yes, this dish can be considered a healthy option "
                    f"with {cal} calories and {fat} grams of fat.")
        return (f"No, this dish may not be the healthiest option due to "
                f"its {cal} calorie and {fat} gram fat content.")

    if "watching their calorie" in q or "calorie intake" in q:
        cal = round(dish["calories"])
        if is_good_for_calorie_watching(dish):
            return (f"Yes, at {cal} calories, this dish is suitable for "
                    f"someone watching their calorie intake.")
        return (f"No, at {cal} calories, this dish may be too "
                f"calorie-dense for someone watching their intake.")

    if "diabetic" in q:
        carb = round(dish["carb"], 1)
        if is_appropriate_for_diabetic(dish):
            return (f"Yes, with {carb} grams of carbohydrates, this dish "
                    f"falls within the recommended range for a diabetic "
                    f"diet.")
        return (f"No, with {carb} grams of carbohydrates, this dish "
                f"exceeds the recommended carbohydrate limit for a "
                f"diabetic meal.")

    if "food groups" in q:
        if not ingredients or ingredients == ["plate only"]:
            return ("This appears to be an empty plate with no food "
                    "groups represented.")
        if len(ingredients) == 1:
            ingr_text = ingredients[0]
        else:
            ingr_text = (", ".join(ingredients[:-1]) +
                         f" and {ingredients[-1]}")
        return (f"The food groups represented in this dish include "
                f"{ingr_text}.")

    # Fallback
    return "UNKNOWN_QUESTION"


def compute_dataset_stats(metadata: dict) -> dict:
    """
    Compute dataset-wide statistics for reporting purposes.
    These are NOT used for thresholds (thresholds are guideline-based),
    but they provide useful context for interpreting results — e.g., "the median
    calorie content in Nutrition5K is X, compared to the FDA per-meal
    guideline of 667 cal."
    """
    calories, proteins, fats, carbs = [], [], [], []

    for dish in metadata.values():
        if dish["calories"] <= 0:
            continue
        calories.append(dish["calories"])
        proteins.append(dish["protein"])
        fats.append(dish["fat"])
        carbs.append(dish["carb"])

    import numpy as np
    return {
        "n_dishes_with_food": len(calories),
        "calories": {
            "mean": float(np.mean(calories)),
            "median": float(np.median(calories)),
            "std": float(np.std(calories)),
            "min": float(np.min(calories)),
            "max": float(np.max(calories)),
        },
        "protein": {
            "mean": float(np.mean(proteins)),
            "median": float(np.median(proteins)),
            "std": float(np.std(proteins)),
        },
        "fat": {
            "mean": float(np.mean(fats)),
            "median": float(np.median(fats)),
            "std": float(np.std(fats)),
        },
        "carb": {
            "mean": float(np.mean(carbs)),
            "median": float(np.median(carbs)),
            "std": float(np.std(carbs)),
        },
    }


def generate_ground_truth():
    """Main function: build ground truth for all test dishes (v2 — 20 questions)."""

    print("=" * 55)
    print("NutriVision — Ground Truth Generator (v2)")
    print("=" * 55)

    # ── Load all metadata ──
    metadata = load_metadata()
    print(f"Loaded metadata for {len(metadata)} dishes")

    # ── Load test IDs ──
    with open(config.SPLITS_DIR / "test_ids.json") as f:
        test_ids = json.load(f)
    print(f"Test set: {len(test_ids)} dishes")

    # ── Print thresholds (for transparency) ──
    print(f"\nVQA thresholds (based on dietary guidelines):")
    print(f"  High calorie:  > {VQA_THRESHOLDS['high_calorie']} cal/meal "
          f"(FDA DV: 2000 cal/day ÷ 3)")
    print(f"  Calorie watch: < {VQA_THRESHOLDS['calorie_watching']} cal/meal "
          f"(weight management target)")
    print(f"  High protein:  > {VQA_THRESHOLDS['high_protein']}g/meal "
          f"(≥50% of FDA DV 50g/day)")
    print(f"  High fat:      > {VQA_THRESHOLDS['high_fat']}g/meal "
          f"(FDA DV: 78g/day ÷ 3)")
    print(f"  Low carb:      < {VQA_THRESHOLDS['low_carb']}g/meal "
          f"(standard low-carb threshold)")
    print(f"  Diabetic safe: < {VQA_THRESHOLDS['diabetic_carb_limit']}g carbs/meal "
          f"(ADA guideline)")
    print(f"  Balanced meal: each macro ≥ {VQA_THRESHOLDS['balanced_min_pct']}% "
          f"of total calories")
    print(f"  Healthy:       < {VQA_THRESHOLDS['healthy_max_cal']} cal AND "
          f"< {VQA_THRESHOLDS['healthy_max_fat']}g fat")
    print(f"\n  Held-out question thresholds (v2):")
    print(f"  Low fat:       < {VQA_THRESHOLDS['low_fat']}g/meal")
    print(f"  Light meal:    < {VQA_THRESHOLDS['light_meal']} cal/meal")
    print(f"  High carb:     > {VQA_THRESHOLDS['high_carb']}g/meal")
    print(f"  Weight loss:   < {VQA_THRESHOLDS['weight_loss_cal']} cal AND "
          f"< {VQA_THRESHOLDS['weight_loss_fat']}g fat")
    print(f"  Post-workout:  > {VQA_THRESHOLDS['post_workout_protein']}g protein AND "
          f"> {VQA_THRESHOLDS['post_workout_carb']}g carbs")

    # ── Compute dataset stats for context ──
    stats = compute_dataset_stats(metadata)
    print(f"\nDataset statistics ({stats['n_dishes_with_food']} dishes with food):")
    print(f"  Calories — mean: {stats['calories']['mean']:.1f}, "
          f"median: {stats['calories']['median']:.1f}, "
          f"range: {stats['calories']['min']:.0f}–{stats['calories']['max']:.0f}")
    print(f"  Protein  — mean: {stats['protein']['mean']:.1f}g, "
          f"median: {stats['protein']['median']:.1f}g")
    print(f"  Fat      — mean: {stats['fat']['mean']:.1f}g, "
          f"median: {stats['fat']['median']:.1f}g")
    print(f"  Carb     — mean: {stats['carb']['mean']:.1f}g, "
          f"median: {stats['carb']['median']:.1f}g")

    # ── All 20 VQA questions (12 seen + 8 held-out) ──
    all_questions = config.VQA_QUESTION_TEMPLATES + config.VQA_HELD_OUT_TEMPLATES
    seen_set = set(config.VQA_QUESTION_TEMPLATES)
    print(f"\nVQA questions: {len(config.VQA_QUESTION_TEMPLATES)} seen + "
          f"{len(config.VQA_HELD_OUT_TEMPLATES)} held-out = "
          f"{len(all_questions)} total")

    # ── Build ground truth for each test dish ──
    ground_truth = {}
    skipped = 0

    for dish_id in test_ids:
        if dish_id not in metadata:
            print(f"  [WARN] {dish_id} not in metadata — skipping")
            skipped += 1
            continue

        dish = metadata[dish_id]
        ingredients = get_ingredient_names(dish)

        gt = {
            "calories": dish["calories"],
            "mass": dish["mass"],
            "macros": {
                "fat": dish["fat"],
                "carb": dish["carb"],
                "protein": dish["protein"],
            },
            "ingredients": ingredients,
            "is_plate_only": (ingredients == ["plate only"] or dish["calories"] == 0),
            "vqa": {},
        }

        # Generate correct answer for ALL 20 VQA questions
        for question in all_questions:
            answer = get_vqa_answer(question, dish)
            split = "seen" if question in seen_set else "unseen"
            gt["vqa"][question] = {
                "answer": answer,
                "split": split,
            }

        ground_truth[dish_id] = gt

    print(f"\nGround truth built for {len(ground_truth)} dishes ({skipped} skipped)")

    # ── Count categories ──
    plate_only = sum(1 for gt in ground_truth.values() if gt["is_plate_only"])
    high_cal = sum(1 for gt in ground_truth.values()
                   if gt["calories"] > VQA_THRESHOLDS["high_calorie"])
    high_prot = sum(1 for gt in ground_truth.values()
                    if gt["macros"]["protein"] > VQA_THRESHOLDS["high_protein"])
    low_carb_count = sum(1 for gt in ground_truth.values()
                         if gt["macros"]["carb"] < VQA_THRESHOLDS["low_carb"])

    print(f"\nTest set breakdown:")
    print(f"  Plate-only:    {plate_only}")
    print(f"  High calorie:  {high_cal} dishes (>{VQA_THRESHOLDS['high_calorie']} cal)")
    print(f"  High protein:  {high_prot} dishes (>{VQA_THRESHOLDS['high_protein']}g)")
    print(f"  Low carb:      {low_carb_count} dishes (<{VQA_THRESHOLDS['low_carb']}g)")

    # ── Held-out question answer distribution ──
    print(f"\n  Held-out question answer distribution:")
    for q in config.VQA_HELD_OUT_TEMPLATES:
        yes_count = sum(
            1 for gt in ground_truth.values()
            if gt["vqa"][q]["answer"] == "yes" and not gt["is_plate_only"]
        )
        no_count = sum(
            1 for gt in ground_truth.values()
            if gt["vqa"][q]["answer"] == "no" and not gt["is_plate_only"]
        )
        total_q = yes_count + no_count
        short_q = q[:45] + "..." if len(q) > 45 else q
        pct = yes_count / total_q * 100 if total_q > 0 else 0
        print(f"    {short_q:48s} yes={yes_count:3d} ({pct:5.1f}%)  no={no_count:3d}")

    # ── Save ground truth ──
    output_dir = config.RESULTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    gt_path = output_dir / "v2_ground_truth.json"
    with open(gt_path, "w") as f:
        json.dump({
            "version": "v2",
            "n_seen_questions": len(config.VQA_QUESTION_TEMPLATES),
            "n_held_out_questions": len(config.VQA_HELD_OUT_TEMPLATES),
            "thresholds": VQA_THRESHOLDS,
            "dataset_stats": stats,
            "dishes": ground_truth,
        }, f, indent=2)
    print(f"\nSaved to: {gt_path}")

    # ── Print samples for manual verification ──
    print(f"\n{'='*60}")
    print("SAMPLE GROUND TRUTH (verify these against the CSV)")
    print(f"{'='*60}")

    sample_ids = [did for did in list(ground_truth.keys())
                  if not ground_truth[did]["is_plate_only"]][:3]

    for dish_id in sample_ids:
        gt = ground_truth[dish_id]
        print(f"\n{dish_id}:")
        print(f"  Calories: {gt['calories']:.1f}")
        print(f"  Macros: fat={gt['macros']['fat']:.1f}g, "
              f"carb={gt['macros']['carb']:.1f}g, "
              f"protein={gt['macros']['protein']:.1f}g")
        print(f"  Ingredients: {gt['ingredients']}")
        print(f"  VQA answers (seen):")
        for q_item in list(gt["vqa"].items())[:6]:
            q_text, q_data = q_item
            if q_data["split"] == "seen":
                short_q = q_text[:50] + "..." if len(q_text) > 50 else q_text
                print(f"    {short_q} → {q_data['answer']}")
        print(f"  VQA answers (unseen / held-out):")
        for q_text, q_data in gt["vqa"].items():
            if q_data["split"] == "unseen":
                short_q = q_text[:50] + "..." if len(q_text) > 50 else q_text
                print(f"    {short_q} → {q_data['answer']}")

    print(f"\n{'='*60}")
    print("VERIFY: grep a sample dish in the CSV to confirm values match:")
    print(f"  grep \"{sample_ids[0]}\" ~/nutrivision/data/raw/metadata/dish_metadata_cafe1.csv")
    print(f"{'='*60}")


if __name__ == "__main__":
    generate_ground_truth()