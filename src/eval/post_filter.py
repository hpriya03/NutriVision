"""
NutriVision — Post-Filter (Phase E)

A post-processing pipeline that checks model outputs for hallucinations
and reasoning errors using lookup tables and consistency checks.

────────────────────────────────────────────────────────────
WHAT THE POST-FILTER DOES (one pipeline, three types of checks):
────────────────────────────────────────────────────────────

  Check 1 — VALIDITY (is this output reasonable on its own?)
    • Ingredients: remove non-food items (qr code, plate, napkin…)
    • Ingredients: optionally flag items not in Nutrition5K vocabulary
    • Calories:    flag values outside 0–2500 range
    • Macros:      flag per-nutrient values outside reasonable ranges

  Check 2 — INTERNAL CONSISTENCY (do Task 5 VQA answers agree
            with each other?)
    • The model answers 4 numeric VQA questions (calories, protein,
      fat, carbs) and 14 yes/no VQA questions about those same values.
    • We take the model's OWN numeric answers, apply VQA_THRESHOLDS,
      and derive what each yes/no answer SHOULD be.
    • If the model's yes/no answer contradicts its own numbers →
      REPLACE with the derived answer.
    • Example: model says protein = 35g but answers "no" to
      "Is this high in protein?" (threshold 25g). 35 > 25 → should
      be "yes" → replace "no" with "yes".

  Check 3 — CROSS-TASK CONSISTENCY (do outputs from different tasks
            agree with each other?)
    • Atwater: calories (Task 3) ≈ protein×4 + carbs×4 + fat×9 (Task 4)
    • Task 3 calories vs VQA "How many calories?" (same question,
      asked in two different tasks — do they give the same number?)
    • Task 2 ingredients vs VQA "Is this vegetarian?" (if Task 2
      lists chicken, VQA should not say vegetarian)

────────────────────────────────────────────────────────────
LOOKUP TABLES USED:
────────────────────────────────────────────────────────────
  • NON_FOOD_ITEMS        — set of non-food objects to remove
  • Nutrition5K vocabulary — all 555 ingredient names (optional)
  • CALORIE_RANGE         — (0, 2500) per meal
  • MACRO_RANGES          — per-nutrient reasonable ranges
  • ATWATER constants     — protein=4, carbs=4, fat=9 cal/g
  • VQA_THRESHOLDS        — dietary guideline thresholds (from
                            ground_truth.py) — the SAME table used
                            to generate ground truth, but applied to
                            the MODEL'S predictions instead of real
                            measurements

────────────────────────────────────────────────────────────
OUTPUTS:
────────────────────────────────────────────────────────────
  • post_filtered/raw_outputs.json — corrected outputs (same format
    as the original, so evaluate.py can process it directly)
  • post_filter_report.json — per-dish details of every check
  • post_filter_summary.csv — aggregate statistics

────────────────────────────────────────────────────────────
USAGE:
────────────────────────────────────────────────────────────
  # Run on zero-shot results:
  python -m src.eval.post_filter \\
      --results-dir results/zero_shot_flan-t5-xl

  # Run on fine-tuned results:
  python -m src.eval.post_filter \\
      --results-dir results/v2_finetuned_qformer

  # Then evaluate the corrected outputs:
  python -m src.eval.evaluate \\
      --results-dir results/v2_finetuned_qformer/post_filtered

  # Compare raw vs post-filtered to see improvement:
  #   raw:           results/v2_finetuned_qformer/evaluation_results.json
  #   post-filtered: results/v2_finetuned_qformer/post_filtered/evaluation_results.json
"""

import argparse
import json
import csv
import re
from pathlib import Path

from src import config
from src.eval.ground_truth import VQA_THRESHOLDS, MEAT_FISH_KEYWORDS


# ═══════════════════════════════════════════════════════════
# PARSING HELPERS (same as evaluate.py — duplicated here so
# post_filter.py is self-contained and importable standalone)
# ═══════════════════════════════════════════════════════════

def parse_number(text: str) -> float | None:
    """Extract the first number from text."""
    if not text or not text.strip():
        return None
    match = re.search(r'(\d+\.?\d*)', text)
    return float(match.group(1)) if match else None


def parse_macros(text: str) -> dict:
    """Extract protein, fat, carb values from macro output text."""
    result = {"protein": None, "fat": None, "carb": None}
    if not text or not text.strip():
        return result

    t = text.lower()

    for nutrient in ["protein", "fat"]:
        match = re.search(rf'(\d+\.?\d*)\s*(?:g(?:rams?)?)\s*(?:of\s+)?{nutrient}', t)
        if not match:
            match = re.search(rf'{nutrient}[:\s]+(\d+\.?\d*)', t)
        if match:
            result[nutrient] = float(match.group(1))

    for carb_word in ["carbohydrate", "carbohydrates", "carbs", "carb"]:
        match = re.search(rf'(\d+\.?\d*)\s*(?:g(?:rams?)?)\s*(?:of\s+)?{carb_word}', t)
        if not match:
            match = re.search(rf'{carb_word}[:\s]+(\d+\.?\d*)', t)
        if match:
            result["carb"] = float(match.group(1))
            break

    if all(v is None for v in result.values()):
        numbers = re.findall(r'(\d+\.?\d*)', t)
        if len(numbers) >= 3:
            result["protein"] = float(numbers[0])
            result["fat"] = float(numbers[1])
            result["carb"] = float(numbers[2])

    return result


def parse_ingredients(text: str) -> list[str]:
    """Extract ingredient names from model output text."""
    if not text or not text.strip():
        return []
    t = text.lower().strip()
    for phrase in ["the ingredients are", "the ingredients in this dish are",
                   "the dish contains", "this dish contains", "it contains",
                   "it's a", "it is a",
                   "the main ingredients in this dish are",
                   "the main ingredients are"]:
        t = t.replace(phrase, "")
    t = t.replace(" and ", ",")
    parts = [p.strip().strip(".").strip() for p in t.split(",")]
    return [p for p in parts if len(p) > 1]


def parse_yes_no(text: str) -> str | None:
    """Extract yes/no from model output."""
    if not text:
        return None
    t = text.lower().strip()
    if t.startswith("yes"):
        return "yes"
    if t.startswith("no"):
        return "no"
    if "yes" in t and "no" not in t:
        return "yes"
    if "no" in t and "yes" not in t:
        return "no"
    return None


def normalize_ingredient(name: str) -> str:
    """Normalize an ingredient name for comparison."""
    n = name.lower().strip()
    if n.endswith("es") and len(n) > 3:
        n = n[:-2]
    elif n.endswith("s") and not n.endswith("ss") and len(n) > 2:
        n = n[:-1]
    return n


# ═══════════════════════════════════════════════════════════
# LOOKUP TABLES
# ═══════════════════════════════════════════════════════════

# ── Non-food items (always removed from ingredient predictions) ──
# These are objects the camera captures that the model sometimes
# lists as "ingredients" — they are never food.
NON_FOOD_ITEMS = {
    "qr code", "qr", "plate", "tray", "paper", "napkin",
    "code", "barcode", "sticker", "label", "table", "fork",
    "knife", "spoon", "bowl", "cup", "container", "wrapper",
    "box", "bag", "lid", "foil", "plastic", "cardboard",
    "water", "cloth", "towel", "receipt",
}

# ── Reasonable per-meal ranges ──
# Used to flag obviously impossible predictions.
# A single cafeteria plate in Nutrition5K: max ~1800 cal observed.
# We use generous bounds to only catch extreme hallucinations.
CALORIE_RANGE = (0, 2500)       # calories per plate
MACRO_RANGES = {
    "protein": (0, 150),         # grams per plate
    "fat":     (0, 150),         # grams per plate
    "carb":    (0, 300),         # grams per plate
}

# ── Atwater energy factors (thermodynamic constants) ──
ATWATER = {"protein": 4, "carb": 4, "fat": 9}   # cal per gram
ATWATER_TOLERANCE = 0.50   # >50% discrepancy = inconsistent


# ═══════════════════════════════════════════════════════════
# INGREDIENT VOCABULARY BUILDER
# ═══════════════════════════════════════════════════════════

def build_ingredient_vocabulary(gt_path: Path = None) -> set:
    """
    Build the set of ALL known ingredient names from Nutrition5K.

    Purpose: if the model predicts "unicorn meat" but no dish in the
    entire dataset contains that ingredient, it's likely hallucinated.

    Tries two data sources:
      1. Full metadata via metadata_parser (covers all ~5000 dishes)
      2. Ground truth JSON (covers 507 test dishes — less complete
         but still useful)

    This is NOT dish-specific ground truth. It's a generic table of
    "what foods exist in Nutrition5K" — like a dictionary of valid
    words. Checking "is chicken a real ingredient?" does not tell you
    whether THIS dish has chicken.
    """
    vocab = set()

    # ── Source 1: full metadata (best — all dishes) ──
    try:
        from src.data_prep.metadata_parser import load_metadata, get_ingredient_names
        metadata = load_metadata()
        for dish in metadata.values():
            for name in get_ingredient_names(dish):
                if name != "plate only":
                    vocab.add(name.lower().strip())
        print(f"  Ingredient vocabulary: {len(vocab)} unique items "
              f"(from full metadata, {len(metadata)} dishes)")
        return vocab
    except Exception:
        pass

    # ── Source 2: ground truth JSON (fallback — test dishes only) ──
    if gt_path and gt_path.exists():
        with open(gt_path) as f:
            gt_data = json.load(f)
        for dish in gt_data.get("dishes", {}).values():
            for name in dish.get("ingredients", []):
                if name != "plate only":
                    vocab.add(name.lower().strip())
        print(f"  Ingredient vocabulary: {len(vocab)} unique items "
              f"(from ground truth, {len(gt_data.get('dishes', {}))} dishes)")
        return vocab

    print("  Ingredient vocabulary: NOT AVAILABLE "
          "(using non-food filter only)")
    return vocab


def ingredient_in_vocabulary(item: str, vocabulary: set) -> bool:
    """
    Check if a predicted ingredient matches anything in the vocabulary.
    Uses substring containment and token overlap (same logic as
    evaluate.py's ingredient_match).
    """
    item_lower = item.lower().strip()
    item_norm = normalize_ingredient(item)

    for v in vocabulary:
        # Exact match after normalization
        if item_norm == normalize_ingredient(v):
            return True
        # Substring containment
        if item_lower in v or v in item_lower:
            return True
        # Token overlap (majority of shorter set)
        item_tokens = set(item_lower.split())
        v_tokens = set(v.split())
        if item_tokens and v_tokens:
            overlap = item_tokens & v_tokens
            smaller = min(len(item_tokens), len(v_tokens))
            if overlap and len(overlap) >= smaller * 0.5:
                return True

    return False


# ═══════════════════════════════════════════════════════════
# CHECK 1: INGREDIENT CLEANING (validity)
# ═══════════════════════════════════════════════════════════

def clean_ingredients(raw_text: str, vocabulary: set) -> dict:
    """
    Remove non-food items from the ingredient prediction.
    Optionally flag items not found in the Nutrition5K vocabulary.

    Returns dict with cleaned list, removed items, and flagged items.
    """
    predicted = parse_ingredients(raw_text)
    removed = []
    flagged = []
    kept = []

    for item in predicted:
        norm = normalize_ingredient(item)
        item_lower = item.lower().strip()

        # Level 1: known non-food → REMOVE
        if norm in NON_FOOD_ITEMS or item_lower in NON_FOOD_ITEMS:
            removed.append({"item": item, "reason": "non-food object"})
            continue

        # Level 2: not in vocabulary → FLAG (but keep — might be valid
        # food just not in Nutrition5K's 555 items)
        if vocabulary and not ingredient_in_vocabulary(item, vocabulary):
            flagged.append({"item": item, "reason": "not in vocabulary"})

        kept.append(item)

    cleaned_text = ", ".join(kept) if kept else ""

    return {
        "cleaned_text": cleaned_text,
        "cleaned_list": kept,
        "removed": removed,
        "flagged": flagged,
        "original_count": len(predicted),
        "cleaned_count": len(kept),
    }


# ═══════════════════════════════════════════════════════════
# CHECK 2: CALORIE & MACRO RANGE + ATWATER (validity + cross-task)
# ═══════════════════════════════════════════════════════════

def check_calorie_range(raw_text: str) -> dict:
    """Check if the calorie prediction falls within a reasonable range."""
    val = parse_number(raw_text)
    if val is None:
        return {"value": None, "in_range": None, "flag": "unparseable"}
    in_range = CALORIE_RANGE[0] <= val <= CALORIE_RANGE[1]
    return {
        "value": val,
        "in_range": in_range,
        "flag": None if in_range else
                f"outside [{CALORIE_RANGE[0]}, {CALORIE_RANGE[1]}]",
    }


def check_macros_and_atwater(macro_text: str, calorie_text: str) -> dict:
    """
    Check per-nutrient ranges and Atwater thermodynamic consistency.

    Atwater equation: calories ≈ protein×4 + carbs×4 + fat×9
    If the model's calorie prediction and macro predictions disagree
    by more than ATWATER_TOLERANCE, at least one of them is wrong.
    """
    macros = parse_macros(macro_text)
    cal = parse_number(calorie_text)

    # Per-nutrient range check
    range_flags = {}
    for nutrient in ["protein", "fat", "carb"]:
        val = macros[nutrient]
        if val is None:
            range_flags[nutrient] = {"value": None, "in_range": None}
        else:
            lo, hi = MACRO_RANGES[nutrient]
            ok = lo <= val <= hi
            range_flags[nutrient] = {
                "value": val,
                "in_range": ok,
                "flag": None if ok else f"outside [{lo}, {hi}]",
            }

    # Atwater cross-check (Task 3 × Task 4)
    atwater_result = {
        "predicted_calories": cal,
        "atwater_calories": None,
        "discrepancy_pct": None,
        "consistent": None,
    }

    if (macros["protein"] is not None and
        macros["fat"] is not None and
        macros["carb"] is not None and
        cal is not None and cal > 0):

        atwater_cal = (macros["protein"] * ATWATER["protein"] +
                       macros["carb"]    * ATWATER["carb"] +
                       macros["fat"]     * ATWATER["fat"])

        discrepancy = abs(cal - atwater_cal) / cal
        consistent = discrepancy <= ATWATER_TOLERANCE

        atwater_result.update({
            "atwater_calories": round(atwater_cal, 1),
            "discrepancy_pct": round(discrepancy * 100, 1),
            "consistent": consistent,
        })

    return {
        "macros": macros,
        "range_flags": range_flags,
        "atwater": atwater_result,
    }


# ═══════════════════════════════════════════════════════════
# CHECK 3: VQA YES/NO CORRECTION (internal consistency)
# ═══════════════════════════════════════════════════════════

def extract_model_numerics(vqa_outputs: list, tasks: dict) -> dict:
    """
    Extract the model's own numeric predictions.

    Primary source:  VQA numeric answers (for internal consistency —
                     VQA numbers checked against VQA yes/no).
    Fallback source: Task 3 calories / Task 4 macros (cross-task —
                     used only when a VQA numeric answer is missing).
    """
    vals = {"calories": None, "protein": None, "fat": None, "carb": None}

    # Primary: VQA numeric answers
    for qa in vqa_outputs:
        q = qa["question"].lower()
        num = parse_number(qa["output"])
        if num is None:
            continue
        if "how many calories" in q:
            vals["calories"] = num
        elif "how much protein" in q:
            vals["protein"] = num
        elif "how much fat" in q:
            vals["fat"] = num
        elif "carbohydrate content" in q:
            vals["carb"] = num

    # Fallback: Task 3 (calories) and Task 4 (macros)
    if vals["calories"] is None:
        vals["calories"] = parse_number(
            tasks.get("calories", {}).get("output", ""))

    if any(vals[n] is None for n in ["protein", "fat", "carb"]):
        task4 = parse_macros(tasks.get("macros", {}).get("output", ""))
        for n in ["protein", "fat", "carb"]:
            if vals[n] is None:
                vals[n] = task4[n]

    return vals


def derive_yes_no(question: str, model_vals: dict,
                  model_ingredients: list) -> str | None:
    """
    Apply VQA_THRESHOLDS to the model's OWN predictions to derive
    what the yes/no answer SHOULD be.

    This is the EXACT SAME logic as ground_truth.py's get_vqa_answer(),
    but applied to the model's predicted values instead of real
    Nutrition5K measurements. Same thresholds, same comparisons,
    different input source.

    Returns "yes", "no", or None (can't derive — question type is
    numeric/ingredient, or required numeric value is missing).
    """
    q = question.lower()
    cal     = model_vals.get("calories")
    protein = model_vals.get("protein")
    fat     = model_vals.get("fat")
    carb    = model_vals.get("carb")

    # ── Skip non-yes/no questions ──
    if any(p in q for p in ["how many calories", "how much protein",
                            "how much fat", "carbohydrate content",
                            "main ingredients", "food groups"]):
        return None   # numeric or ingredient list — not correctable

    # ── Original 12 seen questions ──
    if "high in protein" in q:
        if protein is None: return None
        return "yes" if protein > VQA_THRESHOLDS["high_protein"] else "no"

    if "low-carb" in q or "low carb" in q:
        if carb is None: return None
        return "yes" if carb < VQA_THRESHOLDS["low_carb"] else "no"

    if "balanced meal" in q:
        if any(v is None for v in [cal, protein, fat, carb]):
            return None
        if cal <= 0:
            return "no"
        min_pct = VQA_THRESHOLDS["balanced_min_pct"]
        p_pct = (protein * 4 / cal) * 100
        c_pct = (carb * 4 / cal) * 100
        f_pct = (fat * 9 / cal) * 100
        return "yes" if (p_pct >= min_pct and c_pct >= min_pct
                         and f_pct >= min_pct) else "no"

    # "healthy" but NOT "calorie" (avoid matching calorie-watching Q)
    if "healthy" in q and "calorie" not in q:
        if cal is None or fat is None: return None
        return ("yes" if (cal <= VQA_THRESHOLDS["healthy_max_cal"] and
                          fat <= VQA_THRESHOLDS["healthy_max_fat"])
                else "no")

    if "watching their calorie" in q or "calorie intake" in q:
        if cal is None: return None
        return "yes" if cal < VQA_THRESHOLDS["calorie_watching"] else "no"

    if "diabetic" in q:
        if carb is None: return None
        return ("yes" if carb < VQA_THRESHOLDS["diabetic_carb_limit"]
                else "no")

    # ── 8 held-out unseen questions ──
    if "low in fat" in q:
        if fat is None: return None
        return "yes" if fat < VQA_THRESHOLDS["low_fat"] else "no"

    if "more than 500 calories" in q:
        if cal is None: return None
        return "yes" if cal > 500 else "no"

    if "light meal" in q:
        if cal is None: return None
        return "yes" if cal < VQA_THRESHOLDS["light_meal"] else "no"

    if "more fat than protein" in q:
        if fat is None or protein is None: return None
        return "yes" if fat > protein else "no"

    if "high in carbohydrates" in q:
        if carb is None: return None
        return "yes" if carb > VQA_THRESHOLDS["high_carb"] else "no"

    if "weight loss" in q:
        if cal is None or fat is None: return None
        return ("yes" if (cal < VQA_THRESHOLDS["weight_loss_cal"] and
                          fat < VQA_THRESHOLDS["weight_loss_fat"])
                else "no")

    if "vegetarian" in q:
        if not model_ingredients:
            return None
        for ingr in model_ingredients:
            for keyword in MEAT_FISH_KEYWORDS:
                if keyword in ingr.lower():
                    return "no"
        return "yes"

    if "post-workout" in q or "post workout" in q:
        if protein is None or carb is None: return None
        return ("yes" if (protein > VQA_THRESHOLDS["post_workout_protein"]
                          and carb > VQA_THRESHOLDS["post_workout_carb"])
                else "no")

    return None   # unrecognised question


def correct_vqa(vqa_outputs: list, model_vals: dict,
                model_ingredients: list) -> dict:
    """
    Check each yes/no VQA answer against the model's own numeric
    predictions and correct mismatches.

    Example:
      Model says protein = 35g.
      Model answers "no" to "Is this high in protein?"
      Threshold: high_protein = 25g.  35 > 25 → should be "yes".
      → REPLACE "no" with "yes".
    """
    corrections = []
    corrected_vqa = []

    for qa in vqa_outputs:
        question = qa["question"]
        original_output = qa["output"]

        # Derive what the answer SHOULD be from model's own numbers
        derived = derive_yes_no(question, model_vals, model_ingredients)

        if derived is not None:
            model_yn = parse_yes_no(original_output)

            if model_yn is not None and model_yn != derived:
                # ── MISMATCH — correct it ──
                nums_str = str({k: v for k, v in model_vals.items()
                                if v is not None})
                corrections.append({
                    "question": question,
                    "original_answer": model_yn,
                    "corrected_to": derived,
                    "reason": (f"Model's own numbers {nums_str} "
                               f"+ threshold → derived '{derived}'"),
                    "split": qa.get("split", "unknown"),
                })
                corrected_qa = dict(qa)
                corrected_qa["output"] = derived
                corrected_qa["_post_filter"] = {
                    "corrected": True,
                    "original": original_output,
                }
                corrected_vqa.append(corrected_qa)
            else:
                # Consistent (or unparseable — leave as-is)
                corrected_vqa.append(dict(qa))
        else:
            # Not a yes/no question or missing values — pass through
            corrected_vqa.append(dict(qa))

    return {
        "corrected_vqa": corrected_vqa,
        "corrections": corrections,
        "n_corrections": len(corrections),
    }


# ═══════════════════════════════════════════════════════════
# CHECK 4: CROSS-TASK CONSISTENCY
# ═══════════════════════════════════════════════════════════

def check_cross_task(tasks: dict, vqa_outputs: list) -> dict:
    """
    Check whether different tasks give the same answer when asked
    about the same thing.

    • Task 3 "Estimate total calories" vs VQA "How many calories?"
    • Task 4 macros vs VQA macro questions
    • Task 2 ingredients vs VQA "Is this vegetarian?"
    """
    flags = []

    # ── Calories: Task 3 vs VQA ──
    task3_cal = parse_number(tasks.get("calories", {}).get("output", ""))
    for qa in vqa_outputs:
        if "how many calories" in qa["question"].lower():
            vqa_cal = parse_number(qa["output"])
            if (task3_cal is not None and vqa_cal is not None
                    and task3_cal > 0):
                diff_pct = abs(task3_cal - vqa_cal) / task3_cal * 100
                if diff_pct > 30:
                    flags.append({
                        "type": "calories_task3_vs_vqa",
                        "task3": task3_cal,
                        "vqa": vqa_cal,
                        "difference_pct": round(diff_pct, 1),
                    })

    # ── Macros: Task 4 vs VQA ──
    task4 = parse_macros(tasks.get("macros", {}).get("output", ""))
    macro_phrases = {
        "protein": "how much protein",
        "fat":     "how much fat",
        "carb":    "carbohydrate content",
    }
    for nutrient, phrase in macro_phrases.items():
        t4_val = task4[nutrient]
        for qa in vqa_outputs:
            if phrase in qa["question"].lower():
                vqa_val = parse_number(qa["output"])
                if t4_val is not None and vqa_val is not None and t4_val > 0:
                    diff_pct = abs(t4_val - vqa_val) / t4_val * 100
                    if diff_pct > 50:
                        flags.append({
                            "type": f"{nutrient}_task4_vs_vqa",
                            "task4": t4_val,
                            "vqa": vqa_val,
                            "difference_pct": round(diff_pct, 1),
                        })

    # ── Vegetarian: Task 2 ingredients vs VQA ──
    ingr_text = tasks.get("ingredients", {}).get("output", "")
    task2_ingrs = parse_ingredients(ingr_text)
    meat_found = [
        ingr for ingr in task2_ingrs
        if any(kw in ingr.lower() for kw in MEAT_FISH_KEYWORDS)
    ]

    for qa in vqa_outputs:
        if "vegetarian" in qa["question"].lower():
            vqa_yn = parse_yes_no(qa["output"])
            if meat_found and vqa_yn == "yes":
                flags.append({
                    "type": "vegetarian_contradiction",
                    "task2_meat_ingredients": meat_found,
                    "vqa_says_vegetarian": True,
                })

    return {"flags": flags, "n_inconsistencies": len(flags)}


# ═══════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════

def run_post_filter(results_dir: str, gt_path: str = None):
    """
    Run the complete post-filter on one model's raw_outputs.json.

    Produces:
      results_dir/post_filtered/raw_outputs.json  — corrected outputs
      results_dir/post_filter_report.json          — per-dish details
      results_dir/post_filter_summary.csv          — aggregate stats
    """
    results_dir = Path(results_dir)
    display_name = results_dir.name

    print("=" * 60)
    print(f"NutriVision — Post-Filter: {display_name}")
    print("=" * 60)

    # ── Load model outputs ──
    outputs_path = results_dir / "raw_outputs.json"
    with open(outputs_path) as f:
        raw_outputs = json.load(f)
    print(f"Loaded {len(raw_outputs)} model outputs from {outputs_path}")

    # ── Resolve ground truth path (for vocabulary only) ──
    if gt_path is None:
        gt_path = config.RESULTS_DIR / "v2_ground_truth.json"
        if not gt_path.exists():
            gt_path = config.RESULTS_DIR / "ground_truth.json"
    else:
        gt_path = Path(gt_path)

    # ── Build ingredient vocabulary ──
    vocabulary = build_ingredient_vocabulary(gt_path)

    # ── Process each dish ──
    corrected_outputs = {}
    dish_reports = {}

    totals = {
        "n_dishes": 0,
        "ingr_items_removed": 0,
        "ingr_items_flagged": 0,
        "cal_out_of_range": 0,
        "macro_out_of_range": 0,
        "atwater_inconsistent": 0,
        "atwater_discrepancies": [],
        "vqa_corrections": 0,
        "vqa_corrections_seen": 0,
        "vqa_corrections_unseen": 0,
        "vqa_corrections_by_question": {},
        "cross_task_inconsistencies": 0,
    }

    for dish_id, output in raw_outputs.items():
        totals["n_dishes"] += 1
        tasks = output.get("tasks", {})
        vqa_list = output.get("vqa", [])

        report = {"dish_id": dish_id}
        corrected = {"dish_id": dish_id, "tasks": {}, "vqa": []}

        # ── Check 1: Ingredient cleaning ──
        ingr_text = tasks.get("ingredients", {}).get("output", "")
        ingr_result = clean_ingredients(ingr_text, vocabulary)
        report["ingredients"] = {
            "removed": ingr_result["removed"],
            "flagged": ingr_result["flagged"],
            "original_count": ingr_result["original_count"],
            "cleaned_count": ingr_result["cleaned_count"],
        }
        totals["ingr_items_removed"] += len(ingr_result["removed"])
        totals["ingr_items_flagged"] += len(ingr_result["flagged"])

        corrected["tasks"]["ingredients"] = {
            "prompt": tasks.get("ingredients", {}).get("prompt", ""),
            "output": ingr_result["cleaned_text"],
        }

        # ── Check 2a: Calorie range ──
        cal_text = tasks.get("calories", {}).get("output", "")
        cal_result = check_calorie_range(cal_text)
        report["calorie_range"] = cal_result
        if cal_result["in_range"] is False:
            totals["cal_out_of_range"] += 1

        # Calories: flagged only, not corrected
        corrected["tasks"]["calories"] = dict(tasks.get("calories", {}))

        # ── Check 2b: Macro range + Atwater ──
        macro_text = tasks.get("macros", {}).get("output", "")
        macro_result = check_macros_and_atwater(macro_text, cal_text)
        report["macros_and_atwater"] = {
            "range_flags": macro_result["range_flags"],
            "atwater": macro_result["atwater"],
        }
        for nutrient, rf in macro_result["range_flags"].items():
            if rf.get("in_range") is False:
                totals["macro_out_of_range"] += 1
        if macro_result["atwater"]["consistent"] is False:
            totals["atwater_inconsistent"] += 1
        if macro_result["atwater"]["discrepancy_pct"] is not None:
            totals["atwater_discrepancies"].append(
                macro_result["atwater"]["discrepancy_pct"])

        # Macros: flagged only, not corrected
        corrected["tasks"]["macros"] = dict(tasks.get("macros", {}))

        # Captioning: no post-filter, pass through
        corrected["tasks"]["captioning"] = dict(
            tasks.get("captioning", {}))

        # ── Check 3: VQA yes/no correction ──
        model_vals = extract_model_numerics(vqa_list, tasks)
        model_ingredients = ingr_result["cleaned_list"]

        vqa_result = correct_vqa(vqa_list, model_vals, model_ingredients)
        report["vqa_correction"] = {
            "model_values_used": {
                k: v for k, v in model_vals.items() if v is not None
            },
            "n_corrections": vqa_result["n_corrections"],
            "corrections": vqa_result["corrections"],
        }
        totals["vqa_corrections"] += vqa_result["n_corrections"]
        for corr in vqa_result["corrections"]:
            q = corr["question"]
            totals["vqa_corrections_by_question"][q] = \
                totals["vqa_corrections_by_question"].get(q, 0) + 1
            if corr.get("split") == "seen":
                totals["vqa_corrections_seen"] += 1
            elif corr.get("split") == "unseen":
                totals["vqa_corrections_unseen"] += 1

        corrected["vqa"] = vqa_result["corrected_vqa"]

        # ── Check 4: Cross-task consistency ──
        cross_result = check_cross_task(tasks, vqa_list)
        report["cross_task"] = cross_result
        totals["cross_task_inconsistencies"] += \
            cross_result["n_inconsistencies"]

        dish_reports[dish_id] = report
        corrected_outputs[dish_id] = corrected

    # ═══════════════════════════════════════════════════════
    # SAVE RESULTS
    # ═══════════════════════════════════════════════════════

    # ── Corrected outputs → subdirectory so evaluate.py works ──
    pf_dir = results_dir / "post_filtered"
    pf_dir.mkdir(parents=True, exist_ok=True)

    corrected_path = pf_dir / "raw_outputs.json"
    with open(corrected_path, "w") as f:
        json.dump(corrected_outputs, f, indent=2)
    print(f"\nCorrected outputs → {corrected_path}")

    # ── Detailed per-dish report ──
    report_path = results_dir / "post_filter_report.json"
    with open(report_path, "w") as f:
        json.dump(dish_reports, f, indent=2)
    print(f"Per-dish report   → {report_path}")

    # ── Print summary ──
    n = totals["n_dishes"]
    avg_atwater = (sum(totals["atwater_discrepancies"])
                   / len(totals["atwater_discrepancies"])
                   if totals["atwater_discrepancies"] else 0)

    print(f"\n{'='*60}")
    print(f"POST-FILTER SUMMARY: {display_name}")
    print(f"{'='*60}")
    print(f"  Dishes processed:            {n}")

    print(f"\n  ── Check 1: Ingredient Cleaning ──")
    print(f"  Non-food items removed:      {totals['ingr_items_removed']}")
    print(f"  Items flagged (not in vocab): {totals['ingr_items_flagged']}")

    print(f"\n  ── Check 2: Range + Atwater ──")
    print(f"  Calories out of range:       {totals['cal_out_of_range']}")
    print(f"  Macro values out of range:   {totals['macro_out_of_range']}")
    print(f"  Atwater inconsistent:        {totals['atwater_inconsistent']} "
          f"({totals['atwater_inconsistent']/n*100:.1f}%)" if n else "")
    print(f"  Avg Atwater discrepancy:     {avg_atwater:.1f}%")

    print(f"\n  ── Check 3: VQA Yes/No Corrections ──")
    print(f"  Total corrections:           {totals['vqa_corrections']}")
    print(f"    Seen questions:            {totals['vqa_corrections_seen']}")
    print(f"    Unseen questions:          {totals['vqa_corrections_unseen']}")
    if totals["vqa_corrections_by_question"]:
        print(f"  Most-corrected questions:")
        sorted_q = sorted(totals["vqa_corrections_by_question"].items(),
                          key=lambda x: -x[1])
        for q, count in sorted_q[:5]:
            short_q = (q[:48] + "...") if len(q) > 48 else q
            print(f"    {count:4d}× {short_q}")

    print(f"\n  ── Check 4: Cross-Task Consistency ──")
    print(f"  Inconsistencies found:       "
          f"{totals['cross_task_inconsistencies']}")

    # ── Save summary CSV ──
    summary_path = results_dir / "post_filter_summary.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["dishes_processed", n])
        writer.writerow(["ingredients_removed", totals["ingr_items_removed"]])
        writer.writerow(["ingredients_flagged", totals["ingr_items_flagged"]])
        writer.writerow(["calories_out_of_range", totals["cal_out_of_range"]])
        writer.writerow(["macros_out_of_range", totals["macro_out_of_range"]])
        writer.writerow(["atwater_inconsistent", totals["atwater_inconsistent"]])
        writer.writerow(["atwater_avg_discrepancy_pct", f"{avg_atwater:.1f}"])
        writer.writerow(["vqa_corrections_total", totals["vqa_corrections"]])
        writer.writerow(["vqa_corrections_seen", totals["vqa_corrections_seen"]])
        writer.writerow(["vqa_corrections_unseen", totals["vqa_corrections_unseen"]])
        writer.writerow(["cross_task_inconsistencies",
                         totals["cross_task_inconsistencies"]])
    print(f"\nSummary CSV        → {summary_path}")

    # ── Next steps ──
    print(f"\n{'='*60}")
    print("NEXT STEPS:")
    print(f"  1. Evaluate CORRECTED outputs:")
    print(f"     python -m src.eval.evaluate "
          f"--results-dir {results_dir}/post_filtered")
    print(f"  2. Compare RAW vs POST-FILTERED:")
    print(f"     Raw:      {results_dir}/evaluation_results.json")
    print(f"     Filtered: {results_dir}/post_filtered/evaluation_results.json")
    print(f"{'='*60}")

    return totals


# ═══════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="NutriVision post-filter — catch hallucinations "
                    "and correct reasoning errors")
    parser.add_argument(
        "--results-dir", type=str, required=True,
        help="Path to results directory containing raw_outputs.json "
             "(e.g. results/zero_shot_flan-t5-xl)")
    parser.add_argument(
        "--ground-truth", type=str, default=None,
        help="Path to ground truth JSON — used ONLY for building the "
             "ingredient vocabulary, not for evaluation "
             "(default: results/v2_ground_truth.json)")
    args = parser.parse_args()

    run_post_filter(
        results_dir=args.results_dir,
        gt_path=args.ground_truth,
    )