"""
NutriVision — Evaluation Script (Phase B5, v2)

Compares model outputs (raw_outputs.json) against ground truth
and computes metrics with bootstrapped 95% CIs.

v2 changes:
    - Added --results-dir flag (replaces --model for flexible paths)
    - Added --ground-truth flag (default: v2_ground_truth.json)
    - VQA metrics are now split into SEEN / UNSEEN / OVERALL
      to measure generalisation to held-out questions.
    - Ground truth format: each VQA entry is {"answer": "...", "split": "seen"|"unseen"}
    - Backward compatible: if ground truth entry is a plain string, treats as seen.
    - Backward compatible: if VQA output has no "split" tag, infers from config.

What it does:
    1. Loads raw model outputs + ground truth
    2. Parses model text into structured values (extracts numbers, ingredient lists)
    3. Computes per-dish scores for each task
    4. Aggregates into summary metrics with bootstrapped 95% CIs
    5. Saves CSV summary tables

Metrics by task:
    Task 1 (Captioning):  BLEU-1/2/4, ROUGE-1, ROUGE-L, METEOR
    Task 2 (Ingredients): Precision, Recall, F1, Hallucination rate
    Task 3 (Calories):    MAE, MAPE, Hallucination rate (>100% error)
    Task 4 (Macros):      MAE per nutrient, MAPE per nutrient
    Task 5 (VQA):         Accuracy (yes/no), MAE (numeric), Hallucination rate
                          — reported for SEEN, UNSEEN, and OVERALL separately
    All metrics:          Bootstrapped 95% CI (1000 resamples)

Usage:
    # Evaluate v2 fine-tuned results:
    python -m src.eval.evaluate --results-dir results/v2_finetuned_qformer

    # Evaluate zero-shot results (with old ground truth):
    python -m src.eval.evaluate --results-dir results/zero_shot_flan-t5-xl \\
        --ground-truth results/ground_truth.json

    # Legacy mode (backward compatible):
    python -m src.eval.evaluate --model flan-t5-xl
"""

import argparse
import json
import re
import csv
from pathlib import Path

import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score as nltk_meteor
from rouge_score import rouge_scorer

from src import config


# ── Constants ──
N_BOOTSTRAP = 1000
BOOTSTRAP_SEED = 42
HALLUCINATION_CALORIE_THRESHOLD = 1.0  # >100% error = hallucination


# ═══════════════════════════════════════════════════════════
# PARSING HELPERS — extract structured data from model text
# ═══════════════════════════════════════════════════════════

def parse_number(text: str) -> float | None:
    """
    Extract the first number from model output text.

    Examples:
        "about 240 calories"  → 240.0
        "4"                   → 4.0
        "1.5g protein"        → 1.5
        "This dish contains approximately 350 calories." → 350.0
        "a plate of chicken"  → None (no number found)
    """
    if not text or not text.strip():
        return None
    # Look for integers or decimals
    match = re.search(r'(\d+\.?\d*)', text)
    if match:
        return float(match.group(1))
    return None


def parse_macros(text: str) -> dict:
    """
    Extract protein, fat, carb values from model macro output.

    Handles formats like:
        "1.5g protein, 1g fat, and 1g carbohydrate"
        "0 grams of protein 0 grams of fat 0 grams of carbohydrates"
        "protein: 10g, fat: 5g, carbs: 20g"
        "1, 2, and 3"  → ambiguous, try positional assignment
    """
    result = {"protein": None, "fat": None, "carb": None}

    if not text or not text.strip():
        return result

    t = text.lower()

    # Try labeled patterns first: "Xg protein" or "protein: Xg" or "X grams of protein"
    for nutrient in ["protein", "fat"]:
        # "Xg protein" or "X grams of protein" or "X g protein"
        match = re.search(rf'(\d+\.?\d*)\s*(?:g(?:rams?)?)\s*(?:of\s+)?{nutrient}', t)
        if not match:
            # "protein: Xg" or "protein Xg"
            match = re.search(rf'{nutrient}[:\s]+(\d+\.?\d*)', t)
        if match:
            result[nutrient] = float(match.group(1))

    # Carb has multiple spellings
    for carb_word in ["carbohydrate", "carbohydrates", "carbs", "carb"]:
        match = re.search(rf'(\d+\.?\d*)\s*(?:g(?:rams?)?)\s*(?:of\s+)?{carb_word}', t)
        if not match:
            match = re.search(rf'{carb_word}[:\s]+(\d+\.?\d*)', t)
        if match:
            result["carb"] = float(match.group(1))
            break

    # If no labeled values found, try to extract all numbers and assign positionally
    # Common format: "1, 2, and 3" or "10 5 20"
    if all(v is None for v in result.values()):
        numbers = re.findall(r'(\d+\.?\d*)', t)
        if len(numbers) >= 3:
            # Assume order: protein, fat, carb (most common in model outputs)
            result["protein"] = float(numbers[0])
            result["fat"] = float(numbers[1])
            result["carb"] = float(numbers[2])

    return result


def parse_ingredients(text: str) -> list[str]:
    """
    Extract ingredient names from model output.

    Handles formats like:
        "Brussel sprouts, green peppers, and olives"
        "chicken, rice, qr code"
        "it's a plate" → ["plate"] or empty
    """
    if not text or not text.strip():
        return []

    # Remove common filler phrases
    t = text.lower().strip()
    for phrase in ["the ingredients are", "the ingredients in this dish are",
                   "the dish contains", "this dish contains", "it contains",
                   "it's a", "it is a",
                   "the main ingredients in this dish are",
                   "the main ingredients are"]:
        t = t.replace(phrase, "")

    # Split by commas and "and"
    t = t.replace(" and ", ",")
    parts = [p.strip().strip(".").strip() for p in t.split(",")]
    # Filter out empty strings and very short non-food words
    ingredients = [p for p in parts if len(p) > 1]
    return ingredients


def parse_yes_no(text: str) -> str | None:
    """
    Extract yes/no from model output.

    Handles both short answers ("yes", "no") and sentence-level
    answers that start with "Yes, ..." or "No, ...".

    Returns "yes", "no", or None if unclear.
    """
    if not text:
        return None
    t = text.lower().strip()

    # Check for clear yes/no at the start
    if t.startswith("yes"):
        return "yes"
    if t.startswith("no"):
        return "no"

    # Check anywhere in text
    if "yes" in t and "no" not in t:
        return "yes"
    if "no" in t and "yes" not in t:
        return "no"

    return None


# ═══════════════════════════════════════════════════════════
# INGREDIENT MATCHING — fuzzy match predicted vs ground truth
# ═══════════════════════════════════════════════════════════

def normalize_ingredient(name: str) -> str:
    """Normalize an ingredient name for comparison."""
    n = name.lower().strip()
    # Remove plurals (simple)
    if n.endswith("es") and len(n) > 3:
        n = n[:-2]  # olives → oliv
    elif n.endswith("s") and not n.endswith("ss") and len(n) > 2:
        n = n[:-1]  # sprouts → sprout
    return n


def ingredient_match(pred: str, truth: str) -> bool:
    """
    Check if a predicted ingredient matches a ground truth ingredient.
    Uses token overlap — if most words in the shorter name appear in the longer one.
    """
    pred_norm = normalize_ingredient(pred)
    truth_norm = normalize_ingredient(truth)

    # Exact match after normalization
    if pred_norm == truth_norm:
        return True

    # One contains the other
    if pred_norm in truth_norm or truth_norm in pred_norm:
        return True

    # Token overlap: check if most tokens match
    pred_tokens = set(pred_norm.split())
    truth_tokens = set(truth_norm.split())
    if not pred_tokens or not truth_tokens:
        return False

    overlap = pred_tokens & truth_tokens
    # Match if majority of smaller set overlaps
    smaller = min(len(pred_tokens), len(truth_tokens))
    return len(overlap) >= smaller * 0.5 and len(overlap) > 0


def compute_ingredient_scores(predicted: list[str], ground_truth: list[str]) -> dict:
    """
    Compute precision, recall, F1 for ingredient prediction.
    Returns per-dish scores + lists of hallucinated ingredients.
    """
    if not ground_truth or ground_truth == ["plate only"]:
        # Skip plate-only dishes
        return {"precision": None, "recall": None, "f1": None,
                "hallucinated": predicted, "n_hallucinated": len(predicted)}

    if not predicted:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0,
                "hallucinated": [], "n_hallucinated": 0}

    # Filter out known non-food items
    non_food = {"qr code", "qr", "plate", "tray", "paper", "water", "napkin",
                "code", "barcode", "sticker", "label"}
    filtered_pred = [p for p in predicted
                     if normalize_ingredient(p) not in non_food and p.strip()]

    # Match each predicted ingredient to ground truth
    matched_truth = set()
    matched_pred = set()
    hallucinated = []

    for i, pred in enumerate(filtered_pred):
        found = False
        for j, truth in enumerate(ground_truth):
            if j not in matched_truth and ingredient_match(pred, truth):
                matched_truth.add(j)
                matched_pred.add(i)
                found = True
                break
        if not found:
            hallucinated.append(pred)

    tp = len(matched_pred)
    precision = tp / len(filtered_pred) if filtered_pred else 0.0
    recall = tp / len(ground_truth) if ground_truth else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "hallucinated": hallucinated,
        "n_hallucinated": len(hallucinated),
    }


# ═══════════════════════════════════════════════════════════
# CAPTIONING METRICS — BLEU, ROUGE, METEOR
# ═══════════════════════════════════════════════════════════

def build_pseudo_reference(ingredients: list[str]) -> str:
    """
    Build a pseudo-reference caption from ground truth ingredients.
    Since Nutrition5K has no human captions, we construct one from metadata.
    """
    if not ingredients or ingredients == ["plate only"]:
        return "an empty plate"

    if len(ingredients) == 1:
        return f"a plate of {ingredients[0]}"
    else:
        listed = ", ".join(ingredients[:-1]) + f" and {ingredients[-1]}"
        return f"a plate of {listed}"


def compute_caption_scores(output: str, reference: str) -> dict:
    """Compute BLEU-1/2/4, ROUGE-1, ROUGE-L, METEOR for one caption pair."""
    if not output or not output.strip():
        return {"bleu1": 0, "bleu2": 0, "bleu4": 0,
                "rouge1": 0, "rougeL": 0, "meteor": 0}

    # Tokenize
    ref_tokens = reference.lower().split()
    out_tokens = output.lower().split()

    # BLEU with smoothing (handles short sentences)
    smooth = SmoothingFunction().method1
    bleu1 = sentence_bleu([ref_tokens], out_tokens,
                          weights=(1, 0, 0, 0), smoothing_function=smooth)
    bleu2 = sentence_bleu([ref_tokens], out_tokens,
                          weights=(0.5, 0.5, 0, 0), smoothing_function=smooth)
    bleu4 = sentence_bleu([ref_tokens], out_tokens,
                          weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth)

    # ROUGE
    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
    rouge_scores = scorer.score(reference, output)

    # METEOR
    try:
        meteor = nltk_meteor([ref_tokens], out_tokens)
    except Exception:
        meteor = 0.0

    return {
        "bleu1": bleu1,
        "bleu2": bleu2,
        "bleu4": bleu4,
        "rouge1": rouge_scores["rouge1"].fmeasure,
        "rougeL": rouge_scores["rougeL"].fmeasure,
        "meteor": meteor,
    }


# ═══════════════════════════════════════════════════════════
# BOOTSTRAPPING — confidence intervals for every metric
# ═══════════════════════════════════════════════════════════

def bootstrap_ci(values: list[float], n_bootstrap: int = N_BOOTSTRAP,
                 seed: int = BOOTSTRAP_SEED) -> dict:
    """
    Compute bootstrapped 95% CI for a list of per-dish scores.

    Args:
        values: list of per-dish metric values (e.g., absolute errors)
    Returns:
        dict with mean, ci_lower (2.5th percentile), ci_upper (97.5th percentile)
    """
    if not values:
        return {"mean": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "n": 0}

    values = np.array(values)
    rng = np.random.RandomState(seed)

    # Resample 1000 times, compute mean each time
    boot_means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(values, size=len(values), replace=True)
        boot_means.append(np.mean(sample))

    boot_means = np.array(boot_means)

    return {
        "mean": float(np.mean(values)),
        "ci_lower": float(np.percentile(boot_means, 2.5)),
        "ci_upper": float(np.percentile(boot_means, 97.5)),
        "n": len(values),
    }


def format_ci(result: dict, decimals: int = 2) -> str:
    """Format a bootstrapped result as 'mean [CI: lower – upper]'."""
    fmt = f".{decimals}f"
    return (f"{result['mean']:{fmt}} "
            f"[{result['ci_lower']:{fmt}} – {result['ci_upper']:{fmt}}]")


# ═══════════════════════════════════════════════════════════
# GROUND TRUTH HELPERS — handle v1 and v2 formats
# ═══════════════════════════════════════════════════════════

def get_gt_vqa_answer(gt_vqa_entry) -> str:
    """
    Extract the short answer from a ground truth VQA entry.
    Handles both formats:
        v1: gt["vqa"][question] = "yes"          (plain string)
        v2: gt["vqa"][question] = {"answer": "yes", "split": "seen"}
    """
    if isinstance(gt_vqa_entry, dict):
        return gt_vqa_entry["answer"]
    return gt_vqa_entry  # v1 format — already a string


def get_gt_vqa_split(gt_vqa_entry, question: str) -> str:
    """
    Determine whether a VQA question is seen or unseen.
    Uses the ground truth tag if available (v2), otherwise infers from config.
    """
    if isinstance(gt_vqa_entry, dict) and "split" in gt_vqa_entry:
        return gt_vqa_entry["split"]
    # Fallback: check against config templates
    if question in set(config.VQA_QUESTION_TEMPLATES):
        return "seen"
    if question in set(config.VQA_HELD_OUT_TEMPLATES):
        return "unseen"
    return "unknown"


# ═══════════════════════════════════════════════════════════
# MAIN EVALUATION
# ═══════════════════════════════════════════════════════════

def evaluate_model(results_dir: str, gt_path: str = None):
    """
    Run full evaluation for one model/experiment.

    Args:
        results_dir: Path to the results directory containing raw_outputs.json
        gt_path: Path to ground truth JSON (default: results/v2_ground_truth.json)
    """

    results_dir = Path(results_dir)
    display_name = results_dir.name

    print("=" * 60)
    print(f"NutriVision — Evaluation: {display_name}")
    print("=" * 60)

    # ── Load data ──
    outputs_path = results_dir / "raw_outputs.json"
    if gt_path is None:
        gt_path = config.RESULTS_DIR / "v2_ground_truth.json"
        # Fallback to v1 if v2 doesn't exist
        if not gt_path.exists():
            gt_path = config.RESULTS_DIR / "ground_truth.json"
    else:
        gt_path = Path(gt_path)

    with open(outputs_path) as f:
        raw_outputs = json.load(f)
    with open(gt_path) as f:
        gt_data = json.load(f)

    ground_truth = gt_data["dishes"]
    thresholds = gt_data["thresholds"]
    gt_version = gt_data.get("version", "v1")

    print(f"Loaded {len(raw_outputs)} model outputs from {outputs_path}")
    print(f"Loaded {len(ground_truth)} ground truth dishes from {gt_path}")
    print(f"Ground truth version: {gt_version}")

    # Ensure NLTK data is available
    import nltk
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)

    # ── Per-dish scoring ──
    # Collect scores for each task
    caption_scores = []     # list of dicts with bleu1, bleu2, etc.
    ingredient_scores = []  # list of dicts with precision, recall, f1
    calorie_errors = []     # list of absolute errors
    calorie_pct_errors = [] # list of percentage errors
    calorie_halluc = []     # list of 0/1 flags
    macro_errors = {"protein": [], "fat": [], "carb": []}
    macro_pct_errors = {"protein": [], "fat": [], "carb": []}

    # VQA — separate by split
    vqa_scores = {
        "seen": {"yesno_correct": [], "numeric_errors": [], "halluc": []},
        "unseen": {"yesno_correct": [], "numeric_errors": [], "halluc": []},
        "overall": {"yesno_correct": [], "numeric_errors": [], "halluc": []},
    }

    n_calorie_unparseable = 0
    n_macro_unparseable = 0
    n_vqa_unparseable = {"seen": 0, "unseen": 0, "overall": 0}

    for dish_id, output in raw_outputs.items():
        if dish_id not in ground_truth:
            continue

        gt = ground_truth[dish_id]

        # Skip plate-only for numeric tasks
        is_plate_only = gt.get("is_plate_only", False)

        # ── Task 1: Captioning ──
        caption_text = output["tasks"].get("captioning", {}).get("output", "")
        reference = build_pseudo_reference(gt["ingredients"])
        cap_scores = compute_caption_scores(caption_text, reference)
        caption_scores.append(cap_scores)

        # ── Task 2: Ingredients ──
        ingr_text = output["tasks"].get("ingredients", {}).get("output", "")
        pred_ingredients = parse_ingredients(ingr_text)
        ingr_scores = compute_ingredient_scores(pred_ingredients, gt["ingredients"])
        if ingr_scores["precision"] is not None:  # skip plate-only
            ingredient_scores.append(ingr_scores)

        # ── Task 3: Calories ──
        if not is_plate_only:
            cal_text = output["tasks"].get("calories", {}).get("output", "")
            pred_cal = parse_number(cal_text)
            if pred_cal is not None and gt["calories"] > 0:
                abs_err = abs(pred_cal - gt["calories"])
                pct_err = abs_err / gt["calories"]
                calorie_errors.append(abs_err)
                calorie_pct_errors.append(pct_err)
                calorie_halluc.append(1 if pct_err > HALLUCINATION_CALORIE_THRESHOLD else 0)
            else:
                n_calorie_unparseable += 1

        # ── Task 4: Macros ──
        if not is_plate_only:
            macro_text = output["tasks"].get("macros", {}).get("output", "")
            pred_macros = parse_macros(macro_text)
            any_parsed = False
            for nutrient in ["protein", "fat", "carb"]:
                if pred_macros[nutrient] is not None:
                    gt_val = gt["macros"][nutrient]
                    abs_err = abs(pred_macros[nutrient] - gt_val)
                    macro_errors[nutrient].append(abs_err)
                    if gt_val > 0:
                        macro_pct_errors[nutrient].append(abs_err / gt_val)
                    any_parsed = True
            if not any_parsed:
                n_macro_unparseable += 1

        # ── Task 5: VQA (with seen/unseen split) ──
        for qa in output.get("vqa", []):
            question = qa["question"]
            model_answer = qa["output"]

            if question not in gt["vqa"]:
                continue

            gt_vqa_entry = gt["vqa"][question]
            correct_answer = get_gt_vqa_answer(gt_vqa_entry)

            # Determine split
            # First try the output tag, then the ground truth tag, then infer
            if "split" in qa:
                split = qa["split"]
            else:
                split = get_gt_vqa_split(gt_vqa_entry, question)

            # Normalize split to seen/unseen
            if split not in ("seen", "unseen"):
                split = "seen"  # default for unknown

            # Determine question type from ground truth answer
            is_numeric = correct_answer.replace(".", "").replace("-", "").isdigit()

            if is_numeric:
                # Numeric VQA
                pred_val = parse_number(model_answer)
                true_val = float(correct_answer)
                if pred_val is not None and true_val > 0:
                    err = abs(pred_val - true_val)
                    vqa_scores[split]["numeric_errors"].append(err)
                    vqa_scores["overall"]["numeric_errors"].append(err)
                    halluc_flag = 1 if err / true_val > HALLUCINATION_CALORIE_THRESHOLD else 0
                    vqa_scores[split]["halluc"].append(halluc_flag)
                    vqa_scores["overall"]["halluc"].append(halluc_flag)
                else:
                    n_vqa_unparseable[split] += 1
                    n_vqa_unparseable["overall"] += 1
            else:
                # Yes/no or text VQA
                pred_yn = parse_yes_no(model_answer)
                if pred_yn is not None:
                    is_correct = (pred_yn == correct_answer.lower())
                    flag = 1 if is_correct else 0
                    vqa_scores[split]["yesno_correct"].append(flag)
                    vqa_scores["overall"]["yesno_correct"].append(flag)
                else:
                    n_vqa_unparseable[split] += 1
                    n_vqa_unparseable["overall"] += 1

    # ═══════════════════════════════════════════════════════
    # AGGREGATE + BOOTSTRAP
    # ═══════════════════════════════════════════════════════

    print(f"\n{'='*60}")
    print(f"RESULTS: {display_name}")
    print(f"{'='*60}")

    results = {}

    # ── Task 1: Captioning ──
    print(f"\n--- Task 1: Captioning (What's On Your Plate) ---")
    for metric in ["bleu1", "bleu2", "bleu4", "rouge1", "rougeL", "meteor"]:
        values = [s[metric] for s in caption_scores]
        result = bootstrap_ci(values)
        results[f"caption_{metric}"] = result
        print(f"  {metric:10s}: {format_ci(result, 4)}")

    # ── Task 2: Ingredients ──
    print(f"\n--- Task 2: Ingredients (Ingredients Detected) ---")
    for metric in ["precision", "recall", "f1"]:
        values = [s[metric] for s in ingredient_scores if s[metric] is not None]
        result = bootstrap_ci(values)
        results[f"ingr_{metric}"] = result
        print(f"  {metric:10s}: {format_ci(result, 4)}")

    # Hallucination rate for ingredients
    halluc_values = [1 if s["n_hallucinated"] > 0 else 0
                     for s in ingredient_scores if s["precision"] is not None]
    result = bootstrap_ci(halluc_values)
    results["ingr_halluc_rate"] = result
    print(f"  {'halluc_rate':10s}: {format_ci(result, 4)}")
    total_halluc_ingr = sum(s["n_hallucinated"] for s in ingredient_scores)
    print(f"  Total hallucinated ingredients: {total_halluc_ingr}")

    # ── Task 3: Calories ──
    print(f"\n--- Task 3: Calories (Estimated Calories) ---")
    result = bootstrap_ci(calorie_errors)
    results["cal_mae"] = result
    print(f"  {'MAE':10s}: {format_ci(result, 2)} cal")

    result = bootstrap_ci([e * 100 for e in calorie_pct_errors])
    results["cal_mape"] = result
    print(f"  {'MAPE':10s}: {format_ci(result, 2)}%")

    result = bootstrap_ci(calorie_halluc)
    results["cal_halluc_rate"] = result
    print(f"  {'halluc_rate':10s}: {format_ci(result, 4)}")
    print(f"  Unparseable calorie outputs: {n_calorie_unparseable}")

    # ── Task 4: Macros ──
    print(f"\n--- Task 4: Macros (Nutrition Breakdown) ---")
    for nutrient in ["protein", "fat", "carb"]:
        result = bootstrap_ci(macro_errors[nutrient])
        results[f"macro_{nutrient}_mae"] = result
        print(f"  {nutrient + ' MAE':15s}: {format_ci(result, 2)}g")

        if macro_pct_errors[nutrient]:
            result = bootstrap_ci([e * 100 for e in macro_pct_errors[nutrient]])
            results[f"macro_{nutrient}_mape"] = result
            print(f"  {nutrient + ' MAPE':15s}: {format_ci(result, 2)}%")

    print(f"  Unparseable macro outputs: {n_macro_unparseable}")

    # ── Task 5: VQA (split by seen/unseen/overall) ──
    for split_name in ["seen", "unseen", "overall"]:
        split_data = vqa_scores[split_name]
        label = split_name.upper()
        has_data = (split_data["yesno_correct"] or split_data["numeric_errors"])

        if not has_data and split_name != "overall":
            continue  # Skip empty splits (e.g. old results with no unseen)

        print(f"\n--- Task 5: VQA — {label} ---")

        if split_data["yesno_correct"]:
            result = bootstrap_ci([v * 100 for v in split_data["yesno_correct"]])
            results[f"vqa_{split_name}_accuracy"] = result
            correct = sum(split_data["yesno_correct"])
            total_yn = len(split_data["yesno_correct"])
            print(f"  {'Accuracy':10s}: {format_ci(result, 2)}% "
                  f"({correct}/{total_yn} correct)")

        if split_data["numeric_errors"]:
            result = bootstrap_ci(split_data["numeric_errors"])
            results[f"vqa_{split_name}_numeric_mae"] = result
            print(f"  {'Numeric MAE':10s}: {format_ci(result, 2)}")

        if split_data["halluc"]:
            result = bootstrap_ci(split_data["halluc"])
            results[f"vqa_{split_name}_halluc_rate"] = result
            print(f"  {'halluc_rate':10s}: {format_ci(result, 4)}")

        print(f"  Unparseable: {n_vqa_unparseable.get(split_name, 0)}")

    # ═══════════════════════════════════════════════════════
    # SAVE RESULTS
    # ═══════════════════════════════════════════════════════

    output_dir = results_dir

    # ── Save full results as JSON ──
    json_path = output_dir / "evaluation_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results saved to: {json_path}")

    # ── Save summary CSV ──
    csv_path = output_dir / "summary_metrics.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["task", "metric", "mean", "ci_lower", "ci_upper", "n"])
        for key, result in results.items():
            # Parse task and metric from key
            parts = key.split("_", 1)
            task = parts[0]
            metric = parts[1] if len(parts) > 1 else key
            writer.writerow([
                task, metric,
                f"{result['mean']:.4f}",
                f"{result['ci_lower']:.4f}",
                f"{result['ci_upper']:.4f}",
                result["n"],
            ])
    print(f"Summary CSV saved to: {csv_path}")

    # ── Save per-dish scores CSV (for detailed analysis) ──
    perdish_path = output_dir / "per_dish_scores.csv"
    with open(perdish_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "dish_id", "is_plate_only",
            "caption_bleu1", "caption_rouge1", "caption_meteor",
            "ingr_precision", "ingr_recall", "ingr_f1", "ingr_n_hallucinated",
            "cal_error", "cal_pct_error",
            "macro_protein_error", "macro_fat_error", "macro_carb_error",
        ])

        dish_idx = {"cal": 0, "ingr": 0}
        for dish_id, output in raw_outputs.items():
            if dish_id not in ground_truth:
                continue

            gt = ground_truth[dish_id]
            is_plate_only = gt.get("is_plate_only", False)

            row = [dish_id, is_plate_only]

            # Caption scores
            if dish_idx.get("cap_idx", 0) < len(caption_scores):
                cap = caption_scores[dish_idx.get("cap_idx", 0)]
                row.extend([f"{cap['bleu1']:.4f}", f"{cap['rouge1']:.4f}",
                           f"{cap['meteor']:.4f}"])
                dish_idx["cap_idx"] = dish_idx.get("cap_idx", 0) + 1
            else:
                row.extend(["", "", ""])

            # Ingredient scores
            if not is_plate_only and dish_idx.get("ingr_idx", 0) < len(ingredient_scores):
                ing = ingredient_scores[dish_idx.get("ingr_idx", 0)]
                row.extend([f"{ing['precision']:.4f}", f"{ing['recall']:.4f}",
                           f"{ing['f1']:.4f}", ing["n_hallucinated"]])
                dish_idx["ingr_idx"] = dish_idx.get("ingr_idx", 0) + 1
            else:
                row.extend(["", "", "", ""])

            # Calorie error
            if not is_plate_only:
                cal_text = output["tasks"].get("calories", {}).get("output", "")
                pred_cal = parse_number(cal_text)
                if pred_cal is not None and gt["calories"] > 0:
                    err = abs(pred_cal - gt["calories"])
                    pct = err / gt["calories"]
                    row.extend([f"{err:.2f}", f"{pct:.4f}"])
                else:
                    row.extend(["", ""])
            else:
                row.extend(["", ""])

            # Macro errors
            if not is_plate_only:
                macro_text = output["tasks"].get("macros", {}).get("output", "")
                pred_macros = parse_macros(macro_text)
                for nutrient in ["protein", "fat", "carb"]:
                    if pred_macros[nutrient] is not None:
                        err = abs(pred_macros[nutrient] - gt["macros"][nutrient])
                        row.append(f"{err:.2f}")
                    else:
                        row.append("")
            else:
                row.extend(["", "", ""])

            writer.writerow(row)

    print(f"Per-dish scores saved to: {perdish_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NutriVision evaluation (v2)")

    # v2: flexible results directory
    parser.add_argument("--results-dir", type=str, default=None,
                        help="Path to results directory containing raw_outputs.json "
                             "(e.g. results/v2_finetuned_qformer)")

    # v2: ground truth path
    parser.add_argument("--ground-truth", type=str, default=None,
                        help="Path to ground truth JSON "
                             "(default: results/v2_ground_truth.json)")

    # Legacy: --model flag for backward compatibility
    parser.add_argument("--model", type=str, default=None,
                        help="(Legacy) Model key — constructs results path as "
                             "results/zero_shot_{model}/")

    args = parser.parse_args()

    # Determine results directory
    if args.results_dir:
        results_dir = args.results_dir
    elif args.model:
        results_dir = str(config.RESULTS_DIR / f"zero_shot_{args.model}")
    else:
        parser.error("Provide either --results-dir or --model")

    evaluate_model(results_dir=results_dir, gt_path=args.ground_truth)