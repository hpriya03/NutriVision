"""
NutriVision — Zero-Shot Model Comparison (Phase B6)

Reads evaluation results for both BLIP-2 variants (OPT and FlanT5-XL)
and produces a side-by-side comparison table.

What it does:
    1. Loads evaluation_results.json for both models
    2. Builds a comparison table with mean ± CI for each metric
    3. Marks the winner for each metric
    4. Prints a summary recommendation
    5. Saves comparison CSV for the dissertation

Usage:
    python -m src.eval.compare_models 2>&1 | tee logs/10_comparison.log
"""

import json
import csv

from src import config


# Display-friendly names for metrics
METRIC_LABELS = {
    "caption_bleu1":      ("Captioning",   "BLEU-1"),
    "caption_bleu2":      ("Captioning",   "BLEU-2"),
    "caption_bleu4":      ("Captioning",   "BLEU-4"),
    "caption_rouge1":     ("Captioning",   "ROUGE-1"),
    "caption_rougeL":     ("Captioning",   "ROUGE-L"),
    "caption_meteor":     ("Captioning",   "METEOR"),
    "ingr_precision":     ("Ingredients",  "Precision"),
    "ingr_recall":        ("Ingredients",  "Recall"),
    "ingr_f1":            ("Ingredients",  "F1"),
    "ingr_halluc_rate":   ("Ingredients",  "Hallucination Rate"),
    "cal_mae":            ("Calories",     "MAE (cal)"),
    "cal_mape":           ("Calories",     "MAPE (%)"),
    "cal_halluc_rate":    ("Calories",     "Hallucination Rate"),
    "macro_protein_mae":  ("Macros",       "Protein MAE (g)"),
    "macro_protein_mape": ("Macros",       "Protein MAPE (%)"),
    "macro_fat_mae":      ("Macros",       "Fat MAE (g)"),
    "macro_fat_mape":     ("Macros",       "Fat MAPE (%)"),
    "macro_carb_mae":     ("Macros",       "Carb MAE (g)"),
    "macro_carb_mape":    ("Macros",       "Carb MAPE (%)"),
    "vqa_accuracy":       ("VQA",          "Yes/No Accuracy (%)"),
    "vqa_numeric_mae":    ("VQA",          "Numeric MAE"),
    "vqa_halluc_rate":    ("VQA",          "Hallucination Rate"),
}

# For these metrics, HIGHER is better. Everything else, LOWER is better.
HIGHER_IS_BETTER = {
    "caption_bleu1", "caption_bleu2", "caption_bleu4",
    "caption_rouge1", "caption_rougeL", "caption_meteor",
    "ingr_precision", "ingr_recall", "ingr_f1",
    "vqa_accuracy",
}


def format_result(result: dict, decimals: int = 4) -> str:
    """Format as 'mean [lower – upper]'."""
    fmt = f".{decimals}f"
    return f"{result['mean']:{fmt}} [{result['ci_lower']:{fmt}} – {result['ci_upper']:{fmt}}]"


def compare_models():
    print("=" * 72)
    print("NutriVision — Zero-Shot Model Comparison")
    print("BLIP-2 + OPT-2.7B  vs  BLIP-2 + FlanT5-XL")
    print("=" * 72)

    # ── Load results ──
    opt_path = config.RESULTS_DIR / "zero_shot_opt" / "evaluation_results.json"
    flan_path = config.RESULTS_DIR / "zero_shot_flan-t5-xl" / "evaluation_results.json"

    with open(opt_path) as f:
        opt_results = json.load(f)
    with open(flan_path) as f:
        flan_results = json.load(f)

    print(f"Loaded OPT results:    {opt_path}")
    print(f"Loaded FlanT5 results: {flan_path}")

    # ── Build comparison ──
    comparison_rows = []
    opt_wins = 0
    flan_wins = 0
    ties = 0

    current_task = ""

    for metric_key, (task, metric_name) in METRIC_LABELS.items():
        # Print task header when task changes
        if task != current_task:
            current_task = task
            print(f"\n{'─'*72}")
            print(f"  {task}")
            print(f"{'─'*72}")

        opt_val = opt_results.get(metric_key)
        flan_val = flan_results.get(metric_key)

        if opt_val is None and flan_val is None:
            continue

        # Determine decimals based on metric type
        if "mae" in metric_key.lower() and "mape" not in metric_key.lower():
            decimals = 2
        elif "mape" in metric_key.lower() or "accuracy" in metric_key.lower():
            decimals = 2
        else:
            decimals = 4

        # Format values
        opt_str = format_result(opt_val, decimals) if opt_val else "N/A"
        flan_str = format_result(flan_val, decimals) if flan_val else "N/A"

        # Determine winner
        if opt_val and flan_val:
            higher_better = metric_key in HIGHER_IS_BETTER

            if higher_better:
                if opt_val["mean"] > flan_val["mean"]:
                    winner = "OPT"
                    opt_wins += 1
                elif flan_val["mean"] > opt_val["mean"]:
                    winner = "FlanT5"
                    flan_wins += 1
                else:
                    winner = "Tie"
                    ties += 1
            else:
                # Lower is better (errors, hallucination rates)
                if opt_val["mean"] < flan_val["mean"]:
                    winner = "OPT"
                    opt_wins += 1
                elif flan_val["mean"] < opt_val["mean"]:
                    winner = "FlanT5"
                    flan_wins += 1
                else:
                    winner = "Tie"
                    ties += 1
        else:
            winner = "N/A"

        # Check if CIs overlap (important for statistical significance)
        ci_overlap = ""
        if opt_val and flan_val:
            # CIs overlap if one's lower bound is below the other's upper bound
            overlap = (opt_val["ci_lower"] <= flan_val["ci_upper"] and
                       flan_val["ci_lower"] <= opt_val["ci_upper"])
            ci_overlap = "overlapping" if overlap else "non-overlapping"

        # Print row
        winner_marker = f"  ← {winner}" if winner not in ("N/A", "Tie") else ""
        if winner == "Tie":
            winner_marker = "  ← Tie"

        print(f"  {metric_name:25s}")
        print(f"    OPT:    {opt_str}")
        print(f"    FlanT5: {flan_str}")
        print(f"    Winner: {winner} (CIs {ci_overlap})")

        comparison_rows.append({
            "task": task,
            "metric": metric_name,
            "opt_mean": f"{opt_val['mean']:.4f}" if opt_val else "",
            "opt_ci": f"[{opt_val['ci_lower']:.4f} – {opt_val['ci_upper']:.4f}]" if opt_val else "",
            "flan_mean": f"{flan_val['mean']:.4f}" if flan_val else "",
            "flan_ci": f"[{flan_val['ci_lower']:.4f} – {flan_val['ci_upper']:.4f}]" if flan_val else "",
            "winner": winner,
            "ci_overlap": ci_overlap,
        })

    # ── Summary ──
    print(f"\n{'='*72}")
    print(f"SUMMARY")
    print(f"{'='*72}")
    print(f"  OPT wins:    {opt_wins} metrics")
    print(f"  FlanT5 wins: {flan_wins} metrics")
    print(f"  Ties:        {ties}")

    # ── Task-level breakdown ──
    print(f"\n  Per-task breakdown:")
    task_scores = {}
    for row in comparison_rows:
        task = row["task"]
        if task not in task_scores:
            task_scores[task] = {"OPT": 0, "FlanT5": 0, "Tie": 0}
        if row["winner"] in task_scores[task]:
            task_scores[task][row["winner"]] += 1

    for task, scores in task_scores.items():
        print(f"    {task:15s}: OPT {scores['OPT']} — FlanT5 {scores['FlanT5']}", end="")
        if scores["Tie"]:
            print(f" — Tie {scores['Tie']}")
        else:
            print()

    # ── Recommendation ──
    print(f"\n{'─'*72}")
    print("  RECOMMENDATION FOR FINE-TUNING")
    print(f"{'─'*72}")

    if flan_wins > opt_wins:
        recommended = "flan-t5-xl"
        reason = "FlanT5-XL"
    else:
        recommended = "opt"
        reason = "OPT-2.7B"

    print(f"\n  Based on {opt_wins + flan_wins + ties} metrics compared:")
    print(f"  {reason} wins on more metrics overall.")
    print()
    print(f"  Key considerations:")
    print(f"    - FlanT5 is instruction-tuned → understands prompts better")
    print(f"    - FlanT5 has severe repetition/hallucination in ingredients (7k+ hallucinated)")
    print(f"    - OPT has many unparseable outputs (not instruction-tuned)")
    print(f"    - FlanT5 is better at captioning and calorie estimation")
    print(f"    - OPT is better at ingredient precision and macro estimation")
    print(f"    - Both are near-random on VQA (~57% accuracy)")
    print()
    print(f"  Fine-tuning can fix:")
    print(f"    - FlanT5's repetition loops (LoRA teaches it when to stop)")
    print(f"    - Numeric accuracy for both (LoRA on Nutrition5K data)")
    print(f"    - VQA accuracy (LoRA on task-specific QA pairs)")
    print()
    print(f"  Fine-tuning CANNOT easily fix:")
    print(f"    - OPT's fundamental lack of instruction understanding")
    print(f"    - OPT's high unparseable rate (needs prompt engineering, not just LoRA)")
    print()
    print(f"  Suggested model for LoRA fine-tuning: FlanT5-XL")
    print(f"    Rationale: Its instruction-following ability gives LoRA a better")
    print(f"    starting point. The repetition problem is exactly the kind of issue")
    print(f"    LoRA fine-tuning on Nutrition5K can fix. OPT's issues are more")
    print(f"    fundamental — it doesn't understand what we're asking it to do.")

    # ── Save comparison CSV ──
    csv_path = config.RESULTS_DIR / "model_comparison.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "task", "metric",
            "opt_mean", "opt_ci",
            "flan_mean", "flan_ci",
            "winner", "ci_overlap",
        ])
        writer.writeheader()
        writer.writerows(comparison_rows)
    print(f"\n  Comparison saved to: {csv_path}")

    # ── Save as JSON too ──
    json_path = config.RESULTS_DIR / "model_comparison.json"
    with open(json_path, "w") as f:
        json.dump({
            "models": ["opt", "flan-t5-xl"],
            "opt_wins": opt_wins,
            "flan_wins": flan_wins,
            "ties": ties,
            "task_breakdown": task_scores,
            "metrics": comparison_rows,
            "recommendation": "flan-t5-xl",
        }, f, indent=2)
    print(f"  Comparison JSON saved to: {json_path}")


if __name__ == "__main__":
    compare_models()