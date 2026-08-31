"""
NutriVision — Ablation Comparison (Phase D companion)

Compares evaluation results between two runs side-by-side.
Designed for comparing zero-shot baseline vs fine-tuned ablation runs,
or comparing different ablation targets against each other.

Usage:
    # Zero-shot vs Q-Former fine-tuned:
    python -m src.eval.compare_ablation \
        --baseline results/zero_shot_flan-t5-xl \
        --experiment results/finetuned_qformer \
        --labels "Zero-Shot FlanT5" "Q-Former LoRA"

    # Compare two ablation targets:
    python -m src.eval.compare_ablation \
        --baseline results/finetuned_qformer \
        --experiment results/finetuned_llm \
        --labels "Q-Former Only" "LLM Only"

    # Three-way comparison (all ablations):
    python -m src.eval.compare_ablation \
        --baseline results/zero_shot_flan-t5-xl \
        --experiment results/finetuned_qformer results/finetuned_llm results/finetuned_both \
        --labels "Zero-Shot" "Q-Former" "LLM" "Both"

Outputs:
    results/ablation_comparison.csv
    results/ablation_comparison.json
"""

import argparse
import csv
import json
from pathlib import Path

from src import config


# ── Metric display configuration ──
# (key_in_json, display_name, task_group, lower_is_better, format_decimals, unit)
METRICS = [
    # Captioning
    ("caption_bleu1",       "BLEU-1",           "Captioning",   False, 4, ""),
    ("caption_bleu2",       "BLEU-2",           "Captioning",   False, 4, ""),
    ("caption_bleu4",       "BLEU-4",           "Captioning",   False, 4, ""),
    ("caption_rouge1",      "ROUGE-1",          "Captioning",   False, 4, ""),
    ("caption_rougeL",      "ROUGE-L",          "Captioning",   False, 4, ""),
    ("caption_meteor",      "METEOR",           "Captioning",   False, 4, ""),
    # Ingredients
    ("ingr_precision",      "Precision",        "Ingredients",  False, 4, ""),
    ("ingr_recall",         "Recall",           "Ingredients",  False, 4, ""),
    ("ingr_f1",             "F1",               "Ingredients",  False, 4, ""),
    ("ingr_halluc_rate",    "Hallucination Rate","Ingredients",  True,  4, ""),
    # Calories
    ("cal_mae",             "MAE",              "Calories",     True,  2, " cal"),
    ("cal_mape",            "MAPE",             "Calories",     True,  2, "%"),
    ("cal_halluc_rate",     "Hallucination Rate","Calories",     True,  4, ""),
    # Macros
    ("macro_protein_mae",   "Protein MAE",      "Macros",       True,  2, "g"),
    ("macro_protein_mape",  "Protein MAPE",     "Macros",       True,  2, "%"),
    ("macro_fat_mae",       "Fat MAE",          "Macros",       True,  2, "g"),
    ("macro_fat_mape",      "Fat MAPE",         "Macros",       True,  2, "%"),
    ("macro_carb_mae",      "Carb MAE",         "Macros",       True,  2, "g"),
    ("macro_carb_mape",     "Carb MAPE",        "Macros",       True,  2, "%"),
    # VQA
    ("vqa_accuracy",        "Accuracy",         "VQA",          False, 2, "%"),
    ("vqa_numeric_mae",     "Numeric MAE",      "VQA",          True,  2, ""),
    ("vqa_halluc_rate",     "Hallucination Rate","VQA",          True,  4, ""),
]


def load_results(results_dir: Path) -> dict:
    """Load evaluation_results.json from a results directory."""
    path = results_dir / "evaluation_results.json"
    if not path.exists():
        raise FileNotFoundError(f"No evaluation_results.json in {results_dir}")
    with open(path) as f:
        return json.load(f)


def format_val(val: float, decimals: int) -> str:
    """Format a value with the given decimal places."""
    return f"{val:.{decimals}f}"


def format_ci(result: dict, decimals: int, unit: str = "") -> str:
    """Format a result dict as 'mean [lower – upper]unit'."""
    m = result["mean"]
    lo = result["ci_lower"]
    hi = result["ci_upper"]
    return f"{m:.{decimals}f} [{lo:.{decimals}f} – {hi:.{decimals}f}]{unit}"


def ci_overlap(a: dict, b: dict) -> bool:
    """Check if two confidence intervals overlap."""
    return a["ci_lower"] <= b["ci_upper"] and b["ci_lower"] <= a["ci_upper"]


def compute_delta(baseline_val: float, experiment_val: float, lower_is_better: bool) -> tuple:
    """Compute absolute and relative change, and whether it's an improvement."""
    abs_delta = experiment_val - baseline_val
    if baseline_val != 0:
        rel_delta = (abs_delta / abs(baseline_val)) * 100
    else:
        rel_delta = float("inf") if abs_delta != 0 else 0.0

    if lower_is_better:
        improved = abs_delta < 0
    else:
        improved = abs_delta > 0

    return abs_delta, rel_delta, improved


def compare_ablation(baseline_dir: Path, experiment_dirs: list[Path],
                     labels: list[str]):
    """Run the comparison and print results."""

    print("=" * 72)
    print("NutriVision — Ablation Comparison")
    print("=" * 72)

    # Load all results
    all_results = []
    for i, d in enumerate([baseline_dir] + experiment_dirs):
        results = load_results(d)
        all_results.append(results)
        print(f"  Loaded: {labels[i]:20s} ← {d}")

    baseline = all_results[0]
    experiments = all_results[1:]

    # For CSV output
    csv_rows = []

    # Two-way comparison (baseline vs one experiment) — detailed format
    if len(experiments) == 1:
        exp = experiments[0]
        exp_label = labels[1]

        wins = {"improved": 0, "degraded": 0, "neutral": 0}
        task_wins = {}

        current_task = None
        for key, display, task, lower_better, dec, unit in METRICS:
            if key not in baseline or key not in exp:
                continue

            b = baseline[key]
            e = exp[key]

            if task != current_task:
                current_task = task
                task_wins[task] = {"improved": 0, "degraded": 0}
                print(f"\n{'─'*72}")
                print(f"  {task}")
                print(f"{'─'*72}")

            abs_delta, rel_delta, improved = compute_delta(
                b["mean"], e["mean"], lower_better
            )
            overlap = ci_overlap(b, e)
            sig = "overlapping" if overlap else "significant"

            # Direction arrow
            if improved:
                arrow = "▲" if not lower_better else "▼"
                tag = f"improved ({sig})"
                wins["improved"] += 1
                task_wins[task]["improved"] += 1
            elif abs_delta == 0:
                arrow = "="
                tag = "no change"
                wins["neutral"] += 1
            else:
                arrow = "▼" if not lower_better else "▲"
                tag = f"degraded ({sig})"
                wins["degraded"] += 1
                task_wins[task]["degraded"] += 1

            sign = "+" if abs_delta >= 0 else ""
            print(f"  {display:22s}")
            print(f"    {labels[0]:16s}: {format_ci(b, dec, unit)}")
            print(f"    {exp_label:16s}: {format_ci(e, dec, unit)}")
            print(f"    {arrow} Δ = {sign}{abs_delta:.{dec}f}{unit}  "
                  f"({sign}{rel_delta:.1f}%)  [{tag}]")

            csv_rows.append({
                "task": task, "metric": display,
                "baseline_mean": b["mean"], "baseline_ci_lower": b["ci_lower"],
                "baseline_ci_upper": b["ci_upper"],
                "experiment_mean": e["mean"], "experiment_ci_lower": e["ci_lower"],
                "experiment_ci_upper": e["ci_upper"],
                "abs_delta": abs_delta, "rel_delta_pct": rel_delta,
                "improved": improved, "significant": not overlap,
            })

        # Summary
        print(f"\n{'='*72}")
        print("  SUMMARY")
        print(f"{'='*72}")
        print(f"  {labels[0]} vs {exp_label}")
        print(f"  Improved:  {wins['improved']} metrics")
        print(f"  Degraded:  {wins['degraded']} metrics")
        print()
        print("  Per-task breakdown:")
        for task, counts in task_wins.items():
            imp = counts["improved"]
            deg = counts["degraded"]
            total = imp + deg
            print(f"    {task:15s}: {imp}/{total} improved")

        sig_improved = sum(1 for r in csv_rows if r["improved"] and r["significant"])
        sig_degraded = sum(1 for r in csv_rows if not r["improved"] and r["significant"])
        print(f"\n  Statistically significant (non-overlapping CIs):")
        print(f"    Improved:  {sig_improved}")
        print(f"    Degraded:  {sig_degraded}")

    # Multi-way comparison — table format
    else:
        current_task = None
        for key, display, task, lower_better, dec, unit in METRICS:
            if key not in baseline:
                continue
            missing_any = any(key not in e for e in experiments)
            if missing_any:
                continue

            if task != current_task:
                current_task = task
                print(f"\n{'─'*72}")
                print(f"  {task}")
                print(f"{'─'*72}")

            print(f"  {display:22s}")
            b = baseline[key]
            print(f"    {labels[0]:16s}: {format_ci(b, dec, unit)}")

            for j, exp in enumerate(experiments):
                e = exp[key]
                abs_delta, rel_delta, improved = compute_delta(
                    b["mean"], e["mean"], lower_better
                )
                sign = "+" if abs_delta >= 0 else ""
                arrow = "▲" if improved else "▼" if abs_delta != 0 else "="
                print(f"    {labels[j+1]:16s}: {format_ci(e, dec, unit)}  "
                      f"{arrow} {sign}{rel_delta:.1f}%")

            csv_rows.append({
                "task": task, "metric": display,
                "baseline_mean": b["mean"],
                **{f"{labels[j+1]}_mean": experiments[j][key]["mean"]
                   for j in range(len(experiments)) if key in experiments[j]},
            })

    # ── Save outputs ──
    output_dir = config.RESULTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # CSV
    csv_path = output_dir / "ablation_comparison.csv"
    if csv_rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"\n  CSV saved to: {csv_path}")

    # JSON
    json_path = output_dir / "ablation_comparison.json"
    comparison_data = {
        "labels": labels,
        "baseline_dir": str(baseline_dir),
        "experiment_dirs": [str(d) for d in experiment_dirs],
        "metrics": csv_rows,
    }
    with open(json_path, "w") as f:
        json.dump(comparison_data, f, indent=2)
    print(f"  JSON saved to: {json_path}")

    print(f"{'='*72}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="NutriVision ablation comparison"
    )
    parser.add_argument(
        "--baseline", required=True,
        help="Baseline results directory (e.g. results/zero_shot_flan-t5-xl)"
    )
    parser.add_argument(
        "--experiment", required=True, nargs="+",
        help="One or more experiment results directories"
    )
    parser.add_argument(
        "--labels", required=True, nargs="+",
        help="Labels for baseline + experiments (must be len(experiment) + 1)"
    )
    args = parser.parse_args()

    baseline_dir = Path(args.baseline)
    experiment_dirs = [Path(d) for d in args.experiment]

    if len(args.labels) != len(experiment_dirs) + 1:
        parser.error(
            f"Need {len(experiment_dirs) + 1} labels "
            f"(1 baseline + {len(experiment_dirs)} experiments), "
            f"got {len(args.labels)}"
        )

    compare_ablation(baseline_dir, experiment_dirs, args.labels)