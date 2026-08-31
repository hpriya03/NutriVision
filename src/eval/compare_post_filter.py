"""
NutriVision — Post-Filter Comparison (Phase E2)

Reads raw and post-filtered evaluation results for all model
configurations and generates comparison tables + an interactive
HTML dashboard.

What it does:
    1. Loads evaluation_results.json (raw) and
       post_filtered/evaluation_results.json for each config
    2. Loads post_filter_summary.csv for correction statistics
    3. Prints terminal comparison tables
    4. Generates an interactive HTML comparison dashboard

Usage:
    python -m src.eval.compare_post_filter 2>&1 | tee logs/15_post_filter_comparison.log

Output files:
    results/post_filter_comparison.csv       — side-by-side metrics
    results/post_filter_comparison.html      — interactive charts
"""

import json
import csv
import os
from pathlib import Path


# ═══════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════

# Model configurations to compare
MODEL_CONFIGS = [
    {
        "key": "zero_shot",
        "dir": "zero_shot_flan-t5-xl",
        "label": "Zero-Shot",
        "has_splits": False,  # no seen/unseen split
    },
    {
        "key": "qformer",
        "dir": "v2_finetuned_qformer",
        "label": "Q-Former LoRA",
        "has_splits": True,
    },
    {
        "key": "llm",
        "dir": "v2_finetuned_llm",
        "label": "LLM LoRA",
        "has_splits": True,
    },
    {
        "key": "both",
        "dir": "v2_finetuned_both",
        "label": "Both LoRA",
        "has_splits": True,
    },
]

# Metrics to compare (key in evaluation_results.json → display name)
# For zero-shot (no splits): vqa_accuracy
# For finetuned (with splits): vqa_seen_accuracy, vqa_unseen_accuracy,
#                               vqa_overall_accuracy
VQA_METRICS_SPLIT = [
    ("vqa_seen_accuracy", "VQA Seen Acc (%)"),
    ("vqa_unseen_accuracy", "VQA Unseen Acc (%)"),
    ("vqa_overall_accuracy", "VQA Overall Acc (%)"),
]

VQA_METRICS_NOSPLIT = [
    ("vqa_accuracy", "VQA Accuracy (%)"),
]

OTHER_METRICS = [
    ("caption_bleu1", "BLEU-1"),
    ("caption_rouge1", "ROUGE-1"),
    ("caption_meteor", "METEOR"),
    ("ingr_f1", "Ingredient F1"),
    ("cal_mae", "Calorie MAE"),
    ("cal_mape", "Calorie MAPE (%)"),
    ("macro_protein_mae", "Protein MAE (g)"),
    ("macro_fat_mae", "Fat MAE (g)"),
    ("macro_carb_mae", "Carb MAE (g)"),
]


# ═══════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════

def load_eval_results(results_dir: str) -> dict:
    """Load evaluation_results.json from a directory."""
    path = Path(results_dir) / "evaluation_results.json"
    if not path.exists():
        print(f"  WARNING: {path} not found")
        return {}
    with open(path) as f:
        return json.load(f)


def load_post_filter_summary(results_dir: str) -> dict:
    """Load post_filter_summary.csv into a dict."""
    path = Path(results_dir) / "post_filter_summary.csv"
    if not path.exists():
        print(f"  WARNING: {path} not found")
        return {}
    summary = {}
    with open(path) as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            if len(row) >= 2:
                summary[row[0]] = row[1]
    return summary


def get_metric_value(results: dict, metric_key: str) -> float | None:
    """Extract mean value for a metric from evaluation results."""
    if metric_key in results:
        entry = results[metric_key]
        if isinstance(entry, dict) and "mean" in entry:
            return entry["mean"]
        elif isinstance(entry, (int, float)):
            return float(entry)
    return None


# ═══════════════════════════════════════════════════════════
# COMPARISON LOGIC
# ═══════════════════════════════════════════════════════════

def build_comparison(base_dir: str) -> dict:
    """
    Build comparison data for all model configurations.

    Returns:
        {
            "configs": [
                {
                    "key": "zero_shot",
                    "label": "Zero-Shot",
                    "has_splits": False,
                    "raw": {metric_key: mean_value, ...},
                    "pf": {metric_key: mean_value, ...},
                    "pf_stats": {stat_key: value, ...},
                },
                ...
            ]
        }
    """
    base = Path(base_dir)
    configs = []

    for cfg in MODEL_CONFIGS:
        results_dir = base / cfg["dir"]
        pf_dir = results_dir / "post_filtered"

        print(f"\nLoading {cfg['label']}...")
        print(f"  Raw:  {results_dir}")
        print(f"  PF:   {pf_dir}")

        raw_results = load_eval_results(str(results_dir))
        pf_results = load_eval_results(str(pf_dir))
        pf_stats = load_post_filter_summary(str(results_dir))

        # Extract metric values
        raw_vals = {}
        pf_vals = {}

        # Determine which VQA metrics to use based on config type
        if cfg["has_splits"]:
            vqa_metrics = VQA_METRICS_SPLIT
        else:
            vqa_metrics = VQA_METRICS_NOSPLIT

        for metric_key, _ in vqa_metrics + OTHER_METRICS:
            raw_vals[metric_key] = get_metric_value(raw_results, metric_key)
            pf_vals[metric_key] = get_metric_value(pf_results, metric_key)

        # ── Fallback for zero-shot PF ──
        # The raw zero-shot evaluation saves "vqa_accuracy" but after
        # post-filtering the evaluator may produce "vqa_overall_accuracy"
        # (because post-filter tags seen/unseen). Check both keys.
        if not cfg["has_splits"]:
            if pf_vals.get("vqa_accuracy") is None:
                # Try the split-based key the evaluator may have used
                fallback = get_metric_value(pf_results, "vqa_overall_accuracy")
                if fallback is not None:
                    pf_vals["vqa_accuracy"] = fallback
                    print(f"  (used vqa_overall_accuracy as fallback "
                          f"for zero-shot PF: {fallback:.2f}%)")
            # Also grab seen/unseen if the PF evaluator produced them
            for extra_key in ["vqa_seen_accuracy", "vqa_unseen_accuracy"]:
                val = get_metric_value(pf_results, extra_key)
                if val is not None:
                    pf_vals[extra_key] = val

        configs.append({
            "key": cfg["key"],
            "label": cfg["label"],
            "has_splits": cfg["has_splits"],
            "raw": raw_vals,
            "pf": pf_vals,
            "pf_stats": pf_stats,
        })

    return {"configs": configs}


# ═══════════════════════════════════════════════════════════
# TERMINAL OUTPUT
# ═══════════════════════════════════════════════════════════

def print_comparison(data: dict):
    """Print comparison tables to terminal."""

    print("\n" + "=" * 80)
    print("POST-FILTER COMPARISON: ALL CONFIGURATIONS")
    print("=" * 80)

    for cfg in data["configs"]:
        print(f"\n{'─' * 60}")
        print(f"  {cfg['label']}")
        print(f"{'─' * 60}")

        # VQA metrics
        if cfg["has_splits"]:
            vqa_metrics = VQA_METRICS_SPLIT
        else:
            vqa_metrics = VQA_METRICS_NOSPLIT

        print(f"\n  {'Metric':<25s}  {'Raw':>10s}  {'Post-Filt':>10s}  {'Delta':>10s}")
        print(f"  {'─' * 60}")

        for metric_key, display_name in vqa_metrics:
            raw_val = cfg["raw"].get(metric_key)
            pf_val = cfg["pf"].get(metric_key)
            if raw_val is not None and pf_val is not None:
                delta = pf_val - raw_val
                sign = "+" if delta >= 0 else ""
                print(f"  {display_name:<25s}  {raw_val:>10.2f}  {pf_val:>10.2f}  "
                      f"{sign}{delta:>9.2f}")
            elif raw_val is not None:
                print(f"  {display_name:<25s}  {raw_val:>10.2f}  {'N/A':>10s}  {'':>10s}")

        # Other metrics
        print()
        for metric_key, display_name in OTHER_METRICS:
            raw_val = cfg["raw"].get(metric_key)
            pf_val = cfg["pf"].get(metric_key)
            if raw_val is not None and pf_val is not None:
                delta = pf_val - raw_val
                sign = "+" if delta >= 0 else ""
                print(f"  {display_name:<25s}  {raw_val:>10.4f}  {pf_val:>10.4f}  "
                      f"{sign}{delta:>9.4f}")

        # Post-filter stats
        stats = cfg["pf_stats"]
        if stats:
            print(f"\n  Post-Filter Statistics:")
            print(f"    Dishes processed:        "
                  f"{stats.get('dishes_processed', 'N/A')}")
            print(f"    Ingredients removed:     "
                  f"{stats.get('ingredients_removed', 'N/A')}")
            print(f"    VQA corrections (total): "
                  f"{stats.get('vqa_corrections_total', 'N/A')}")
            print(f"      Seen corrections:      "
                  f"{stats.get('vqa_corrections_seen', 'N/A')}")
            print(f"      Unseen corrections:    "
                  f"{stats.get('vqa_corrections_unseen', 'N/A')}")
            print(f"    Atwater inconsistent:    "
                  f"{stats.get('atwater_inconsistent', 'N/A')}")
            print(f"    Cross-task incon.:        "
                  f"{stats.get('cross_task_inconsistencies', 'N/A')}")

    # ── Key findings ──
    print(f"\n{'=' * 80}")
    print("KEY FINDINGS")
    print(f"{'=' * 80}")

    # Find best overall accuracy
    best_acc = 0
    best_label = ""
    for cfg in data["configs"]:
        if cfg["has_splits"]:
            acc = cfg["pf"].get("vqa_overall_accuracy", 0) or 0
        else:
            acc = cfg["pf"].get("vqa_accuracy", 0) or 0
        if acc > best_acc:
            best_acc = acc
            best_label = cfg["label"]

    print(f"\n  Best overall accuracy:  {best_acc:.2f}%  ({best_label} + Post-Filter)")

    # Largest unseen improvement
    best_delta = 0
    best_delta_label = ""
    for cfg in data["configs"]:
        if cfg["has_splits"]:
            raw = cfg["raw"].get("vqa_unseen_accuracy", 0) or 0
            pf = cfg["pf"].get("vqa_unseen_accuracy", 0) or 0
            delta = pf - raw
            if delta > best_delta:
                best_delta = delta
                best_delta_label = cfg["label"]

    if best_delta > 0:
        print(f"  Largest unseen gain:   +{best_delta:.2f} pp  ({best_delta_label})")

    # Fine-tuning effect on seen corrections
    print(f"\n  Seen corrections by config:")
    for cfg in data["configs"]:
        stats = cfg["pf_stats"]
        seen_corr = stats.get("vqa_corrections_seen", "N/A")
        print(f"    {cfg['label']:<20s}  {seen_corr}")


# ═══════════════════════════════════════════════════════════
# CSV OUTPUT
# ═══════════════════════════════════════════════════════════

def save_comparison_csv(data: dict, output_path: str):
    """Save comparison as CSV for dissertation tables."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "config", "metric", "raw", "post_filtered", "delta",
        ])

        for cfg in data["configs"]:
            if cfg["has_splits"]:
                vqa_metrics = VQA_METRICS_SPLIT
            else:
                vqa_metrics = VQA_METRICS_NOSPLIT

            for metric_key, display_name in vqa_metrics + OTHER_METRICS:
                raw_val = cfg["raw"].get(metric_key)
                pf_val = cfg["pf"].get(metric_key)
                delta = ""
                if raw_val is not None and pf_val is not None:
                    delta = f"{pf_val - raw_val:.4f}"
                writer.writerow([
                    cfg["label"],
                    display_name,
                    f"{raw_val:.4f}" if raw_val is not None else "",
                    f"{pf_val:.4f}" if pf_val is not None else "",
                    delta,
                ])

        # Stats rows
        writer.writerow([])
        writer.writerow(["config", "statistic", "value", "", ""])
        for cfg in data["configs"]:
            stats = cfg["pf_stats"]
            for stat_key in [
                "dishes_processed", "ingredients_removed",
                "vqa_corrections_total", "vqa_corrections_seen",
                "vqa_corrections_unseen", "atwater_inconsistent",
                "cross_task_inconsistencies",
            ]:
                writer.writerow([
                    cfg["label"], stat_key,
                    stats.get(stat_key, ""), "", "",
                ])

    print(f"\nComparison CSV saved to: {path}")


# ═══════════════════════════════════════════════════════════
# HTML DASHBOARD
# ═══════════════════════════════════════════════════════════

def generate_html_dashboard(data: dict, output_path: str):
    """Generate interactive HTML comparison dashboard."""

    # ── Prepare data for the template ──
    configs_json = []
    for cfg in data["configs"]:
        entry = {
            "key": cfg["key"],
            "label": cfg["label"],
            "has_splits": cfg["has_splits"],
            "raw": {k: v for k, v in cfg["raw"].items() if v is not None},
            "pf": {k: v for k, v in cfg["pf"].items() if v is not None},
            "stats": cfg["pf_stats"],
        }
        configs_json.append(entry)

    data_json = json.dumps(configs_json, indent=2)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>NutriVision — Post-Filter Comparison</title>
<style>
  :root {{
    --c1: #2a78d6;  --c2: #eb6834;  --c3: #1baf7a;  --c4: #eda100;
    --surface: #ffffff;  --surface-alt: #f7f8fa;
    --text-primary: #1a1a1a;  --text-secondary: #555;  --text-muted: #888;
    --grid: #e8e8e8;  --gap-color: #ffffff;
  }}
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
         Helvetica, Arial, sans-serif;
         background: var(--surface-alt); color: var(--text-primary);
         padding: 24px; line-height: 1.5; }}
  h1 {{ font-size: 22px; font-weight: 600; margin-bottom: 4px; }}
  .subtitle {{ color: var(--text-secondary); font-size: 14px; margin-bottom: 24px; }}

  /* ── stat tiles ── */
  .stat-row {{ display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 28px; }}
  .stat-tile {{
    background: var(--surface); border-radius: 8px; padding: 16px 20px;
    flex: 1; min-width: 180px; box-shadow: 0 1px 3px rgba(0,0,0,0.06);
  }}
  .stat-label {{ font-size: 12px; color: var(--text-muted); text-transform: uppercase;
                 letter-spacing: 0.5px; margin-bottom: 4px; }}
  .stat-value {{ font-size: 28px; font-weight: 600; }}
  .stat-detail {{ font-size: 12px; color: var(--text-secondary); margin-top: 2px; }}

  /* ── chart cards ── */
  .chart-card {{
    background: var(--surface); border-radius: 8px; padding: 24px;
    margin-bottom: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.06);
  }}
  .chart-title {{ font-size: 15px; font-weight: 600; margin-bottom: 4px; }}
  .chart-sub {{ font-size: 12px; color: var(--text-muted); margin-bottom: 16px; }}
  .chart-area {{ position: relative; }}

  /* ── toggle ── */
  .toggle-row {{ display: flex; justify-content: flex-end; margin-bottom: 8px; }}
  .toggle-btn {{
    font-size: 11px; padding: 4px 10px; border: 1px solid var(--grid);
    background: var(--surface); border-radius: 4px; cursor: pointer;
    color: var(--text-secondary);
  }}
  .toggle-btn:hover {{ background: var(--surface-alt); }}

  /* ── bar groups ── */
  .bar-group {{ margin-bottom: 14px; }}
  .bar-group-label {{ font-size: 13px; color: var(--text-primary);
                      margin-bottom: 4px; font-weight: 500; }}
  .bar-pair {{ display: flex; align-items: center; margin-bottom: 3px; height: 26px; }}
  .bar-tag {{ font-size: 10px; color: var(--text-muted); width: 24px;
              text-align: right; margin-right: 8px; flex-shrink: 0; }}
  .bar-track {{ flex: 1; height: 20px; position: relative; background: var(--surface-alt);
                border-radius: 4px; overflow: visible; }}
  .bar-fill {{ height: 100%; border-radius: 4px 0 0 4px; position: relative; }}
  .bar-fill-end {{ border-radius: 0 4px 4px 0; }}
  .bar-val {{ position: absolute; right: -56px; top: 0; font-size: 12px;
              color: var(--text-primary); line-height: 20px; width: 52px;
              text-align: left; font-variant-numeric: tabular-nums; }}
  .bar-delta {{ font-size: 11px; margin-left: 64px; flex-shrink: 0;
                font-variant-numeric: tabular-nums; }}
  .delta-pos {{ color: #1a8a5a; }}
  .delta-neg {{ color: #c44; }}

  /* ── legend ── */
  .legend {{ display: flex; gap: 16px; margin-bottom: 12px; flex-wrap: wrap; }}
  .legend-item {{ display: flex; align-items: center; gap: 6px; font-size: 12px;
                  color: var(--text-secondary); }}
  .legend-swatch {{ width: 14px; height: 3px; border-radius: 1px; }}

  /* ── table view ── */
  .table-view {{ display: none; }}
  .table-view.active {{ display: block; }}
  .chart-view.hidden {{ display: none; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  th {{ text-align: left; padding: 8px 10px; border-bottom: 2px solid var(--grid);
       color: var(--text-secondary); font-weight: 500; }}
  td {{ padding: 6px 10px; border-bottom: 1px solid var(--grid); }}
  tr:hover td {{ background: var(--surface-alt); }}

  /* ── tooltip ── */
  .tooltip {{
    display: none; position: absolute; background: var(--text-primary);
    color: #fff; padding: 6px 10px; border-radius: 4px; font-size: 12px;
    white-space: nowrap; pointer-events: none; z-index: 10;
    box-shadow: 0 2px 8px rgba(0,0,0,0.2);
  }}

  /* ── responsive ── */
  @media (max-width: 600px) {{
    body {{ padding: 12px; }}
    .stat-tile {{ min-width: 140px; }}
    .bar-val {{ right: -48px; width: 44px; font-size: 11px; }}
    .bar-delta {{ margin-left: 52px; }}
  }}
</style>
</head>
<body>

<h1>Post-Filter Comparison</h1>
<p class="subtitle">Raw vs post-filtered evaluation across all model configurations</p>

<div id="dashboard"></div>

<script>
// ═══════════════════════════════════════════════════
// DATA (injected by compare_post_filter.py)
// ═══════════════════════════════════════════════════
const DATA = {data_json};

const COLORS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100'];
const RAW_OPACITY = '0.35';

// ═══════════════════════════════════════════════════
// HELPERS
// ═══════════════════════════════════════════════════

function fmt(v, dec) {{
  if (v == null) return 'N/A';
  return v.toFixed(dec !== undefined ? dec : 2);
}}

function delta(raw, pf, dec) {{
  if (raw == null || pf == null) return '';
  const d = pf - raw;
  const sign = d >= 0 ? '+' : '';
  return sign + d.toFixed(dec !== undefined ? dec : 2);
}}

function deltaClass(raw, pf, higherBetter) {{
  if (raw == null || pf == null) return '';
  const d = pf - raw;
  if (higherBetter) return d >= 0 ? 'delta-pos' : 'delta-neg';
  return d <= 0 ? 'delta-pos' : 'delta-neg';
}}

function esc(s) {{
  const d = document.createElement('div');
  d.textContent = s;
  return d.innerHTML;
}}

// ═══════════════════════════════════════════════════
// STAT TILES
// ═══════════════════════════════════════════════════

function renderStatTiles() {{
  // Best overall accuracy
  let bestAcc = 0, bestLabel = '';
  DATA.forEach(c => {{
    const acc = c.has_splits
      ? (c.pf.vqa_overall_accuracy || 0)
      : (c.pf.vqa_accuracy || 0);
    if (acc > bestAcc) {{ bestAcc = acc; bestLabel = c.label; }}
  }});

  // Largest unseen improvement
  let bestDelta = 0, bestDeltaLabel = '';
  DATA.forEach(c => {{
    if (c.has_splits) {{
      const d = (c.pf.vqa_unseen_accuracy || 0) - (c.raw.vqa_unseen_accuracy || 0);
      if (d > bestDelta) {{ bestDelta = d; bestDeltaLabel = c.label; }}
    }}
  }});

  // Total corrections across all configs
  let totalCorr = 0;
  DATA.forEach(c => {{
    totalCorr += parseInt(c.stats.vqa_corrections_total || '0');
  }});

  // Fewest seen corrections (among finetuned)
  let fewestSeen = Infinity, fewestLabel = '';
  DATA.forEach(c => {{
    if (c.has_splits) {{
      const s = parseInt(c.stats.vqa_corrections_seen || '999999');
      if (s < fewestSeen) {{ fewestSeen = s; fewestLabel = c.label; }}
    }}
  }});

  return `
    <div class="stat-row">
      <div class="stat-tile">
        <div class="stat-label">Best Overall Accuracy</div>
        <div class="stat-value">${{fmt(bestAcc, 1)}}%</div>
        <div class="stat-detail">${{esc(bestLabel)}} + Post-Filter</div>
      </div>
      <div class="stat-tile">
        <div class="stat-label">Largest Unseen Gain</div>
        <div class="stat-value">+${{fmt(bestDelta, 1)}} pp</div>
        <div class="stat-detail">${{esc(bestDeltaLabel)}}</div>
      </div>
      <div class="stat-tile">
        <div class="stat-label">Total VQA Corrections</div>
        <div class="stat-value">${{totalCorr.toLocaleString()}}</div>
        <div class="stat-detail">Across all configs</div>
      </div>
      <div class="stat-tile">
        <div class="stat-label">Fewest Seen Corrections</div>
        <div class="stat-value">${{fewestSeen === Infinity ? 'N/A' : fewestSeen}}</div>
        <div class="stat-detail">${{esc(fewestLabel)}} (best fine-tuned)</div>
      </div>
    </div>`;
}}

// ═══════════════════════════════════════════════════
// CHART 1: VQA Overall Accuracy (Raw vs PF)
// ═══════════════════════════════════════════════════

function renderAccuracyChart() {{
  // For each config: raw overall vs PF overall
  let maxVal = 0;
  const rows = DATA.map((c, i) => {{
    const rawKey = c.has_splits ? 'vqa_overall_accuracy' : 'vqa_accuracy';
    const raw = c.raw[rawKey] || 0;
    const pf = c.pf[rawKey] || 0;
    maxVal = Math.max(maxVal, raw, pf);
    return {{ label: c.label, raw, pf, color: COLORS[i] }};
  }});
  const scale = maxVal > 0 ? 100 / (maxVal * 1.15) : 1;

  let barsHtml = rows.map(r => `
    <div class="bar-group">
      <div class="bar-group-label">${{esc(r.label)}}</div>
      <div class="bar-pair">
        <span class="bar-tag">Raw</span>
        <div class="bar-track" style="margin-right:60px">
          <div class="bar-fill bar-fill-end" style="width:${{r.raw * scale}}%;
               background:${{r.color}}; opacity:${{RAW_OPACITY}}">
            <span class="bar-val">${{fmt(r.raw, 2)}}%</span>
          </div>
        </div>
      </div>
      <div class="bar-pair">
        <span class="bar-tag">PF</span>
        <div class="bar-track" style="margin-right:60px">
          <div class="bar-fill bar-fill-end" style="width:${{r.pf * scale}}%;
               background:${{r.color}}">
            <span class="bar-val">${{fmt(r.pf, 2)}}%</span>
          </div>
        </div>
        <span class="bar-delta ${{deltaClass(r.raw, r.pf, true)}}">${{delta(r.raw, r.pf)}}</span>
      </div>
    </div>
  `).join('');

  // Table view
  let tableHtml = `<table>
    <tr><th>Config</th><th>Raw (%)</th><th>Post-Filtered (%)</th><th>Delta (pp)</th></tr>
    ${{rows.map(r => `<tr>
      <td>${{esc(r.label)}}</td><td>${{fmt(r.raw, 2)}}</td>
      <td>${{fmt(r.pf, 2)}}</td>
      <td class="${{deltaClass(r.raw, r.pf, true)}}">${{delta(r.raw, r.pf)}}</td>
    </tr>`).join('')}}
  </table>`;

  return chartCard('VQA Overall Accuracy', 'Raw vs post-filtered, all configurations',
                   barsHtml, tableHtml, 'acc');
}}

// ═══════════════════════════════════════════════════
// CHART 2: Seen vs Unseen Impact (finetuned only)
// ═══════════════════════════════════════════════════

function renderSeenUnseenChart() {{
  const finetuned = DATA.filter(c => c.has_splits);
  let maxVal = 0;
  const rows = finetuned.map((c, i) => {{
    const rawSeen = c.raw.vqa_seen_accuracy || 0;
    const pfSeen = c.pf.vqa_seen_accuracy || 0;
    const rawUnseen = c.raw.vqa_unseen_accuracy || 0;
    const pfUnseen = c.pf.vqa_unseen_accuracy || 0;
    maxVal = Math.max(maxVal, rawSeen, pfSeen, rawUnseen, pfUnseen);
    return {{
      label: c.label, rawSeen, pfSeen, rawUnseen, pfUnseen,
      color: COLORS[DATA.indexOf(c)],
    }};
  }});
  const scale = maxVal > 0 ? 100 / (maxVal * 1.15) : 1;

  let barsHtml = rows.map(r => `
    <div class="bar-group">
      <div class="bar-group-label">${{esc(r.label)}} — Seen</div>
      <div class="bar-pair">
        <span class="bar-tag">Raw</span>
        <div class="bar-track" style="margin-right:60px">
          <div class="bar-fill bar-fill-end" style="width:${{r.rawSeen * scale}}%;
               background:${{r.color}}; opacity:${{RAW_OPACITY}}">
            <span class="bar-val">${{fmt(r.rawSeen, 2)}}%</span>
          </div>
        </div>
      </div>
      <div class="bar-pair">
        <span class="bar-tag">PF</span>
        <div class="bar-track" style="margin-right:60px">
          <div class="bar-fill bar-fill-end" style="width:${{r.pfSeen * scale}}%;
               background:${{r.color}}">
            <span class="bar-val">${{fmt(r.pfSeen, 2)}}%</span>
          </div>
        </div>
        <span class="bar-delta ${{deltaClass(r.rawSeen, r.pfSeen, true)}}">${{delta(r.rawSeen, r.pfSeen)}}</span>
      </div>
      <div class="bar-group-label" style="margin-top:8px">${{esc(r.label)}} — Unseen</div>
      <div class="bar-pair">
        <span class="bar-tag">Raw</span>
        <div class="bar-track" style="margin-right:60px">
          <div class="bar-fill bar-fill-end" style="width:${{r.rawUnseen * scale}}%;
               background:${{r.color}}; opacity:${{RAW_OPACITY}}">
            <span class="bar-val">${{fmt(r.rawUnseen, 2)}}%</span>
          </div>
        </div>
      </div>
      <div class="bar-pair">
        <span class="bar-tag">PF</span>
        <div class="bar-track" style="margin-right:60px">
          <div class="bar-fill bar-fill-end" style="width:${{r.pfUnseen * scale}}%;
               background:${{r.color}}">
            <span class="bar-val">${{fmt(r.pfUnseen, 2)}}%</span>
          </div>
        </div>
        <span class="bar-delta ${{deltaClass(r.rawUnseen, r.pfUnseen, true)}}">${{delta(r.rawUnseen, r.pfUnseen)}}</span>
      </div>
    </div>
  `).join('');

  let tableHtml = `<table>
    <tr><th>Config</th><th>Split</th><th>Raw (%)</th><th>PF (%)</th><th>Delta</th></tr>
    ${{rows.map(r => `
      <tr><td>${{esc(r.label)}}</td><td>Seen</td><td>${{fmt(r.rawSeen,2)}}</td>
          <td>${{fmt(r.pfSeen,2)}}</td>
          <td class="${{deltaClass(r.rawSeen, r.pfSeen, true)}}">${{delta(r.rawSeen, r.pfSeen)}}</td></tr>
      <tr><td>${{esc(r.label)}}</td><td>Unseen</td><td>${{fmt(r.rawUnseen,2)}}</td>
          <td>${{fmt(r.pfUnseen,2)}}</td>
          <td class="${{deltaClass(r.rawUnseen, r.pfUnseen, true)}}">${{delta(r.rawUnseen, r.pfUnseen)}}</td></tr>
    `).join('')}}
  </table>`;

  return chartCard('Seen vs Unseen Impact',
    'Post-filter improvement by question split (fine-tuned configs only)',
    barsHtml, tableHtml, 'split');
}}

// ═══════════════════════════════════════════════════
// CHART 3: Corrections Breakdown
// ═══════════════════════════════════════════════════

function renderCorrectionsChart() {{
  let maxVal = 0;
  const rows = DATA.map((c, i) => {{
    const total = parseInt(c.stats.vqa_corrections_total || '0');
    const seen = parseInt(c.stats.vqa_corrections_seen || '0');
    const unseen = parseInt(c.stats.vqa_corrections_unseen || '0');
    maxVal = Math.max(maxVal, total);
    return {{ label: c.label, total, seen, unseen, color: COLORS[i] }};
  }});
  const scale = maxVal > 0 ? 100 / (maxVal * 1.15) : 1;

  let barsHtml = rows.map(r => `
    <div class="bar-group">
      <div class="bar-group-label">${{esc(r.label)}}</div>
      <div class="bar-pair">
        <span class="bar-tag" style="width:48px">Total</span>
        <div class="bar-track" style="margin-right:60px">
          <div class="bar-fill bar-fill-end" style="width:${{r.total * scale}}%;
               background:${{r.color}}">
            <span class="bar-val">${{r.total.toLocaleString()}}</span>
          </div>
        </div>
      </div>
      <div class="bar-pair">
        <span class="bar-tag" style="width:48px">Seen</span>
        <div class="bar-track" style="margin-right:60px">
          <div class="bar-fill bar-fill-end" style="width:${{r.seen * scale}}%;
               background:${{r.color}}; opacity:0.6">
            <span class="bar-val">${{r.seen.toLocaleString()}}</span>
          </div>
        </div>
      </div>
      <div class="bar-pair">
        <span class="bar-tag" style="width:48px">Unseen</span>
        <div class="bar-track" style="margin-right:60px">
          <div class="bar-fill bar-fill-end" style="width:${{r.unseen * scale}}%;
               background:${{r.color}}; opacity:0.8">
            <span class="bar-val">${{r.unseen.toLocaleString()}}</span>
          </div>
        </div>
      </div>
    </div>
  `).join('');

  let tableHtml = `<table>
    <tr><th>Config</th><th>Total</th><th>Seen</th><th>Unseen</th></tr>
    ${{rows.map(r => `<tr><td>${{esc(r.label)}}</td><td>${{r.total}}</td>
        <td>${{r.seen}}</td><td>${{r.unseen}}</td></tr>`).join('')}}
  </table>`;

  return chartCard('VQA Corrections Breakdown',
    'Number of yes/no answers corrected by post-filter',
    barsHtml, tableHtml, 'corr');
}}

// ═══════════════════════════════════════════════════
// CHART 4: Post-Filter Statistics
// ═══════════════════════════════════════════════════

function renderStatsChart() {{
  const statKeys = [
    {{ key: 'ingredients_removed', label: 'Ingredients Removed' }},
    {{ key: 'atwater_inconsistent', label: 'Atwater Inconsistent' }},
    {{ key: 'cross_task_inconsistencies', label: 'Cross-Task Inconsistencies' }},
  ];

  let maxVal = 0;
  statKeys.forEach(sk => {{
    DATA.forEach(c => {{
      maxVal = Math.max(maxVal, parseInt(c.stats[sk.key] || '0'));
    }});
  }});
  const scale = maxVal > 0 ? 100 / (maxVal * 1.15) : 1;

  let barsHtml = statKeys.map(sk => {{
    let groupHtml = `<div class="bar-group">
      <div class="bar-group-label">${{esc(sk.label)}}</div>`;
    DATA.forEach((c, i) => {{
      const val = parseInt(c.stats[sk.key] || '0');
      groupHtml += `
      <div class="bar-pair">
        <span class="bar-tag" style="width:72px; font-size:10px">${{esc(c.label)}}</span>
        <div class="bar-track" style="margin-right:60px">
          <div class="bar-fill bar-fill-end" style="width:${{val * scale}}%;
               background:${{COLORS[i]}}">
            <span class="bar-val">${{val.toLocaleString()}}</span>
          </div>
        </div>
      </div>`;
    }});
    groupHtml += '</div>';
    return groupHtml;
  }}).join('');

  let tableHtml = `<table>
    <tr><th>Statistic</th>${{DATA.map(c => '<th>' + esc(c.label) + '</th>').join('')}}</tr>
    ${{statKeys.map(sk => `<tr><td>${{esc(sk.label)}}</td>
      ${{DATA.map(c => '<td>' + (c.stats[sk.key] || '0') + '</td>').join('')}}
    </tr>`).join('')}}
  </table>`;

  return chartCard('Post-Filter Statistics',
    'Flags and corrections across all checks',
    barsHtml, tableHtml, 'stats');
}}

// ═══════════════════════════════════════════════════
// CARD WRAPPER + TABLE TOGGLE
// ═══════════════════════════════════════════════════

function chartCard(title, subtitle, chartHtml, tableHtml, id) {{
  return `
  <div class="chart-card">
    <div class="chart-title">${{esc(title)}}</div>
    <div class="chart-sub">${{esc(subtitle)}}</div>
    <div class="toggle-row">
      <button class="toggle-btn" onclick="toggleView('${{id}}')"
              id="toggle-${{id}}">Table view</button>
    </div>
    <div class="chart-area">
      <div class="chart-view" id="chart-${{id}}">${{chartHtml}}</div>
      <div class="table-view" id="table-${{id}}">${{tableHtml}}</div>
    </div>
  </div>`;
}}

function toggleView(id) {{
  const chart = document.getElementById('chart-' + id);
  const table = document.getElementById('table-' + id);
  const btn = document.getElementById('toggle-' + id);
  const showingTable = table.classList.contains('active');
  if (showingTable) {{
    table.classList.remove('active');
    chart.classList.remove('hidden');
    btn.textContent = 'Table view';
  }} else {{
    table.classList.add('active');
    chart.classList.add('hidden');
    btn.textContent = 'Chart view';
  }}
}}

// ═══════════════════════════════════════════════════
// RENDER
// ═══════════════════════════════════════════════════

document.getElementById('dashboard').innerHTML =
  renderStatTiles() +
  renderAccuracyChart() +
  renderSeenUnseenChart() +
  renderCorrectionsChart() +
  renderStatsChart();
</script>
</body>
</html>"""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(html)
    print(f"HTML dashboard saved to: {path}")


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="NutriVision post-filter comparison — "
                    "compare raw vs post-filtered results across all configs")
    parser.add_argument(
        "--results-dir", type=str, default="results",
        help="Base results directory containing model subdirectories "
             "(default: results)")
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Where to save comparison outputs "
             "(default: same as --results-dir)")
    parser.add_argument(
        "--no-html", action="store_true",
        help="Skip HTML dashboard generation")
    args = parser.parse_args()

    output_dir = args.output_dir or args.results_dir

    # Build comparison
    data = build_comparison(args.results_dir)

    # Terminal output
    print_comparison(data)

    # CSV
    save_comparison_csv(data, os.path.join(output_dir,
                                           "post_filter_comparison.csv"))

    # HTML dashboard
    if not args.no_html:
        generate_html_dashboard(data, os.path.join(output_dir,
                                                    "post_filter_comparison.html"))

    print("\nDone.")