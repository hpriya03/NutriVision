"""
NutriVision — Training Visualization (Phase D companion)
Reads v2_training_log_{run}.json (saved by the LoRA training script) and produces
charts showing loss convergence, learning rate schedule, and epoch comparison.
Generates PNG images using matplotlib — reliable, no browser dependencies.
Usage:
    # Single run:
    python -m src.train.plot_training --runs qformer
    # Compare multiple ablation runs:
    python -m src.train.plot_training --runs qformer llm both
Outputs:
    results/training_plots/
        loss_curve_{run}.png        — training loss over steps
        lr_schedule_{run}.png       — learning rate schedule
        epoch_comparison_{run}.png  — average loss per epoch (bar chart)
        ablation_comparison.png     — overlay of all runs (if >1 run)
        training_summary_{run}.txt  — key stats printed to terminal
"""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — works on headless servers
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from src import config
# ── Style ──
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
BG_COLOR = "#fafaf8"
GRID_COLOR = "#e1e0d9"
TEXT_COLOR = "#0b0b0b"
MUTED_COLOR = "#898781"
RUN_LABELS = {
    "qformer": "Q-Former Only",
    "llm": "FlanT5 LLM Only",
    "both": "Q-Former + LLM",
}
def _style_ax(ax, xlabel="", ylabel=""):
    """Apply consistent styling to an axes object."""
    ax.set_facecolor(BG_COLOR)
    ax.grid(True, axis="y", color=GRID_COLOR, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GRID_COLOR)
    ax.spines["bottom"].set_color(GRID_COLOR)
    ax.tick_params(colors=MUTED_COLOR, labelsize=9)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=10, color=TEXT_COLOR)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=10, color=TEXT_COLOR)
def load_training_log(run_name: str) -> dict | None:
    """Load a v2 training log for a given run name."""
    path = config.LOGS_DIR / f"v2_training_log_{run_name}.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None
def plot_loss_curve(steps, losses, run_name, output_dir):
    """Plot training loss over steps."""
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor(BG_COLOR)
    _style_ax(ax, xlabel="Training Step", ylabel="Loss")
    ax.plot(steps, losses, color=COLORS[0], linewidth=2, zorder=3)
    ax.fill_between(steps, losses, alpha=0.08, color=COLORS[0], zorder=2)
    # Mark the final point
    ax.scatter([steps[-1]], [losses[-1]], color=COLORS[0], s=40,
               edgecolors="white", linewidth=1.5, zorder=4)
    ax.annotate(f"{losses[-1]:.4f}", (steps[-1], losses[-1]),
                textcoords="offset points", xytext=(10, -2),
                fontsize=9, color=TEXT_COLOR, weight="bold")
    # Mark the best (minimum) loss
    best_idx = losses.index(min(losses))
    if best_idx != len(losses) - 1:
        ax.scatter([steps[best_idx]], [losses[best_idx]], color="#1baf7a", s=40,
                   edgecolors="white", linewidth=1.5, zorder=4, marker="D")
        ax.annotate(f"Best: {losses[best_idx]:.4f}", (steps[best_idx], losses[best_idx]),
                    textcoords="offset points", xytext=(10, 8),
                    fontsize=8, color="#1baf7a")
    label = RUN_LABELS.get(run_name, run_name)
    ax.set_title(f"Training Loss — {label}", fontsize=13, fontweight="bold",
                 color=TEXT_COLOR, pad=12)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    plt.tight_layout()
    path = output_dir / f"loss_curve_{run_name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path
def plot_lr_schedule(steps, lrs, run_name, output_dir):
    """Plot learning rate schedule over steps."""
    fig, ax = plt.subplots(figsize=(10, 3.5))
    fig.patch.set_facecolor(BG_COLOR)
    _style_ax(ax, xlabel="Training Step", ylabel="Learning Rate")
    ax.plot(steps, lrs, color=COLORS[1], linewidth=2, zorder=3)
    ax.fill_between(steps, lrs, alpha=0.08, color=COLORS[1], zorder=2)
    ax.scatter([steps[-1]], [lrs[-1]], color=COLORS[1], s=40,
               edgecolors="white", linewidth=1.5, zorder=4)
    label = RUN_LABELS.get(run_name, run_name)
    ax.set_title(f"Learning Rate Schedule — {label}", fontsize=13,
                 fontweight="bold", color=TEXT_COLOR, pad=12)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1e"))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    plt.tight_layout()
    path = output_dir / f"lr_schedule_{run_name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path
def plot_epoch_comparison(epoch_avgs, run_name, output_dir):
    """Bar chart of average loss per epoch."""
    fig, ax = plt.subplots(figsize=(6, 4))
    fig.patch.set_facecolor(BG_COLOR)
    _style_ax(ax, xlabel="Epoch", ylabel="Average Loss")
    epochs = sorted(epoch_avgs.keys(), key=lambda x: int(x))
    values = [epoch_avgs[e] for e in epochs]
    x_labels = [f"Epoch {e}" for e in epochs]
    x_pos = range(len(epochs))
    bars = ax.bar(x_pos, values, color=COLORS[0], width=0.5, zorder=3,
                  edgecolor="white", linewidth=0.5)
    for bar in bars:
        bar.set_capstyle("round")
    # Value labels on top of bars
    for i, (x, v) in enumerate(zip(x_pos, values)):
        ax.text(x, v + max(values) * 0.02, f"{v:.4f}", ha="center", va="bottom",
                fontsize=9, fontweight="bold", color=TEXT_COLOR)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels)
    ax.set_ylim(0, max(values) * 1.18)
    label = RUN_LABELS.get(run_name, run_name)
    ax.set_title(f"Average Loss Per Epoch — {label}", fontsize=13,
                 fontweight="bold", color=TEXT_COLOR, pad=12)
    plt.tight_layout()
    path = output_dir / f"epoch_comparison_{run_name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path
def plot_ablation_comparison(all_logs, output_dir):
    """Overlay loss curves from multiple ablation runs on one chart."""
    fig, ax = plt.subplots(figsize=(10, 5.5))
    fig.patch.set_facecolor(BG_COLOR)
    _style_ax(ax, xlabel="Training Step", ylabel="Loss")
    for i, (name, log_data) in enumerate(all_logs.items()):
        entries = log_data["log"]
        steps = [e["step"] for e in entries]
        losses = [e["loss"] for e in entries]
        color = COLORS[i % len(COLORS)]
        label = RUN_LABELS.get(name, name)
        ax.plot(steps, losses, color=color, linewidth=2, label=label, zorder=3)
        ax.scatter([steps[-1]], [losses[-1]], color=color, s=40,
                   edgecolors="white", linewidth=1.5, zorder=4)
        ax.annotate(f"{losses[-1]:.4f}", (steps[-1], losses[-1]),
                    textcoords="offset points", xytext=(10, -2 + i * 14),
                    fontsize=9, color=color, weight="bold")
    ax.legend(loc="upper right", frameon=True, facecolor=BG_COLOR,
              edgecolor=GRID_COLOR, fontsize=10)
    ax.set_title("Ablation Comparison — Training Loss", fontsize=14,
                 fontweight="bold", color=TEXT_COLOR, pad=12)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    plt.tight_layout()
    path = output_dir / "ablation_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path
def print_summary(log_data, run_name):
    """Print a text summary of the training run."""
    entries = log_data["log"]
    lora_config = log_data.get("config", {})
    best_loss = log_data.get("best_loss", min(e["loss"] for e in entries))
    total_time = log_data.get("total_time_min", 0)
    total_steps = log_data.get("total_steps", max(e["step"] for e in entries))
    # Per-epoch averages
    epoch_losses = {}
    for e in entries:
        ep = e["epoch"]
        if ep not in epoch_losses:
            epoch_losses[ep] = []
        epoch_losses[ep].append(e["loss"])
    epoch_avgs = {ep: sum(ls) / len(ls) for ep, ls in epoch_losses.items()}
    label = RUN_LABELS.get(run_name, run_name)
    print(f"\n{'='*55}")
    print(f"  {label}")
    print(f"{'='*55}")
    print(f"  LoRA rank (r):      {lora_config.get('lora_r', 'N/A')}")
    print(f"  LoRA alpha:         {lora_config.get('lora_alpha', 'N/A')}")
    print(f"  Learning rate:      {lora_config.get('learning_rate', 'N/A')}")
    print(f"  Epochs:             {lora_config.get('num_epochs', 'N/A')}")
    print(f"  Total steps:        {total_steps:,}")
    print(f"  Training time:      {total_time:.1f} min")
    print(f"  Best loss:          {best_loss:.4f}")
    print(f"  Initial loss:       {entries[0]['loss']:.4f}")
    print(f"  Final loss:         {entries[-1]['loss']:.4f}")
    for ep in sorted(epoch_avgs.keys(), key=lambda x: int(x)):
        print(f"  Epoch {ep} avg loss:   {epoch_avgs[ep]:.4f}")
    print()
def plot_training(run_names: list[str]):
    """Main function: generate training visualization."""
    print("=" * 55)
    print("NutriVision — Training Visualization")
    print("=" * 55)
    # Output directory for all plots
    output_dir = config.RESULTS_DIR / "training_plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    all_logs = {}
    for name in run_names:
        log_data = load_training_log(name)
        if log_data:
            all_logs[name] = log_data
            n_entries = len(log_data.get("log", []))
            print(f"  Loaded: {name} ({n_entries} entries)")
        else:
            print(f"  [WARN] No v2 log found for '{name}' — skipping")
    if not all_logs:
        print("\n  ERROR: No v2 training logs found!")
        print("  Expected files like: logs/v2_training_log_qformer.json")
        print("  Run training first, then come back here.")
        return
    # ── Generate per-run charts ──
    for name, log_data in all_logs.items():
        entries = log_data["log"]
        if not entries:
            print(f"  [WARN] {name} has no log entries — skipping")
            continue
        steps = [e["step"] for e in entries]
        losses = [e["loss"] for e in entries]
        lrs = [e["lr"] for e in entries]
        # Per-epoch averages
        epoch_losses = {}
        for e in entries:
            ep = e["epoch"]
            if ep not in epoch_losses:
                epoch_losses[ep] = []
            epoch_losses[ep].append(e["loss"])
        epoch_avgs = {ep: sum(ls) / len(ls) for ep, ls in epoch_losses.items()}
        print(f"\n  Plotting: {RUN_LABELS.get(name, name)}")
        plot_loss_curve(steps, losses, name, output_dir)
        plot_lr_schedule(steps, lrs, name, output_dir)
        plot_epoch_comparison(epoch_avgs, name, output_dir)
        print_summary(log_data, name)
    # ── Ablation comparison (only if multiple runs) ──
    if len(all_logs) >= 2:
        print("  Plotting: Ablation Comparison")
        plot_ablation_comparison(all_logs, output_dir)
    print(f"\n{'='*55}")
    print(f"  All charts saved to: {output_dir}/")
    print(f"{'='*55}")
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="NutriVision training visualization"
    )
    parser.add_argument(
        "--runs", nargs="+", default=["both"],
        help="Which training runs to plot (qformer, llm, both)"
    )
    args = parser.parse_args()
    plot_training(run_names=args.runs)