"""
NutriVision — LoRA Fine-Tuning (Phase D, v2)

Fine-tunes BLIP-2 + FlanT5-XL on Nutrition5K training data using LoRA.
Only ~1% of parameters are trainable — the rest stay frozen.

v2 changes:
    - Loads v2_train_examples.json (sentence-level VQA answers)
    - Saves checkpoints to v2_lora_flan-t5-xl_{target}/
    - Saves training log to v2_training_log_{target}.json
    - Added --lr flag for learning rate override (use 1e-4 for LLM target
      to avoid float16 NaN instability in FlanT5 attention layers)

What it does:
    1. Loads BLIP-2 + FlanT5-XL (frozen weights, float16)
    2. Attaches LoRA adapters to Q-Former + LLM attention layers
    3. Loads the ~44,080 training examples from v2_train_examples.json
    4. Trains for N epochs with gradient accumulation
    5. Saves LoRA adapter weights (NOT the full model — just the small trained part)
    6. Logs training loss per step

LoRA targets:
    Q-Former: query, key, value, output projections (learns better image→text mapping)
    LLM (FlanT5): query, value projections (learns nutrition-specific language)

Hardware requirements:
    ~14-16 GB VRAM (A10G 24GB is fine)
    Training time: ~2-4 hours depending on epochs

Usage:
    # Q-Former ablation (default lr=2e-4):
    python -m src.train.lora_finetune --target qformer 2>&1 | tee logs/12_v2_lora_train_qformer.log

    # LLM ablation (use lr=1e-4 to avoid float16 NaN):
    python -m src.train.lora_finetune --target llm --lr 1e-4 2>&1 | tee logs/12_v2_lora_train_llm.log

    # Both ablation:
    python -m src.train.lora_finetune --target both 2>&1 | tee logs/12_v2_lora_train_both.log

    # Quick sanity check:
    python -m src.train.lora_finetune --target qformer --test-run
"""

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from peft import LoraConfig, get_peft_model, TaskType

from src import config


# ── Training hyperparameters ──
# These are standard values for LoRA fine-tuning of VLMs.
# Each choice is explained in the comment beside it.

TRAIN_CONFIG = {
    # ── LoRA architecture ──
    "lora_r": 16,             # Rank — controls adapter capacity
                               # 16 is the standard middle ground:
                               # 8 = less capacity, faster
                               # 32 = more capacity, slower, risk of overfitting
                               # 16 balances expressiveness vs efficiency

    "lora_alpha": 32,          # Scaling factor — controls learning rate of LoRA
                               # Common rule: alpha = 2 × r
                               # Higher alpha = bigger updates per step

    "lora_dropout": 0.05,      # Dropout on LoRA layers — mild regularisation
                               # 0.05 is conservative; prevents overfitting on
                               # repeated patterns in training data

    # ── Training schedule ──
    "num_epochs": 3,           # Number of full passes through the data
                               # 3 is standard for LoRA — enough to learn,
                               # not so much that it overfits

    "batch_size": 4,           # Per-GPU batch size (limited by VRAM)
                               # A10G 24GB: 4 is safe with FlanT5-XL in float16

    "gradient_accumulation": 8, # Effective batch = batch_size × grad_accum = 32
                                # Simulates a larger batch without needing more VRAM
                                # 32 is standard for VLM fine-tuning

    "learning_rate": 2e-4,     # Peak learning rate
                               # 2e-4 is the LoRA standard (from the original paper)
                               # Higher than full fine-tuning because only a tiny
                               # fraction of parameters are being updated
                               #
                               # NOTE: For --target llm, use --lr 1e-4 to avoid
                               # float16 NaN in FlanT5 attention layers (known issue:
                               # T5 was designed for bfloat16, not float16)

    "warmup_ratio": 0.03,      # Warmup for first 3% of steps
                               # Prevents large gradient updates at the start
                               # when the model hasn't adapted yet

    "weight_decay": 0.01,      # L2 regularisation — prevents large weights

    # ── Saving ──
    "save_every_n_steps": 500, # Checkpoint frequency
    "log_every_n_steps": 50,   # Print loss every N steps
}


class NutriVisionDataset(Dataset):
    """
    PyTorch Dataset for NutriVision training.

    Each item returns:
        - image: PIL Image (the food photo)
        - prompt: str (the question/instruction)
        - answer: str (the correct response)

    The DataLoader collates these, and we tokenize in the training loop
    using the BLIP-2 processor (which handles both image + text).
    """

    def __init__(self, examples: list, project_root: Path):
        self.examples = examples
        self.project_root = project_root

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        image_path = self.project_root / ex["image_path"]
        image = Image.open(image_path).convert("RGB")
        return {
            "image": image,
            "prompt": ex["prompt"],
            "answer": ex["answer"],
        }


def collate_fn(batch, processor, device):
    """
    Custom collate: tokenize a batch of (image, prompt, answer) triples.

    For BLIP-2 FlanT5 training:
        - Processor encodes images + prompts into input_ids, attention_mask, pixel_values
        - Answers are tokenized separately as labels
        - Labels use -100 for padding tokens (ignored in loss computation)
    """
    images = [item["image"] for item in batch]
    prompts = [item["prompt"] for item in batch]
    answers = [item["answer"] for item in batch]

    # Encode images + prompts
    inputs = processor(
        images=images,
        text=prompts,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt",
    )

    # Encode answers as labels
    labels = processor.tokenizer(
        answers,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt",
    )

    # Replace padding token IDs with -100 so they're ignored in loss
    label_ids = labels.input_ids.clone()
    label_ids[label_ids == processor.tokenizer.pad_token_id] = -100

    inputs["labels"] = label_ids

    # Move everything to device
    inputs = {k: v.to(device) if hasattr(v, 'to') else v
              for k, v in inputs.items()}

    return inputs


def get_lora_target_modules(model, target: str = "both"):
    """
    Identify which layers to attach LoRA adapters to.

    Args:
        model: The BLIP-2 model
        target: Which component(s) to target:
            "qformer" — Q-Former only (Q,K,V,O projections)
            "llm"     — FlanT5 LLM only (Q,V projections)
            "both"    — Q-Former + LLM (default)

    Ablation study rationale:
        - qformer only: tests if better image→text bridging improves results
        - llm only: tests if better language generation improves results
        - both: tests if they complement each other (expected best)

    Q-Former targets all 4 projections (Q,K,V,O) because:
        - Q-Former is small (~100M params), so cost is low
        - It's the critical bridge that was never trained on nutrition data
        - All projections need to change for new attention patterns
        - Reference: LAVIS codebase (Salesforce BLIP-2 fine-tuning)
        - Architecture: 12 self-attention layers (4 projections each = 48)
                       + 6 cross-attention layers (every other block, 4 each = 24)
                       = 72 total target modules

    FlanT5 targets only Q,V projections because:
        - FlanT5 is large (~3B params), so targeting all 4 would be expensive
        - Q+V gives best performance/cost tradeoff
        - Reference: Hu et al., 2021, Table 6 (LoRA paper, ICLR 2022)
        - Architecture: 24 encoder blocks (self-attn q,v = 48)
                       + 24 decoder blocks (self-attn q,v = 48; cross-attn q,v = 48)
                       = 144 total target modules
    """
    target_modules = []
    seen = set()

    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        if name in seen:
            continue

        # Q-Former attention: query, key, value, output projections
        # Module names are leaf nodes, e.g.:
        #   qformer.encoder.layer.0.attention.attention.query
        #   qformer.encoder.layer.0.crossattention.output.dense
        if target in ("qformer", "both"):
            if "qformer" in name:
                # Q, K, V projections (leaf modules ending with .query/.key/.value)
                if any(name.endswith(s) for s in (".query", ".key", ".value")):
                    target_modules.append(name)
                    seen.add(name)
                # Output projection — only attention output, NOT FFN output
                # "attention.output.dense" matches both self-attn and cross-attn
                # but excludes qformer.encoder.layer.X.output.dense (the FFN)
                elif "attention.output.dense" in name:
                    target_modules.append(name)
                    seen.add(name)

        # FlanT5 LLM attention: q and v projections
        # Module names are leaf nodes, e.g.:
        #   language_model.encoder.block.0.layer.0.SelfAttention.q
        #   language_model.decoder.block.0.layer.1.EncDecAttention.v
        if target in ("llm", "both"):
            if "language_model" in name:
                if any(name.endswith(s) for s in (".q", ".v")):
                    target_modules.append(name)
                    seen.add(name)

    return target_modules


def run_training(test_run: bool = False, target: str = "both",
                 lr_override: float = None):
    """
    Main training function.

    Args:
        test_run: If True, only 50 examples and 1 epoch (quick sanity check)
        target: Which component to attach LoRA to: "qformer", "llm", or "both"
        lr_override: Override learning rate (e.g. 1e-4 for LLM to avoid NaN)
    """

    print("=" * 60)
    print("NutriVision — LoRA Fine-Tuning (v2)")
    print("=" * 60)
    print(f"Target: {target}")
    if test_run:
        print("*** TEST RUN: 50 examples, 1 epoch ***")

    tc = TRAIN_CONFIG.copy()

    # Apply learning rate override if specified
    if lr_override is not None:
        print(f"  Learning rate override: {lr_override} (default was {tc['learning_rate']})")
        tc["learning_rate"] = lr_override

    # ── Load model ──
    model_name = config.MODELS["flan-t5-xl"]["name"]
    print(f"\nLoading {model_name}...")

    processor = Blip2Processor.from_pretrained(model_name)
    model = Blip2ForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    device = model.device
    print(f"  Model loaded on {device}")

    # ── Find LoRA target modules ──
    print(f"\nFinding LoRA target modules (target={target})...")
    target_modules = get_lora_target_modules(model, target=target)
    print(f"  Found {len(target_modules)} target modules")
    if target_modules:
        # Breakdown by component and projection type
        qformer_targets = [t for t in target_modules if "qformer" in t]
        llm_targets = [t for t in target_modules if "language_model" in t]
        print(f"  Q-Former layers: {len(qformer_targets)}")
        if qformer_targets:
            # Show breakdown: self-attn vs cross-attn, by projection
            self_attn = [t for t in qformer_targets if "crossattention" not in t]
            cross_attn = [t for t in qformer_targets if "crossattention" in t]
            print(f"    Self-attention:  {len(self_attn)}  (12 layers × Q,K,V,O)")
            print(f"    Cross-attention: {len(cross_attn)}  (6 layers × Q,K,V,O)")
            print(f"    Sample: {qformer_targets[0]}")
        print(f"  LLM layers:      {len(llm_targets)}")
        if llm_targets:
            enc_targets = [t for t in llm_targets if ".encoder." in t]
            dec_targets = [t for t in llm_targets if ".decoder." in t]
            print(f"    Encoder: {len(enc_targets)}  (24 blocks × q,v)")
            print(f"    Decoder: {len(dec_targets)}  (24 blocks × self q,v + cross q,v)")
            print(f"    Sample: {llm_targets[0]}")

    if not target_modules:
        print("  ERROR: No target modules found! Check model architecture.")
        print("  Dumping all Linear modules for debugging:")
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                print(f"    {name}")
        raise RuntimeError("No LoRA target modules found — cannot proceed.")

    # ── Attach LoRA ──
    print(f"\nAttaching LoRA adapters...")
    print(f"  r={tc['lora_r']}, alpha={tc['lora_alpha']}, dropout={tc['lora_dropout']}")

    lora_config = LoraConfig(
        r=tc["lora_r"],
        lora_alpha=tc["lora_alpha"],
        lora_dropout=tc["lora_dropout"],
        target_modules=target_modules,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM,  # FlanT5 is encoder-decoder
    )

    model = get_peft_model(model, lora_config)

    # Print parameter counts
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable parameters: {trainable:,} ({100 * trainable / total:.2f}%)")
    print(f"  Total parameters:     {total:,}")

    # ── Load training data (v2) ──
    print(f"\nLoading training data (v2 — sentence-level VQA)...")
    training_data_path = config.TRAINING_DIR / "v2_train_examples.json"
    with open(training_data_path) as f:
        train_data = json.load(f)

    print(f"  Version: {train_data.get('version', 'v1')}")
    print(f"  VQA answer style: {train_data.get('vqa_answer_style', 'single-word')}")

    examples = train_data["examples"]
    if test_run:
        examples = examples[:50]

    dataset = NutriVisionDataset(examples, config.PROJECT_ROOT)
    print(f"  {len(dataset)} training examples")

    # Custom collate: keep items as a list of dicts (don't try to
    # stack PIL Images into tensors — that's what caused the TypeError).
    # We tokenize each batch in the training loop using collate_fn().
    def list_collate(batch):
        return batch

    dataloader = DataLoader(
        dataset,
        batch_size=tc["batch_size"],
        shuffle=True,
        num_workers=2,
        pin_memory=False,  # Can't pin PIL Images
        collate_fn=list_collate,
    )

    # ── Optimizer + Scheduler ──
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=tc["learning_rate"],
        weight_decay=tc["weight_decay"],
    )

    num_epochs = 1 if test_run else tc["num_epochs"]
    total_steps = len(dataloader) * num_epochs
    warmup_steps = int(total_steps * tc["warmup_ratio"])

    # Linear warmup then linear decay
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        remaining = total_steps - step
        total_remaining = total_steps - warmup_steps
        return max(0.0, remaining / max(1, total_remaining))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    effective_batch = tc["batch_size"] * tc["gradient_accumulation"]
    print(f"\n  Epochs:             {num_epochs}")
    print(f"  Batch size:         {tc['batch_size']}")
    print(f"  Gradient accum:     {tc['gradient_accumulation']}")
    print(f"  Effective batch:    {effective_batch}")
    print(f"  Total steps:        {total_steps}")
    print(f"  Warmup steps:       {warmup_steps}")
    print(f"  Learning rate:      {tc['learning_rate']}")

    # ── Training loop ──
    print(f"\n{'='*60}")
    print("TRAINING STARTED")
    print(f"{'='*60}")

    model.train()
    global_step = 0
    total_loss = 0.0
    log_loss = 0.0
    log_steps = 0
    best_loss = float("inf")
    training_log = []

    # v2: checkpoint dir uses v2_ prefix
    checkpoint_dir = config.CHECKPOINTS_DIR / f"v2_lora_flan-t5-xl_{target}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    train_start = time.time()

    for epoch in range(num_epochs):
        epoch_start = time.time()
        epoch_loss = 0.0
        epoch_steps = 0

        print(f"\n--- Epoch {epoch + 1}/{num_epochs} ---")

        for batch_idx, batch in enumerate(dataloader):
            # Tokenize batch
            try:
                inputs = collate_fn(batch, processor, device)
            except Exception as e:
                print(f"  [WARN] Batch {batch_idx} failed: {e}")
                continue

            # Forward pass
            outputs = model(**inputs)
            loss = outputs.loss / tc["gradient_accumulation"]

            # Backward pass
            loss.backward()

            # Gradient accumulation — only step every N batches
            if (batch_idx + 1) % tc["gradient_accumulation"] == 0 or (batch_idx + 1) == len(dataloader):
                # Clip gradients to prevent exploding
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            step_loss = loss.item() * tc["gradient_accumulation"]
            epoch_loss += step_loss
            epoch_steps += 1
            total_loss += step_loss
            log_loss += step_loss
            log_steps += 1

            # Logging
            if log_steps >= tc["log_every_n_steps"]:
                avg_loss = log_loss / log_steps
                elapsed = time.time() - train_start
                lr = scheduler.get_last_lr()[0]

                log_entry = {
                    "step": global_step,
                    "epoch": epoch + 1,
                    "loss": avg_loss,
                    "lr": lr,
                    "elapsed_min": elapsed / 60,
                }
                training_log.append(log_entry)

                remaining_batches = (num_epochs - epoch) * len(dataloader) - batch_idx
                time_per_batch = elapsed / (epoch * len(dataloader) + batch_idx + 1)
                eta_min = remaining_batches * time_per_batch / 60

                print(f"  Step {global_step:5d} | "
                      f"Loss: {avg_loss:.4f} | "
                      f"LR: {lr:.2e} | "
                      f"Elapsed: {elapsed/60:.1f}min | "
                      f"ETA: {eta_min:.0f}min")

                log_loss = 0.0
                log_steps = 0

            # Checkpointing
            if global_step > 0 and global_step % tc["save_every_n_steps"] == 0:
                ckpt_path = checkpoint_dir / f"checkpoint-{global_step}"
                model.save_pretrained(ckpt_path)
                print(f"  Checkpoint saved: {ckpt_path}")

        # End of epoch
        avg_epoch_loss = epoch_loss / max(1, epoch_steps)
        epoch_time = time.time() - epoch_start
        print(f"\n  Epoch {epoch + 1} complete — "
              f"Avg loss: {avg_epoch_loss:.4f} — "
              f"Time: {epoch_time/60:.1f}min")

        # Save end-of-epoch checkpoint
        ckpt_path = checkpoint_dir / f"epoch-{epoch + 1}"
        model.save_pretrained(ckpt_path)
        print(f"  Epoch checkpoint saved: {ckpt_path}")

        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            best_path = checkpoint_dir / "best"
            model.save_pretrained(best_path)
            print(f"  New best model saved: {best_path} (loss: {best_loss:.4f})")

    # ── Training complete ──
    total_time = time.time() - train_start
    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"  Total time:      {total_time/60:.1f} minutes")
    print(f"  Best loss:       {best_loss:.4f}")
    print(f"  Final checkpoint: {checkpoint_dir}/best")
    print(f"  Adapter size:    {sum(f.stat().st_size for f in (checkpoint_dir / 'best').rglob('*') if f.is_file()) / 1024 / 1024:.1f} MB")

    # ── Save training log (v2) ──
    log_path = config.LOGS_DIR / f"v2_training_log_{target}.json"
    with open(log_path, "w") as f:
        json.dump({
            "version": "v2",
            "target": target,
            "config": tc,
            "total_time_min": total_time / 60,
            "best_loss": best_loss,
            "total_steps": global_step,
            "trainable_params": trainable,
            "total_params": total,
            "trainable_pct": 100 * trainable / total,
            "log": training_log,
        }, f, indent=2)
    print(f"  Training log saved: {log_path}")

    print(f"\n{'='*60}")
    print("NEXT STEPS:")
    print(f"  1. Visualize training:")
    print(f"     python -m src.train.plot_training --runs {target}")
    print(f"  2. Run inference with the fine-tuned model:")
    print(f"     python -m src.eval.finetuned_inference --target {target} --test-run")
    print(f"  3. Evaluate:")
    print(f"     python -m src.eval.evaluate --results-dir results/v2_finetuned_{target}")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NutriVision LoRA fine-tuning (v2)")
    parser.add_argument("--test-run", action="store_true",
                        help="Quick test with 50 examples, 1 epoch")
    parser.add_argument("--target", default="both",
                        choices=["qformer", "llm", "both"],
                        help="Which component to attach LoRA to (for ablation study)")
    parser.add_argument("--lr", type=float, default=None,
                        help="Override learning rate (e.g. 1e-4 for LLM target)")
    args = parser.parse_args()

    run_training(test_run=args.test_run, target=args.target,
                 lr_override=args.lr)