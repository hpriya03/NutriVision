"""
NutriVision — Fine-Tuned Inference (Phase D companion, v2)

Runs inference using the LoRA-adapted BLIP-2 model on the test set.
Loads the base model + LoRA adapter weights, then runs the same prompts
as zero-shot inference so results are directly comparable.

v2 changes:
    - Asks ALL 20 VQA questions per dish (12 seen + 8 unseen)
      instead of randomly sampling 4 from 12.
    - Each VQA entry is tagged with "split": "seen" or "unseen"
      so the evaluation script can report separate metrics.
    - Loads checkpoints from v2_lora_flan-t5-xl_{target}/
    - Saves to results/v2_finetuned_{target}/

Usage:
    # Test run (5 dishes, quick sanity check):
    python -m src.eval.finetuned_inference --target qformer --test-run

    # Full test set (507 dishes):
    python -m src.eval.finetuned_inference --target qformer

    # Other ablation targets:
    python -m src.eval.finetuned_inference --target llm
    python -m src.eval.finetuned_inference --target both

What it does:
    1. Loads BLIP-2 + FlanT5-XL (the base model, same as zero-shot)
    2. Loads the LoRA adapter from checkpoints/v2_lora_flan-t5-xl_{target}/best
    3. Runs Tasks 1-4 + Task 5 (ALL 20 VQA questions) for each test dish
    4. Saves outputs to results/v2_finetuned_{target}/raw_outputs.json

    The output format is compatible with the evaluation script (evaluate.py),
    with the addition of "split" tags on VQA entries.
"""

import argparse
import json
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from peft import PeftModel

from src import config


def run_inference(target: str, test_run: bool = False, checkpoint: str = "best"):
    """
    Run inference with a LoRA-adapted model (v2).

    Args:
        target: Which ablation target was trained: "qformer", "llm", or "both"
        test_run: If True, only 5 dishes (quick sanity check)
        checkpoint: Which checkpoint to load — "best" (default), "epoch-1", etc.
    """
    print("=" * 60)
    print("NutriVision — Fine-Tuned Inference (v2)")
    print("=" * 60)
    print(f"  Ablation target: {target}")
    print(f"  Checkpoint:      {checkpoint}")
    if test_run:
        print("  *** TEST RUN: 5 dishes only ***")

    # ── Locate adapter checkpoint (v2 path) ──
    adapter_dir = config.CHECKPOINTS_DIR / f"v2_lora_flan-t5-xl_{target}" / checkpoint
    if not adapter_dir.exists():
        raise FileNotFoundError(
            f"Adapter not found at {adapter_dir}\n"
            f"Available checkpoints: "
            f"{[d.name for d in (config.CHECKPOINTS_DIR / f'v2_lora_flan-t5-xl_{target}').iterdir() if d.is_dir()]}"
        )
    print(f"  Adapter path:    {adapter_dir}")

    # ── Load base model ──
    model_name = config.MODELS["flan-t5-xl"]["name"]
    print(f"\nLoading base model: {model_name}...")

    processor = Blip2Processor.from_pretrained(model_name)
    model = Blip2ForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    print(f"  Base model loaded on {model.device}")

    # ── Load LoRA adapter ──
    print(f"Loading LoRA adapter ({target})...")
    model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()

    # Show adapter info
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  LoRA adapter loaded")
    print(f"  Trainable: {trainable:,} ({100 * trainable / total:.2f}%)")

    device = model.device

    # ── Load test dish IDs ──
    splits_file = config.SPLITS_DIR / "test_ids.json"
    with open(splits_file) as f:
        test_ids = json.load(f)

    if test_run:
        test_ids = test_ids[:5]

    print(f"\n  Test dishes: {len(test_ids)}")

    # ── Get FlanT5-specific prompts (same as zero-shot) ──
    task_prompts = config.TASK_PROMPTS["flan-t5-xl"]
    vqa_wrapper = config.VQA_PROMPT_WRAPPER["flan-t5-xl"]

    # ── ALL 20 VQA questions (12 seen + 8 unseen) ──
    seen_questions = config.VQA_QUESTION_TEMPLATES
    unseen_questions = config.VQA_HELD_OUT_TEMPLATES
    all_questions = seen_questions + unseen_questions
    seen_set = set(seen_questions)

    print(f"  VQA questions: {len(seen_questions)} seen + "
          f"{len(unseen_questions)} unseen = {len(all_questions)} total")

    # ── Inference loop ──
    print(f"\n{'='*60}")
    print("INFERENCE STARTED")
    print(f"{'='*60}")

    all_results = {}
    total_start = time.time()

    for i, dish_id in enumerate(test_ids):
        dish_start = time.time()

        # Load image
        image_path = config.IMAGES_DIR / dish_id / "rgb.png"
        if not image_path.exists():
            print(f"  [SKIP] {dish_id} — rgb.png not found")
            continue
        image = Image.open(image_path).convert("RGB")

        dish_results = {"dish_id": dish_id, "tasks": {}, "vqa": []}

        # ── Tasks 1-4 ──
        for task_key, prompt in task_prompts.items():
            inputs = processor(images=image, text=prompt, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    **config.GENERATION_CONFIG,
                )

            output_text = processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0].strip()

            # FlanT5 is encoder-decoder — output is already just the answer
            dish_results["tasks"][task_key] = {
                "prompt": prompt,
                "output": output_text,
            }

        # ── Task 5: VQA — ALL 20 questions ──
        for question in all_questions:
            prompt = vqa_wrapper.format(question=question)
            inputs = processor(images=image, text=prompt, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    **config.GENERATION_CONFIG,
                )

            output_text = processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0].strip()

            split = "seen" if question in seen_set else "unseen"

            dish_results["vqa"].append({
                "question": question,
                "prompt_sent": prompt,
                "output": output_text,
                "split": split,
            })

        all_results[dish_id] = dish_results

        # Progress
        elapsed = time.time() - dish_start
        total_elapsed = time.time() - total_start
        avg_per_dish = total_elapsed / (i + 1)
        remaining = avg_per_dish * (len(test_ids) - i - 1)
        print(f"  [{i+1}/{len(test_ids)}] {dish_id} — "
              f"{elapsed:.1f}s this dish, ~{remaining/60:.0f}min remaining")

    total_time = time.time() - total_start
    print(f"\nDone! {len(all_results)} dishes in {total_time/60:.1f} minutes")

    # ── Save results (v2 path) ──
    output_dir = config.RESULTS_DIR / f"v2_finetuned_{target}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Raw JSON (compatible with evaluation script)
    raw_path = output_dir / "raw_outputs.json"
    with open(raw_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved to: {raw_path}")

    # Human-readable sample (first 3 dishes)
    sample_path = output_dir / "sample_outputs.txt"
    with open(sample_path, "w") as f:
        sample_ids = list(all_results.keys())[:3]
        for dish_id in sample_ids:
            dish = all_results[dish_id]
            f.write(f"\n{'='*60}\n")
            f.write(f"DISH: {dish_id}\n")
            f.write(f"{'='*60}\n")

            for task_key in ["captioning", "ingredients", "calories", "macros"]:
                task = dish["tasks"][task_key]
                display = config.TASK_DISPLAY_NAMES.get(task_key, task_key)
                f.write(f"\n--- {display} ---\n")
                f.write(f"Prompt: {task['prompt']}\n")
                f.write(f"Output: {task['output']}\n")

            f.write(f"\n--- {config.TASK_DISPLAY_NAMES['vqa']} (SEEN questions) ---\n")
            for qa in dish["vqa"]:
                if qa["split"] == "seen":
                    f.write(f"Q: {qa['question']}\n")
                    f.write(f"A: {qa['output']}\n\n")

            f.write(f"\n--- {config.TASK_DISPLAY_NAMES['vqa']} (UNSEEN questions) ---\n")
            for qa in dish["vqa"]:
                if qa["split"] == "unseen":
                    f.write(f"Q: {qa['question']}\n")
                    f.write(f"A: {qa['output']}\n\n")

    print(f"Samples saved to: {sample_path}")

    # Print samples to terminal
    with open(sample_path) as f:
        print(f.read())

    print(f"\n{'='*60}")
    print("NEXT STEPS:")
    print(f"  1. Evaluate:")
    print(f"     python -m src.eval.evaluate "
          f"--results-dir results/v2_finetuned_{target} "
          f"2>&1 | tee logs/14_v2_eval_finetuned_{target}.log")
    print(f"  2. Compare with zero-shot:")
    print(f"     python -m src.eval.compare_models "
          f"--dirs results/zero_shot_flan-t5-xl results/v2_finetuned_{target}")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="NutriVision fine-tuned inference (v2)"
    )
    parser.add_argument(
        "--target",
        required=True,
        choices=["qformer", "llm", "both"],
        help="Which ablation target to load (qformer, llm, or both)",
    )
    parser.add_argument(
        "--checkpoint",
        default="best",
        help="Which checkpoint to load (default: best)",
    )
    parser.add_argument(
        "--test-run",
        action="store_true",
        help="Quick sanity check with 5 dishes only",
    )
    args = parser.parse_args()

    run_inference(target=args.target, test_run=args.test_run, checkpoint=args.checkpoint)