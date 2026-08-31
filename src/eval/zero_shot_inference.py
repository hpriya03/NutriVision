"""
NutriVision — Zero-Shot Inference (Phase B3)

Runs a BLIP-2 model on the test set WITHOUT any fine-tuning.
This is the baseline: we're measuring what the pre-trained model can do
out of the box, so we can later show how much LoRA fine-tuning improves it.

Usage:
    python -m src.eval.zero_shot_inference --model opt --test-run
    python -m src.eval.zero_shot_inference --model flan-t5-xl --test-run
    python -m src.eval.zero_shot_inference --model opt          # full 507 dishes

What it does:
    1. Loads one BLIP-2 variant (OPT or FlanT5-XL)
    2. Loads the test split (507 dishes, or 5 with --test-run)
    3. For each dish, runs Tasks 1-4 (one prompt each) + Task 5 (4 VQA questions)
    4. Saves raw outputs to results/zero_shot_{model_key}/

What it does NOT do:
    - Score or evaluate anything (that's the evaluation script, B5)
    - Compare models (you run this twice, once per model)
"""

import argparse
import json
import os
import random
import time

import torch
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration

from src import config


def strip_prompt(output_text: str, prompt_text: str) -> str:
    """
    For causal LMs (OPT), model.generate() returns the full sequence
    including the input prompt tokens. When decoded, the output starts
    with the prompt text followed by the model's actual answer.

    This function removes the prompt prefix so we keep only the new content.

    For encoder-decoder models (FlanT5), this is never called because
    the decoder output is already separate from the encoder input.
    """
    if not prompt_text:
        return output_text

    # Try exact prefix match
    if output_text.startswith(prompt_text):
        return output_text[len(prompt_text):].strip()

    # Try case-insensitive (some tokenizers normalise case)
    if output_text.lower().startswith(prompt_text.lower()):
        return output_text[len(prompt_text):].strip()

    # Fallback: return as-is (prompt wasn't in the output)
    return output_text.strip()


def run_inference(model_key: str, test_run: bool = False):
    """
    Main inference loop.

    Args:
        model_key: "opt" or "flan-t5-xl" — selects the BLIP-2 variant.
        test_run:  If True, only process 5 dishes (quick sanity check).
    """
    # ── Validate model choice ──
    if model_key not in config.MODELS:
        valid = ", ".join(config.MODELS.keys())
        raise ValueError(f"Unknown model '{model_key}'. Choose from: {valid}")

    model_info = config.MODELS[model_key]
    model_name = model_info["name"]
    is_causal = model_info["is_causal"]

    print("=" * 55)
    print("NutriVision — Zero-Shot Inference")
    print("=" * 55)
    if test_run:
        print("*** TEST RUN: using only 5 dishes ***")

    # ── Load model ──
    print(f"Loading {model_name}...")
    print(f"  ({model_info['description']})")

    processor = Blip2Processor.from_pretrained(model_name)
    model = Blip2ForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()
    device = model.device
    print(f"  Model loaded on {device}")

    # ── Load test dish IDs ──
    splits_file = config.SPLITS_DIR / "test_ids.json"
    with open(splits_file) as f:
        test_ids = json.load(f)

    if test_run:
        test_ids = test_ids[:5]

    # ── Get model-specific prompts ──
    task_prompts = config.TASK_PROMPTS[model_key]
    vqa_wrapper = config.VQA_PROMPT_WRAPPER[model_key]

    # ── Select 4 VQA questions per dish (seeded for reproducibility) ──
    rng = random.Random(42)
    num_vqa = 4

    # ── Inference loop ──
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
            # For OPT captioning: empty prompt means pure image captioning
            if prompt == "":
                inputs = processor(images=image, return_tensors="pt")
            else:
                inputs = processor(images=image, text=prompt, return_tensors="pt")

            inputs = {k: v.to(device) for k, v in inputs.items()}

            # don't compute gradients, not training, just predicting
            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    **config.GENERATION_CONFIG,
                )

            output_text = processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0].strip()

            # For causal LMs (OPT): strip the prompt from the output
            if is_causal and prompt:
                output_text = strip_prompt(output_text, prompt)

            dish_results["tasks"][task_key] = {
                "prompt": prompt if prompt else "(no text prompt — pure captioning)",
                "output": output_text,
            }

        # ── Task 5: VQA ──
        # Sample 4 questions for this dish (same seed = same questions every run)
        questions = rng.sample(config.VQA_QUESTION_TEMPLATES, num_vqa)

        for question in questions:
            # Wrap the question in the model-appropriate format
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

            # For causal LMs (OPT): strip the prompt from the output
            if is_causal:
                output_text = strip_prompt(output_text, prompt)

            dish_results["vqa"].append({
                "question": question,
                "prompt_sent": prompt,
                "output": output_text,
            })

        all_results[dish_id] = dish_results

        # Progress
        elapsed = time.time() - dish_start
        remaining = (time.time() - total_start) / (i + 1) * (len(test_ids) - i - 1)
        print(f"  [{i+1}/{len(test_ids)}] {dish_id} — {elapsed:.1f}s this dish, ~{remaining/60:.0f}min remaining")

    total_time = time.time() - total_start
    print(f"Done! {len(all_results)} dishes processed in {total_time/60:.1f} minutes")

    # ── Save results ──
    output_dir = config.RESULTS_DIR / f"zero_shot_{model_key}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Raw JSON (structured data for the evaluation script to consume)
    raw_path = output_dir / "raw_outputs.json"
    with open(raw_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved to: {raw_path}")

    # Human-readable sample (first 3 dishes, for eyeballing)
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

            f.write(f"\n--- {config.TASK_DISPLAY_NAMES['vqa']} ---\n")
            for qa in dish["vqa"]:
                f.write(f"Q: {qa['question']}\n")
                f.write(f"A: {qa['output']}\n\n")

    print(f"Samples saved to: {sample_path}")

    # Also print the samples to terminal
    with open(sample_path) as f:
        print(f.read())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="NutriVision zero-shot inference"
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=list(config.MODELS.keys()),
        help="Which BLIP-2 variant to run (opt or flan-t5-xl)",
    )
    parser.add_argument(
        "--test-run",
        action="store_true",
        help="Quick sanity check with only 5 dishes",
    )
    args = parser.parse_args()

    run_inference(model_key=args.model, test_run=args.test_run)