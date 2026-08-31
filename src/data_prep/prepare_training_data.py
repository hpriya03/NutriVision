"""
NutriVision — Training Data Preparation (Phase C, v2)

Builds the training dataset for LoRA fine-tuning on BLIP-2 + FlanT5-XL.
Creates prompt-answer pairs for all 5 tasks using the 2,755 training dishes.

v2 changes:
    - VQA answers are now SENTENCE-LEVEL (not single-word) to exploit VLM
      capabilities and teach the model descriptive, evidence-based responses.
    - Only the 12 SEEN question templates are used in training.
      The 8 held-out questions are NEVER included — they test generalisation.
    - Output file: v2_train_examples.json (preserves old train_examples.json)

What it does:
    1. Loads metadata for all dishes
    2. Loads the 2,755 training dish IDs (from data/splits/train_ids.json)
    3. For each training dish, creates prompt → answer pairs:
       - Task 1 (captioning):   1 pair
       - Task 2 (ingredients):  1 pair
       - Task 3 (calories):     1 pair
       - Task 4 (macros):       1 pair
       - Task 5 (VQA):         12 pairs (all SEEN question templates)
         *** VQA answers are now SENTENCES, not single words ***
       Total: 16 pairs per dish × 2,755 dishes = ~44,080 examples
    4. Verifies all referenced images exist
    5. Saves to data/training/v2_train_examples.json

Usage:
    python -m src.data_prep.prepare_training_data 2>&1 | tee logs/11_training_data_v2.log
"""

import json
from pathlib import Path

from src import config
from src.data_prep.metadata_parser import load_metadata, get_ingredient_names
from src.eval.ground_truth import get_vqa_sentence_answer


def build_caption_answer(dish: dict) -> str:
    """
    Build a training caption from ingredients.

    Same logic as the pseudo-reference in evaluation — the model learns
    to produce captions like "a plate of chicken, rice, and broccoli".

    For plate-only dishes: "an empty plate".
    """
    ingredients = get_ingredient_names(dish)

    if not ingredients or ingredients == ["plate only"]:
        return "an empty plate"

    if len(ingredients) == 1:
        return f"a plate of {ingredients[0]}"
    else:
        listed = ", ".join(ingredients[:-1]) + f" and {ingredients[-1]}"
        return f"a plate of {listed}"


def build_ingredients_answer(dish: dict) -> str:
    """
    Build the expected ingredient list answer.

    Format: "chicken, rice, broccoli"
    For plate-only: "plate only"
    """
    ingredients = get_ingredient_names(dish)
    if not ingredients:
        return "plate only"
    return ", ".join(ingredients)


def build_calories_answer(dish: dict) -> str:
    """
    Build the expected calorie answer.

    Format: "350 calories"
    Rounded to nearest integer — sub-calorie precision isn't meaningful.
    """
    cal = round(dish["calories"])
    return f"{cal} calories"


def build_macros_answer(dish: dict) -> str:
    """
    Build the expected macronutrient answer.

    Format: "20g protein, 15g fat, 30g carbohydrates"
    Rounded to 1 decimal — matches nutrition label precision.
    """
    protein = round(dish["protein"], 1)
    fat = round(dish["fat"], 1)
    carb = round(dish["carb"], 1)
    return f"{protein}g protein, {fat}g fat, {carb}g carbohydrates"


def prepare_training_data():
    """Main function: build all training examples (v2 — sentence-level VQA)."""

    print("=" * 60)
    print("NutriVision — Training Data Preparation (v2)")
    print("=" * 60)

    # ── Load metadata ──
    metadata = load_metadata()
    print(f"Loaded metadata for {len(metadata)} dishes")

    # ── Load training IDs ──
    with open(config.SPLITS_DIR / "train_ids.json") as f:
        train_ids = json.load(f)
    print(f"Training set: {len(train_ids)} dishes")

    # ── Get FlanT5 prompts (we're fine-tuning FlanT5) ──
    task_prompts = config.TASK_PROMPTS["flan-t5-xl"]
    vqa_wrapper = config.VQA_PROMPT_WRAPPER["flan-t5-xl"]

    print(f"\nTask prompts (FlanT5-XL format):")
    for task, prompt in task_prompts.items():
        print(f"  {task}: \"{prompt}\"")
    print(f"VQA questions: {len(config.VQA_QUESTION_TEMPLATES)} SEEN templates "
          f"(all used in training)")
    print(f"  (8 held-out questions are EXCLUDED from training)")
    print(f"\nv2: VQA answers are SENTENCE-LEVEL (not single-word)")

    # ── Build examples ──
    examples = []
    skipped_no_metadata = 0
    skipped_no_image = 0
    plate_only_count = 0

    task_counts = {
        "captioning": 0,
        "ingredients": 0,
        "calories": 0,
        "macros": 0,
        "vqa": 0,
    }

    for i, dish_id in enumerate(train_ids):
        # Check metadata exists
        if dish_id not in metadata:
            skipped_no_metadata += 1
            continue

        dish = metadata[dish_id]
        ingredients = get_ingredient_names(dish)
        is_plate_only = (ingredients == ["plate only"] or dish["calories"] == 0)

        if is_plate_only:
            plate_only_count += 1

        # Check image exists
        image_path = config.RAW_DATA_DIR / "realsense_overhead" / dish_id / "rgb.png"
        if not image_path.exists():
            skipped_no_image += 1
            continue

        # Relative path from project root (for portability)
        image_rel = str(image_path.relative_to(config.PROJECT_ROOT))

        # ── Task 1: Captioning ──
        examples.append({
            "dish_id": dish_id,
            "task": "captioning",
            "prompt": task_prompts["captioning"],
            "answer": build_caption_answer(dish),
            "image_path": image_rel,
        })
        task_counts["captioning"] += 1

        # ── Task 2: Ingredients ──
        examples.append({
            "dish_id": dish_id,
            "task": "ingredients",
            "prompt": task_prompts["ingredients"],
            "answer": build_ingredients_answer(dish),
            "image_path": image_rel,
        })
        task_counts["ingredients"] += 1

        # ── Task 3: Calories ──
        # Include even plate-only dishes (answer = "0 calories")
        # The model should learn that empty plates have 0 calories
        examples.append({
            "dish_id": dish_id,
            "task": "calories",
            "prompt": task_prompts["calories"],
            "answer": build_calories_answer(dish),
            "image_path": image_rel,
        })
        task_counts["calories"] += 1

        # ── Task 4: Macros ──
        examples.append({
            "dish_id": dish_id,
            "task": "macros",
            "prompt": task_prompts["macros"],
            "answer": build_macros_answer(dish),
            "image_path": image_rel,
        })
        task_counts["macros"] += 1

        # ── Task 5: VQA — all 12 SEEN questions per dish ──
        # v2: Uses get_vqa_sentence_answer() for natural language answers
        # instead of get_vqa_answer() which returns single words.
        for question in config.VQA_QUESTION_TEMPLATES:
            answer = get_vqa_sentence_answer(question, dish)
            prompt = vqa_wrapper.format(question=question)

            examples.append({
                "dish_id": dish_id,
                "task": "vqa",
                "question": question,
                "prompt": prompt,
                "answer": answer,
                "image_path": image_rel,
            })
            task_counts["vqa"] += 1

        # Progress
        if (i + 1) % 500 == 0:
            print(f"  Processed {i + 1}/{len(train_ids)} dishes...")

    # ── Summary ──
    print(f"\n{'='*60}")
    print(f"TRAINING DATA SUMMARY (v2)")
    print(f"{'='*60}")
    print(f"  Total examples:       {len(examples)}")
    print(f"  Dishes used:          {len(train_ids) - skipped_no_metadata - skipped_no_image}")
    print(f"  Skipped (no metadata): {skipped_no_metadata}")
    print(f"  Skipped (no image):    {skipped_no_image}")
    print(f"  Plate-only dishes:     {plate_only_count}")
    print(f"\n  Per-task breakdown:")
    for task, count in task_counts.items():
        print(f"    {task:15s}: {count:,} examples")

    # ── Print samples for verification ──
    print(f"\n{'='*60}")
    print("SAMPLE TRAINING EXAMPLES (v2 — sentence-level VQA)")
    print(f"{'='*60}")

    # Show one example per task from the first non-plate-only dish
    sample_dish = None
    for ex in examples:
        if ex["task"] == "captioning" and "plate only" not in ex["answer"]:
            sample_dish = ex["dish_id"]
            break

    if sample_dish:
        print(f"\nDish: {sample_dish}")
        seen_tasks = set()
        for ex in examples:
            if ex["dish_id"] == sample_dish:
                task = ex["task"]
                # Show each task once (for VQA, show first 3)
                if task != "vqa" and task not in seen_tasks:
                    seen_tasks.add(task)
                    print(f"\n  [{task.upper()}]")
                    print(f"    Prompt: \"{ex['prompt']}\"")
                    print(f"    Answer: \"{ex['answer']}\"")
                    print(f"    Image:  {ex['image_path']}")
                elif task == "vqa" and task not in seen_tasks:
                    seen_tasks.add(task)
                    print(f"\n  [VQA] (showing 3 of 12 — NOTE: sentence-level answers)")
                    vqa_count = 0
                    for vqa_ex in examples:
                        if vqa_ex["dish_id"] == sample_dish and vqa_ex["task"] == "vqa":
                            print(f"    Q: \"{vqa_ex['question']}\"")
                            print(f"    A: \"{vqa_ex['answer']}\"")
                            vqa_count += 1
                            if vqa_count >= 3:
                                break

    # ── Verify answer distributions ──
    print(f"\n{'='*60}")
    print("ANSWER DISTRIBUTION CHECK")
    print(f"{'='*60}")

    # Check VQA sentence answer starts (should start with Yes/No or be descriptive)
    vqa_yes = sum(1 for ex in examples
                  if ex["task"] == "vqa" and ex["answer"].lower().startswith("yes"))
    vqa_no = sum(1 for ex in examples
                 if ex["task"] == "vqa" and ex["answer"].lower().startswith("no"))
    vqa_other = sum(1 for ex in examples
                    if ex["task"] == "vqa" and
                    not ex["answer"].lower().startswith("yes") and
                    not ex["answer"].lower().startswith("no"))
    print(f"\n  VQA answer distribution (sentence-level):")
    print(f"    starts with 'Yes': {vqa_yes:,} ({vqa_yes/(vqa_yes+vqa_no+vqa_other)*100:.1f}%)")
    print(f"    starts with 'No':  {vqa_no:,} ({vqa_no/(vqa_yes+vqa_no+vqa_other)*100:.1f}%)")
    print(f"    other (numeric/ingredient sentences): {vqa_other:,}")

    # Check VQA answer lengths
    vqa_lengths = [len(ex["answer"].split()) for ex in examples if ex["task"] == "vqa"]
    if vqa_lengths:
        import numpy as np
        print(f"\n  VQA answer length (words):")
        print(f"    min:    {min(vqa_lengths)}")
        print(f"    max:    {max(vqa_lengths)}")
        print(f"    mean:   {np.mean(vqa_lengths):.1f}")
        print(f"    median: {np.median(vqa_lengths):.1f}")

    # Check calorie range
    cal_answers = []
    for ex in examples:
        if ex["task"] == "calories":
            try:
                cal = int(ex["answer"].replace(" calories", ""))
                cal_answers.append(cal)
            except ValueError:
                pass

    if cal_answers:
        import numpy as np
        print(f"\n  Calorie answer distribution:")
        print(f"    min:    {min(cal_answers)}")
        print(f"    max:    {max(cal_answers)}")
        print(f"    mean:   {np.mean(cal_answers):.1f}")
        print(f"    median: {np.median(cal_answers):.1f}")

    # ── Save ──
    output_dir = config.TRAINING_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "v2_train_examples.json"
    with open(output_path, "w") as f:
        json.dump({
            "version": "v2",
            "model": "flan-t5-xl",
            "model_name": config.MODELS["flan-t5-xl"]["name"],
            "n_examples": len(examples),
            "n_dishes": len(train_ids) - skipped_no_metadata - skipped_no_image,
            "task_counts": task_counts,
            "vqa_answer_style": "sentence-level",
            "vqa_templates_used": len(config.VQA_QUESTION_TEMPLATES),
            "held_out_templates": len(config.VQA_HELD_OUT_TEMPLATES),
            "examples": examples,
        }, f, indent=2)
    print(f"\nSaved to: {output_path}")
    print(f"File size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")

    print(f"\n{'='*60}")
    print("NEXT STEP: Run LoRA fine-tuning (v2)")
    print(f"  python -m src.train.lora_finetune --target qformer 2>&1 | tee logs/12_v2_lora_train_qformer.log")
    print(f"{'='*60}")


if __name__ == "__main__":
    prepare_training_data()