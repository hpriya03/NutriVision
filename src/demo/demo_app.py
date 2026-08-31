"""
NutriVision — Interactive Demo (Gradio)

A polished dissertation demo that lets examiners:
  • Upload any food image
  • See all 5 tasks (captioning, ingredients, calories, macros, VQA)
  • Ask free-form VQA questions
  • Run all 20 standard VQA questions at once
  • Toggle between model configurations (Zero-Shot / Q-Former / LLM / Both)
  • See raw vs post-filtered outputs side-by-side with explanations

Usage:
    # Launch locally on RONIN (accessible via port forwarding):
    python -m src.demo.demo_app

    # Launch with a shareable public link:
    python -m src.demo.demo_app --share

    # Use a specific model config:
    python -m src.demo.demo_app --target both --share

Architecture:
    demo_app.py (this file)
      └── loads BLIP-2 + LoRA adapter
      └── imports post_filter.py functions for live correction
      └── Gradio UI with tabs for each task

Place at: src/demo/demo_app.py
"""

import argparse
import json
import time
from pathlib import Path

import torch
import gradio as gr
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration

from src import config
from src.eval.post_filter import (
    parse_number,
    parse_macros,
    parse_ingredients,
    clean_ingredients,
    check_calorie_range,
    check_macros_and_atwater,
    extract_model_numerics,
    correct_vqa,
    check_cross_task,
    build_ingredient_vocabulary,
)


# ═══════════════════════════════════════════════════════════
# GLOBAL STATE
# ═══════════════════════════════════════════════════════════

MODEL = None           # the loaded model (base or base+LoRA)
PROCESSOR = None       # the BLIP-2 processor
DEVICE = None          # cuda / cpu
VOCABULARY = None      # ingredient vocabulary for post-filter
CURRENT_TARGET = None  # which LoRA target is loaded (None = zero-shot)
BASE_MODEL = None      # reference to unmerged base for switching


# ═══════════════════════════════════════════════════════════
# MODEL LOADING
# ═══════════════════════════════════════════════════════════

def load_model(target: str = None):
    """
    Load BLIP-2 FlanT5-XL, optionally with a LoRA adapter.

    Args:
        target: "qformer", "llm", "both", or None for zero-shot
    """
    global MODEL, PROCESSOR, DEVICE, CURRENT_TARGET, VOCABULARY

    model_name = config.MODELS["flan-t5-xl"]["name"]
    print(f"\nLoading base model: {model_name}...")

    PROCESSOR = Blip2Processor.from_pretrained(model_name)
    model = Blip2ForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    print(f"  Base model loaded on {model.device}")

    if target is not None:
        from peft import PeftModel
        adapter_dir = (config.CHECKPOINTS_DIR
                       / f"v2_lora_flan-t5-xl_{target}" / "best")
        if not adapter_dir.exists():
            print(f"  WARNING: Adapter not found at {adapter_dir}")
            print(f"  Falling back to zero-shot")
            target = None
        else:
            print(f"Loading LoRA adapter ({target})...")
            model = PeftModel.from_pretrained(model, adapter_dir)
            print(f"  LoRA adapter loaded")

    model.eval()
    MODEL = model
    DEVICE = model.device
    CURRENT_TARGET = target

    # Build ingredient vocabulary for post-filter
    print("Building ingredient vocabulary...")
    VOCABULARY = build_ingredient_vocabulary()
    print(f"  Vocabulary: {len(VOCABULARY)} ingredients")

    label = target if target else "zero-shot"
    print(f"\nReady! Model: {label}")
    return label


# ═══════════════════════════════════════════════════════════
# INFERENCE HELPERS
# ═══════════════════════════════════════════════════════════

def run_single_prompt(image: Image.Image, prompt: str) -> str:
    """Run a single prompt on an image and return the text output."""
    inputs = PROCESSOR(images=image, text=prompt, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        generated_ids = MODEL.generate(
            **inputs,
            **config.GENERATION_CONFIG,
        )

    output = PROCESSOR.batch_decode(
        generated_ids, skip_special_tokens=True
    )[0].strip()

    return output


def run_all_tasks(image: Image.Image) -> dict:
    """
    Run Tasks 1-4 on an image.

    Returns:
        {"captioning": {"prompt": ..., "output": ...}, ...}
    """
    task_prompts = config.TASK_PROMPTS["flan-t5-xl"]
    results = {}

    for task_key, prompt in task_prompts.items():
        output = run_single_prompt(image, prompt)
        results[task_key] = {"prompt": prompt, "output": output}

    return results


def run_vqa_question(image: Image.Image, question: str) -> dict:
    """
    Run a single VQA question.

    Returns:
        {"question": ..., "output": ..., "split": "seen"/"unseen"/"custom"}
    """
    vqa_wrapper = config.VQA_PROMPT_WRAPPER["flan-t5-xl"]
    prompt = vqa_wrapper.format(question=question)
    output = run_single_prompt(image, prompt)

    # Determine split
    seen_set = set(config.VQA_QUESTION_TEMPLATES)
    unseen_set = set(config.VQA_HELD_OUT_TEMPLATES)
    if question in seen_set:
        split = "seen"
    elif question in unseen_set:
        split = "unseen"
    else:
        split = "custom"

    return {"question": question, "prompt_sent": prompt,
            "output": output, "split": split}


def run_all_vqa(image: Image.Image) -> list:
    """Run all 20 standard VQA questions."""
    all_questions = (config.VQA_QUESTION_TEMPLATES
                     + config.VQA_HELD_OUT_TEMPLATES)
    results = []
    for q in all_questions:
        results.append(run_vqa_question(image, q))
    return results


# ═══════════════════════════════════════════════════════════
# POST-FILTER (single dish, live)
# ═══════════════════════════════════════════════════════════

def run_post_filter_single(tasks: dict, vqa_list: list) -> dict:
    """
    Run post-filter on a single dish's outputs.

    Returns a dict with:
        - corrected_tasks: corrected task outputs
        - corrected_vqa: corrected VQA outputs
        - report: detailed report of what was checked/changed
    """
    report = {}

    # ── Check 1: Ingredient cleaning ──
    ingr_text = tasks.get("ingredients", {}).get("output", "")
    ingr_result = clean_ingredients(ingr_text, VOCABULARY)
    report["ingredients"] = {
        "removed": ingr_result["removed"],
        "flagged": ingr_result["flagged"],
        "original_count": ingr_result["original_count"],
        "cleaned_count": ingr_result["cleaned_count"],
    }

    corrected_tasks = {}
    corrected_tasks["ingredients"] = {
        "prompt": tasks.get("ingredients", {}).get("prompt", ""),
        "output": ingr_result["cleaned_text"],
    }

    # ── Check 2a: Calorie range ──
    cal_text = tasks.get("calories", {}).get("output", "")
    cal_result = check_calorie_range(cal_text)
    report["calorie_range"] = cal_result

    corrected_tasks["calories"] = dict(tasks.get("calories", {}))

    # ── Check 2b: Macro range + Atwater ──
    macro_text = tasks.get("macros", {}).get("output", "")
    macro_result = check_macros_and_atwater(macro_text, cal_text)
    report["macros_and_atwater"] = {
        "range_flags": macro_result["range_flags"],
        "atwater": macro_result["atwater"],
    }

    corrected_tasks["macros"] = dict(tasks.get("macros", {}))

    # Captioning: pass through
    corrected_tasks["captioning"] = dict(tasks.get("captioning", {}))

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

    # ── Check 4: Cross-task consistency ──
    cross_result = check_cross_task(tasks, vqa_list)
    report["cross_task"] = cross_result

    return {
        "corrected_tasks": corrected_tasks,
        "corrected_vqa": vqa_result["corrected_vqa"],
        "report": report,
    }


# ═══════════════════════════════════════════════════════════
# FORMATTING HELPERS
# ═══════════════════════════════════════════════════════════

def format_task_output(task_key: str, raw: str, corrected: str,
                       report: dict) -> str:
    """Format a single task's output as markdown with raw vs corrected."""
    display_name = config.TASK_DISPLAY_NAMES.get(task_key, task_key)
    lines = [f"### {display_name}\n"]

    if raw == corrected:
        lines.append(f"**Output:** {raw}\n")
    else:
        lines.append(f"**Raw output:** {raw}\n")
        lines.append(f"**Post-filtered:** {corrected}\n")

    # Add check details
    if task_key == "ingredients" and report.get("ingredients"):
        ingr = report["ingredients"]
        if ingr["removed"]:
            lines.append(f"**Removed** (non-food): {', '.join(ingr['removed'])}\n")
        if ingr["flagged"]:
            lines.append(f"**Flagged** (not in Nutrition5K vocab): "
                         f"{', '.join(ingr['flagged'])}\n")

    if task_key == "calories" and report.get("calorie_range"):
        cal = report["calorie_range"]
        if cal.get("in_range") is False:
            lines.append(f"**Flag:** Calories out of expected range "
                         f"(0-2500 cal)\n")

    if task_key == "macros" and report.get("macros_and_atwater"):
        ma = report["macros_and_atwater"]
        atwater = ma.get("atwater", {})
        if atwater.get("consistent") is False:
            disc = atwater.get("discrepancy_pct")
            if disc is not None:
                lines.append(f"**Atwater flag:** Calorie estimate and "
                             f"macros disagree by {disc:.1f}%\n")

    return "\n".join(lines)


def format_vqa_table(vqa_list: list, corrected_vqa: list,
                     corrections: list) -> str:
    """Format VQA results as a markdown table with corrections highlighted."""
    correction_map = {}
    for c in corrections:
        correction_map[c["question"]] = c

    lines = ["### VQA Results\n"]
    lines.append("| Split | Question | Raw Answer | Post-Filtered | Corrected? |")
    lines.append("|-------|----------|------------|---------------|------------|")

    for raw_qa, pf_qa in zip(vqa_list, corrected_vqa):
        q = raw_qa["question"]
        split = raw_qa.get("split", "?")
        raw_ans = raw_qa["output"]
        pf_ans = pf_qa["output"]
        corr = correction_map.get(q)

        if corr:
            reason_short = corr.get("reason", "")[:60]
            mark = f"Yes"
            lines.append(f"| {split} | {q} | {raw_ans} | **{pf_ans}** | "
                         f"{mark} |")
        else:
            lines.append(f"| {split} | {q} | {raw_ans} | {pf_ans} | — |")

    return "\n".join(lines)


def format_corrections_detail(corrections: list) -> str:
    """Format detailed correction explanations."""
    if not corrections:
        return "*No corrections needed — model outputs are internally consistent.*"

    lines = [f"### Correction Details ({len(corrections)} corrections)\n"]
    for c in corrections:
        lines.append(f"**Q:** {c['question']}")
        lines.append(f"  - **Was:** {c.get('original_answer', '?')} → "
                     f"**Now:** {c.get('corrected_to', '?')}")
        lines.append(f"  - **Why:** {c.get('reason', '')}\n")

    return "\n".join(lines)


def format_cross_task(cross: dict) -> str:
    """Format cross-task consistency results."""
    if not cross:
        return ""

    lines = ["### Cross-Task Consistency\n"]
    n = cross.get("n_inconsistencies", 0)

    if n == 0:
        lines.append("All cross-task checks passed.\n")
    else:
        lines.append(f"**{n} inconsistencies found:**\n")
        for item in cross.get("details", []):
            lines.append(f"- {item.get('check', '?')}: {item.get('detail', '')}")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════
# GRADIO CALLBACK FUNCTIONS
# ═══════════════════════════════════════════════════════════

def analyze_image(image):
    """
    Main callback: run all tasks + all VQA + post-filter on one image.

    Returns outputs for all Gradio components.
    """
    if image is None:
        return ("Upload an image to start.", "", "", "", "",
                "", "", "")

    if MODEL is None:
        return ("Model not loaded yet.", "", "", "", "",
                "", "", "")

    start = time.time()

    # Convert to PIL if needed
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    image = image.convert("RGB")

    # ── Run Tasks 1-4 ──
    tasks = run_all_tasks(image)

    # ── Run all 20 VQA questions ──
    vqa_list = run_all_vqa(image)

    # ── Run post-filter ──
    pf = run_post_filter_single(tasks, vqa_list)

    elapsed = time.time() - start

    # ── Format outputs ──

    # Caption
    caption_md = format_task_output(
        "captioning",
        tasks["captioning"]["output"],
        pf["corrected_tasks"]["captioning"]["output"],
        pf["report"],
    )

    # Ingredients
    ingr_md = format_task_output(
        "ingredients",
        tasks["ingredients"]["output"],
        pf["corrected_tasks"]["ingredients"]["output"],
        pf["report"],
    )

    # Calories
    cal_md = format_task_output(
        "calories",
        tasks["calories"]["output"],
        pf["corrected_tasks"]["calories"]["output"],
        pf["report"],
    )

    # Macros
    macro_md = format_task_output(
        "macros",
        tasks["macros"]["output"],
        pf["corrected_tasks"]["macros"]["output"],
        pf["report"],
    )

    # VQA table
    vqa_md = format_vqa_table(
        vqa_list,
        pf["corrected_vqa"],
        pf["report"]["vqa_correction"]["corrections"],
    )

    # Correction details
    corr_md = format_corrections_detail(
        pf["report"]["vqa_correction"]["corrections"]
    )

    # Cross-task
    cross_md = format_cross_task(pf["report"]["cross_task"])

    # Summary stats
    n_corr = pf["report"]["vqa_correction"]["n_corrections"]
    n_ingr_rm = len(pf["report"]["ingredients"]["removed"])
    atwater = pf["report"]["macros_and_atwater"]["atwater"]
    atwater_ok = "consistent" if atwater.get("consistent") else "inconsistent"

    target_label = CURRENT_TARGET if CURRENT_TARGET else "zero-shot"
    summary = (
        f"**Model:** {target_label} &nbsp;|&nbsp; "
        f"**Time:** {elapsed:.1f}s &nbsp;|&nbsp; "
        f"**VQA corrections:** {n_corr} &nbsp;|&nbsp; "
        f"**Ingredients removed:** {n_ingr_rm} &nbsp;|&nbsp; "
        f"**Atwater:** {atwater_ok}"
    )

    return (summary, caption_md, ingr_md, cal_md, macro_md,
            vqa_md, corr_md, cross_md)


def ask_single_question(image, question):
    """Callback for the free-form VQA tab."""
    if image is None:
        return "Upload an image first."
    if not question or not question.strip():
        return "Type a question."
    if MODEL is None:
        return "Model not loaded yet."

    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    image = image.convert("RGB")

    start = time.time()
    result = run_vqa_question(image, question.strip())
    elapsed = time.time() - start

    # For a single question, run a mini post-filter if it's a yes/no
    # by also running the numeric tasks
    raw_answer = result["output"]
    split_label = result["split"]

    output_lines = [
        f"**Question:** {question}",
        f"**Answer:** {raw_answer}",
        f"**Split:** {split_label}",
        f"**Time:** {elapsed:.1f}s",
    ]

    return "\n\n".join(output_lines)


# ═══════════════════════════════════════════════════════════
# EXAMPLE IMAGES
# ═══════════════════════════════════════════════════════════

def get_example_images(n: int = 6) -> list:
    """Get paths to example images from the test set."""
    splits_file = config.SPLITS_DIR / "test_ids.json"
    if not splits_file.exists():
        return []

    with open(splits_file) as f:
        test_ids = json.load(f)

    examples = []
    for dish_id in test_ids[:n * 3]:  # check more in case some missing
        image_path = config.IMAGES_DIR / dish_id / "rgb.png"
        if image_path.exists():
            examples.append([str(image_path)])
        if len(examples) >= n:
            break

    return examples


# ═══════════════════════════════════════════════════════════
# GRADIO UI
# ═══════════════════════════════════════════════════════════

def build_ui():
    """Build the Gradio interface."""

    target_label = CURRENT_TARGET if CURRENT_TARGET else "zero-shot"

    with gr.Blocks(
        title="NutriVision Demo",
        theme=gr.themes.Soft(),
        css="""
        .main-header { text-align: center; margin-bottom: 8px; }
        .summary-bar { background: #f0f4ff; padding: 10px 16px;
                       border-radius: 8px; margin: 8px 0; }
        """
    ) as demo:
        gr.Markdown(
            f"""
            # NutriVision
            **Investigating BLIP-2 Limitations on Food Image Understanding
            with LoRA Fine-Tuning on Nutrition5K**

            *Current model: **{target_label}** - Upload a food image to
            analyse all 5 tasks with live post-filter corrections.*
            """,
            elem_classes="main-header",
        )

        # ── Summary bar ──
        summary_out = gr.Markdown("", elem_classes="summary-bar")

        with gr.Row():
            # ── Left column: image input ──
            with gr.Column(scale=1):
                image_input = gr.Image(
                    type="pil", label="Upload Food Image",
                    height=350,
                )
                analyze_btn = gr.Button(
                    "Analyse Image", variant="primary", size="lg",
                )

                # Example images
                examples = get_example_images(6)
                if examples:
                    gr.Examples(
                        examples=examples,
                        inputs=[image_input],
                        label="Example dishes from Nutrition5K test set",
                    )

            # ── Right column: results ──
            with gr.Column(scale=2):
                with gr.Tabs():
                    # ── Tab 1: All Tasks Overview ──
                    with gr.Tab("Caption"):
                        caption_out = gr.Markdown("")
                    with gr.Tab("Ingredients"):
                        ingr_out = gr.Markdown("")
                    with gr.Tab("Calories"):
                        cal_out = gr.Markdown("")
                    with gr.Tab("Macros"):
                        macro_out = gr.Markdown("")

                    # ── Tab 5: VQA (all 20 questions) ──
                    with gr.Tab("VQA (All 20 Questions)"):
                        vqa_out = gr.Markdown("")

                    # ── Tab 6: Ask Your Own ──
                    with gr.Tab("Ask a Question"):
                        gr.Markdown(
                            "*Type any question about the uploaded food "
                            "image. The model will answer using visual "
                            "question answering.*"
                        )
                        with gr.Row():
                            question_input = gr.Textbox(
                                label="Your question",
                                placeholder="e.g. Is this dish high in protein?",
                                scale=4,
                            )
                            ask_btn = gr.Button("Ask", variant="primary",
                                                scale=1)
                        custom_vqa_out = gr.Markdown("")

                    # ── Tab 7: Post-Filter Details ──
                    with gr.Tab("Post-Filter Details"):
                        corr_out = gr.Markdown("")
                        cross_out = gr.Markdown("")

        # ── Wire up callbacks ──
        analyze_btn.click(
            fn=analyze_image,
            inputs=[image_input],
            outputs=[summary_out, caption_out, ingr_out, cal_out,
                     macro_out, vqa_out, corr_out, cross_out],
        )

        # Also trigger on image upload
        image_input.change(
            fn=analyze_image,
            inputs=[image_input],
            outputs=[summary_out, caption_out, ingr_out, cal_out,
                     macro_out, vqa_out, corr_out, cross_out],
        )

        ask_btn.click(
            fn=ask_single_question,
            inputs=[image_input, question_input],
            outputs=[custom_vqa_out],
        )

        # Also trigger on Enter in the text box
        question_input.submit(
            fn=ask_single_question,
            inputs=[image_input, question_input],
            outputs=[custom_vqa_out],
        )

    return demo


# ═══════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="NutriVision interactive demo")
    parser.add_argument(
        "--target", type=str, default="both",
        choices=["qformer", "llm", "both", "zero-shot"],
        help="Which model config to load (default: both)")
    parser.add_argument(
        "--share", action="store_true",
        help="Create a public Gradio share link")
    parser.add_argument(
        "--port", type=int, default=7860,
        help="Port to run on (default: 7860)")
    args = parser.parse_args()

    target = None if args.target == "zero-shot" else args.target

    # Load model
    load_model(target)

    # Build and launch UI
    demo = build_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share,
    )