# src/config.py — Central configuration for NutriVision
#
# All paths, model definitions, task prompts, and evaluation settings
# live here so every script pulls from one source of truth.

from pathlib import Path

# ── Paths ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
IMAGES_DIR = RAW_DATA_DIR / "realsense_overhead"
SPLITS_DIR = DATA_DIR / "splits"
TRAINING_DIR = DATA_DIR / "training"
OFFICIAL_SPLITS_DIR = DATA_DIR / "official_splits"
RESULTS_DIR = PROJECT_ROOT / "results"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
LOGS_DIR = PROJECT_ROOT / "logs"
METADATA_DIR = RAW_DATA_DIR / "metadata"

# ── Models (to be tested in zero-shot; results will determine which to fine-tune) ──
MODELS = {
    "opt": {
        "name": "Salesforce/blip2-opt-2.7b",
        "description": "BLIP-2 + OPT-2.7B (autoregressive, NOT instruction-tuned)",
        # OPT is a causal (decoder-only) LM — generate() output includes
        # the prompt tokens, so we must strip them before saving. casual attention
        "is_causal": True,
    },
    "flan-t5-xl": {
        "name": "Salesforce/blip2-flan-t5-xl",
        "description": "BLIP-2 + FlanT5-XL (encoder-decoder, instruction-tuned)",
        # FlanT5 is encoder-decoder — generate() returns only the decoder
        # output (the answer), so no stripping needed.
        "is_causal": False,
    },
}

# ── Task prompts (model-specific) ──
# OPT is NOT instruction-tuned, so it doesn't understand commands like
# "Describe this food plate." — it treats that as text to continue, and
# since it's already a complete sentence, OPT just outputs EOS.
#
# Fix: use completion-style prompts for OPT ("Question: X Answer:")
# and instruction-style prompts for FlanT5. Both express the same
# semantic intent — the format just matches what each model understands.
# This is standard practice in VLM evaluation.
TASK_PROMPTS = {
    "opt": {
        # Empty string = pure image captioning (no text prompt).
        # BLIP-2 generates a description from the image alone.
        "captioning": "",
        "ingredients": "Question: What are the ingredients in this dish? Answer:",
        "calories": "Question: How many total calories are in this dish? Answer:",
        "macros": "Question: What are the protein, fat, and carbohydrate amounts in grams for this dish? Answer:",
    },
    "flan-t5-xl": {
        "captioning": "Describe this food plate in detail.",
        "ingredients": "List all the ingredients visible in this dish.",
        "calories": "Estimate the total calories in this dish.",
        "macros": "What is the macronutrient breakdown of this dish? Include protein, fat, and carbohydrates in grams.",
    },
}

# ── VQA prompt wrapping ──
# How to wrap a VQA question for each model.
# {question} is replaced with the actual question text.
VQA_PROMPT_WRAPPER = {
    "opt": "Question: {question} Answer:",
    "flan-t5-xl": "{question}",
}

# ── Task display names (for output reports) ──
TASK_DISPLAY_NAMES = {
    "captioning": "What's On Your Plate",
    "ingredients": "Ingredients Detected",
    "calories": "Estimated Calories",
    "macros": "Nutrition Breakdown",
    "vqa": "Dietary Insights",
}

# ── VQA question templates ──
# 12 diverse questions covering nutrition, ingredients, and dietary suitability.
# During inference, 4 are randomly sampled per dish (seeded for reproducibility).
VQA_QUESTION_TEMPLATES = [
    "How many calories does this dish have?",
    "Is this dish high in protein?",
    "What are the main ingredients in this dish?",
    "Would this be considered a balanced meal?",
    "Is this dish suitable for a low-carb diet?",
    "How much fat does this dish contain?",
    "Is this a healthy meal option?",
    "What food groups are represented in this dish?",
    "Is this dish suitable for someone watching their calorie intake?",
    "How much protein does this dish provide?",
    "What is the approximate carbohydrate content?",
    "Would this dish be appropriate for a diabetic diet?",
]

# ── Held-out VQA question templates (unseen during training) ──
# These 8 questions are NEVER included in training data.
# They test the model's ability to generalise its nutritional reasoning
# to novel question formulations — a key contribution of this work.
# All have computable ground truth from Nutrition5K metadata.
VQA_HELD_OUT_TEMPLATES = [
    "Is this dish low in fat?",
    "Does this dish have more than 500 calories?",
    "Is this a light meal?",
    "Does this dish have more fat than protein?",
    "Is this dish high in carbohydrates?",
    "Would this be suitable for a weight loss diet?",
    "Is this a vegetarian dish?",
    "Would this be a good post-workout meal?",
]

# ── Generation settings ──
GENERATION_CONFIG = {
    "max_new_tokens": 128,   # Upper bound; model stops at EOS if done earlier
    "do_sample": False,       # Greedy decoding for reproducibility, no sampling
    "num_beams": 1,           # No beam search (greedy)
}