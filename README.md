# NutriVision

Food image understanding and nutrition estimation with BLIP-2 and LoRA fine-tuning.

**Highlights:** LoRA fine-tuning of BLIP-2 (Flan-T5-XL) on Nutrition5K cut calorie error from 153 to 92 kcal while answering for almost every dish (506/507, up from 356/507 zero-shot), more than doubled ingredient F1 (0.29 → 0.66), and reached 84% accuracy on trained dietary question types. A rule-based post-filter raised accuracy on 8 never-seen question types from 61% to 80%.

Given a photo of a plate of food, NutriVision:

1. **Describes** the dish (captioning)
2. **Lists the ingredients**
3. **Estimates total calories**
4. **Estimates macronutrients** (protein, fat, carbohydrates in grams)
5. **Answers dietary questions** (VQA), e.g. *"Is this dish suitable for a low-carb diet?"*

The project compares two zero-shot BLIP-2 variants, fine-tunes the stronger one with LoRA in a three-way ablation (Q-Former only, LLM only, both), and adds a rule-based post-filter that catches hallucinations and inconsistent answers.

## Approach

| Stage | What happens |
|---|---|
| Zero-shot baseline | BLIP-2 + OPT-2.7B vs BLIP-2 + Flan-T5-XL, evaluated without any training |
| LoRA fine-tuning | Flan-T5-XL variant fine-tuned on Nutrition5K (r=16, α=32, 3 epochs, effective batch 32). Ablation over where the adapters go: `qformer`, `llm`, `both` |
| Generalisation test | Training uses 12 VQA question templates; 8 further templates are held out and only seen at test time |
| Post-filter | Removes non-food ingredients, flags out-of-range values, checks Atwater consistency (calories vs macros) and cross-task consistency, and corrects VQA answers that contradict the model's own numeric estimates |
| Demo | Gradio app to upload an image and see all five tasks, raw vs post-filtered |

## Results

Test set: 507 Nutrition5K dishes. Means with 95% bootstrap confidence intervals are in each run's `summary_metrics.csv`.

### Zero-shot model selection

| Metric | OPT-2.7B | Flan-T5-XL |
|---|---|---|
| Captioning ROUGE-1 | 0.379 | **0.427** |
| Ingredient F1 | **0.384** | 0.290 |
| Calorie MAE (kcal) | 244.7 | **152.8** |
| VQA yes/no accuracy (%) | 56.5 | **58.3** |

Flan-T5-XL was chosen for fine-tuning (instruction-tuned, better calorie estimates and captions). Full table: [`results/model_comparison.csv`](results/model_comparison.csv).

### Fine-tuning ablation (Flan-T5-XL)

| Metric | Zero-shot | Q-Former LoRA | LLM LoRA | Both LoRA |
|---|---|---|---|---|
| Captioning BLEU-1 | 0.280 | **0.630** | 0.585 | 0.590 |
| Captioning METEOR | 0.309 | **0.657** | 0.629 | 0.626 |
| Ingredient F1 | 0.290 | **0.655** | 0.560 | 0.577 |
| Calorie MAE (kcal) | 152.8 | **92.5** | 111.1 | 99.4 |
| Protein MAE (g) | 17.7 | 18.6 | 13.6 | **11.5** |
| Fat MAE (g) | 13.4 | **7.1** | 11.1 | 9.2 |
| Carb MAE (g) | 20.7 | **12.6** | 16.3 | 15.9 |
| VQA seen accuracy (%) | — | 74.7 | 80.7 | **83.7** |
| VQA unseen accuracy (%) | — | 58.6 | **64.8** | 60.7 |

Notes on reading this table:

- **Calories:** zero-shot Flan-T5 produced a usable number for only 356 of 507 dishes, so its MAE is averaged over those; fine-tuned models answered 506/507.
- **Macros:** zero-shot Flan-T5 almost never produced gram values (MAPE is exactly 100%), so its macro MAEs are effectively a predict-zero baseline.
- **VQA:** zero-shot was scored on 4 randomly sampled trained-type questions per dish against one-word answers, so it is not directly comparable to the fine-tuned seen/unseen accuracies, which cover all 20 questions with sentence-level answers. Zero-shot yes/no accuracy was 58.3%.
- Each configuration is a single training run (one seed), so small differences between the three LoRA variants may not be significant.

### Effect of the post-filter on VQA accuracy (%)

| Config | Raw | Post-filtered |
|---|---|---|
| Zero-shot | 58.3 | 76.6 |
| Q-Former LoRA — unseen questions | 58.6 | 77.1 |
| LLM LoRA — unseen questions | 64.8 | 81.0 |
| Both LoRA — unseen questions | 60.7 | 80.3 |
| Both LoRA — overall | 70.6 | **81.7** |

The post-filter mainly helps on unseen question types. On trained question types it is neutral or slightly negative (e.g. Q-Former 74.7% → 71.3%).

Full table: [`results/post_filter_comparison.csv`](results/post_filter_comparison.csv). Interactive version: [`results/post_filter_comparison.html`](results/post_filter_comparison.html).

### Training loss

![Training loss for the three LoRA variants](results/training_plots/ablation_comparison.png)

The LLM and Both runs reach a lower loss early but become unstable after about 1,000 steps and eventually produce NaN losses, which is where their curves end. The evaluated checkpoints are the best epoch before divergence (LLM: epoch 1, Both: epoch 2). See [Known limitations](#known-limitations).

## Known limitations

- **Training instability in float16.** The model is fine-tuned in float16 without loss scaling. Flan-T5 is known to overflow in float16, and the LLM and Both runs diverged to NaN. Lowering the learning rate from 2e-4 to 1e-4 only delayed it. Training in bfloat16 would be the fix.
- **Learning-rate schedule.** The scheduler's total step count is computed from batches rather than optimizer steps (gradient accumulation is 8), so warmup lasts about a quarter of training and the learning rate never decays below ~90% of peak. This likely contributed to the divergence above.
- Because of these two issues, the LLM and Both results probably understate what those variants can achieve, and the ablation should be read with that in mind.
- Single seed per configuration; no separate validation set, so the best checkpoint is selected on training loss.

## Repository structure

```
src/
  config.py                 Paths, model names, prompts, VQA templates, generation settings
  data_prep/
    create_splits.py        Train/test split from the official Nutrition5K splits
    metadata_parser.py      Reads Nutrition5K dish metadata CSVs
    prepare_training_data.py  Builds prompt/answer pairs for fine-tuning
  train/
    lora_finetune.py        LoRA fine-tuning (--target qformer | llm | both)
    plot_training.py        Loss curves and LR schedule plots
  eval/
    zero_shot_inference.py  Zero-shot inference (OPT or Flan-T5-XL)
    finetuned_inference.py  Inference with a LoRA adapter
    ground_truth.py         Builds the answer key from Nutrition5K metadata
    evaluate.py             BLEU / ROUGE / METEOR, ingredient P/R/F1, MAE/MAPE, VQA accuracy
    post_filter.py          Hallucination and consistency post-filter
    compare_models.py       Zero-shot OPT vs Flan-T5 table
    compare_ablation.py     Baseline vs fine-tuned table
    compare_post_filter.py  Raw vs post-filtered table and HTML dashboard
  demo/
    demo_app.py             Gradio demo
results/                    Model outputs, metrics, comparison tables and plots
logs/                       Console logs from every run, numbered in execution order
archive/v1/                 Results and logs from the first fine-tuning iteration
```

Files prefixed `v2_` are the final fine-tuning runs: sentence-level VQA answers and a split into 12 trained and 8 held-out question types. The zero-shot runs have no prefix; they were only run once and serve as the baseline.

`archive/v1/` holds the first iteration (one-word VQA answers, Q-Former only), kept for reference. It is not used by any reported result.

## Setup

Requires Python 3.12 and a CUDA GPU (experiments were run on a single GPU; BLIP-2 Flan-T5-XL is loaded in float16).

```bash
git clone https://github.com/hpriya03/nutrivision.git
cd nutrivision
pip install -r requirements.txt
```

### Data

The [Nutrition5K dataset](https://github.com/google-research-datasets/Nutrition5k) is not included. Download it and arrange it as:

```
data/
  raw/
    metadata/               dish_metadata_cafe1.csv, dish_metadata_cafe2.csv
    realsense_overhead/     one folder per dish with rgb.png
  official_splits/          official RGB train/test split files
```

`data/` and `checkpoints/` are excluded from git.

## Running the pipeline

Run all commands from the repository root. The numbers match the files in `logs/`.

```bash
# 1. Splits
python src/data_prep/create_splits.py

# 3–6. Zero-shot inference (add --test-run for a 5-dish sanity check)
python -m src.eval.zero_shot_inference --model opt
python -m src.eval.zero_shot_inference --model flan-t5-xl

# 7. Ground truth
python -m src.eval.ground_truth

# 8–10. Evaluate and compare zero-shot models
python -m src.eval.evaluate --results-dir results/zero_shot_opt --ground-truth results/ground_truth.json
python -m src.eval.evaluate --results-dir results/zero_shot_flan-t5-xl --ground-truth results/ground_truth.json
python -m src.eval.compare_models

# 11. Training data
python -m src.data_prep.prepare_training_data

# 12. LoRA fine-tuning (ablation)
python -m src.train.lora_finetune --target qformer
python -m src.train.lora_finetune --target llm --lr 1e-4   # 1e-4 used for the reported run
python -m src.train.lora_finetune --target both

# Training plots (no log file)
python -m src.train.plot_training --runs qformer llm both

# 13–14. Fine-tuned inference and evaluation (repeat for llm and both)
python -m src.eval.finetuned_inference --target qformer
python -m src.eval.evaluate --results-dir results/v2_finetuned_qformer

# 15. Post-filter (repeat for each v2 results directory and zero_shot_flan-t5-xl) and comparison
python -m src.eval.post_filter --results-dir results/v2_finetuned_qformer
python -m src.eval.compare_post_filter
```

## Demo

```bash
python -m src.demo.demo_app --target both          # or qformer | llm | zero-shot
python -m src.demo.demo_app --target both --share  # public Gradio link
```

The demo needs the LoRA checkpoints in `checkpoints/v2_lora_flan-t5-xl_{target}/best`.

## Acknowledgements

- [Nutrition5K](https://github.com/google-research-datasets/Nutrition5k) — Thames et al., CVPR 2021
- [BLIP-2](https://huggingface.co/Salesforce/blip2-flan-t5-xl) — Li et al., ICML 2023
- [LoRA](https://arxiv.org/abs/2106.09685) — Hu et al., ICLR 2022
