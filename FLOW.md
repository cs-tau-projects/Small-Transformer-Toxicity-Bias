# Project Flow -- End-to-End Pipeline Documentation

> **Small Transformer Toxicity Bias** is a research pipeline that measures **bias in small pre-trained transformers** using the [Jigsaw Unintended Bias in Toxicity Classification](https://huggingface.co/datasets/shuttie/jigsaw-unintended-bias) dataset and [ToxiGen](https://huggingface.co/datasets/skg/toxigen-data) for out-of-domain evaluation. Models are evaluated both *before* and *after* fine-tuning across identity subgroups (race, gender, religion, etc.) using ROC-AUC, Subgroup AUC, BPSN AUC, BNSP AUC, FNR, and FPR.

---

## Table of Contents

1. [High-Level Architecture](#1-high-level-architecture)
2. [Entry Point & CLI](#2-entry-point--cli)
3. [Step 1 -- Data Loading & Preprocessing](#3-step-1--data-loading--preprocessing)
4. [Step 2 -- Baselines](#4-step-2--baselines)
5. [Step 3 -- Raw Transformer Evaluation](#5-step-3--raw-transformer-evaluation)
6. [Step 4 -- Fine-Tuning Transformers](#6-step-4--fine-tuning-transformers)
7. [Step 5 -- Fine-Tuned Evaluation](#7-step-5--fine-tuned-evaluation)
8. [Step 6 -- Out-of-Domain Evaluation](#8-step-6--out-of-domain-evaluation)
9. [Step 7 -- LLaMA Zero-Shot Evaluation](#9-step-7--llama-zero-shot-evaluation)
10. [Step 8 -- Dataset & Error Analysis](#10-step-8--dataset--error-analysis)
11. [Step 9 -- Report Generation](#11-step-9--report-generation)
12. [Directory Layout & Saved Artifacts](#12-directory-layout--saved-artifacts)
13. [Evaluation Metrics Deep-Dive](#13-evaluation-metrics-deep-dive)
14. [Understanding the Output](#14-understanding-the-output)
15. [Reproducibility](#15-reproducibility)

---

## 1. High-Level Architecture

```
main.py  (CLI entry point)
│
├── data           → src/steps/data_step.py       → src/data/dataset.py + src/data/data_loader.py
├── baseline       → src/steps/baseline_step.py   → src/model/baseline.py + src/model/naive_baseline.py
├── eval-raw       → src/steps/eval_raw_step.py   → src/steps/utils.py + src/evaluator.py
├── finetune       → src/steps/finetune_step.py   ──spawns──▶ python -m src.train
├── eval-finetuned → src/steps/eval_ft_step.py    → src/steps/utils.py + src/evaluator.py
├── eval-ood       → src/steps/eval_ood_step.py   → src/evaluator.py (ToxiGen dataset)
├── llama          → src/steps/llama_step.py      → src/evaluator.py (ID + OOD)
├── analysis       → src/analysis.py              → dataset stats + error sampling
└── report         → src/steps/report_step.py     → aggregation + structured logging
```

Each step is **independently runnable** via `--step <name>`. Most steps (data, baseline, finetune, eval-finetuned, eval-ood, llama, analysis, report) are also run sequentially via `--step all`. The `eval-raw` step is excluded from `all` and must be run explicitly.

---

## 2. Entry Point & CLI

**File:** [`main.py`](main.py) -- `main()` (line 7)

```bash
python main.py \
  --step all \
  --output_dir ./outputs \
  --models distilbert-base-uncased distilroberta-base google/bert_uncased_L-4_H-512_A-8 \
  --train_samples 20000 \
  --eval_samples 5000 \
  --seed 42
```

| Argument | Default | Description |
|---|---|---|
| `--step` | `all` | Which pipeline step to run (`data`, `baseline`, `eval-raw`, `finetune`, `eval-finetuned`, `eval-ood`, `llama`, `analysis`, `report`, `all`) |
| `--output_dir` | `./outputs` | Root directory for all caches, data, models, and results |
| `--models` | 3 small transformers | List of HuggingFace model identifiers to evaluate |
| `--llama_model` | `meta-llama/Llama-3.2-1B` | LLaMA model for zero-shot step |
| `--llama_batch_size` | `32` | Batch size for LLaMA zero-shot evaluation |
| `--train_samples` | `20000` | Max training samples for baseline and fine-tuning. Pass `-1` to use the **full** training set. |
| `--eval_samples` | `5000` | Max evaluation samples across all evaluation steps. Pass `-1` for the **full** evaluation set. |
| `--seed` | `42` | Global random seed for reproducibility |

**Reproducibility setup** happens at [`main.py` lines 34-37](main.py#L34-L37): `set_seed()`, `deterministic=True`, `benchmark=False`.

**Shared directories** are created at [`main.py` lines 40-46](main.py#L40-L46):

```
outputs/
├── .cache/          ← HuggingFace model + dataset cache
├── data/            ← Preprocessed & saved dataset splits
└── results/         ← Per-step CSV metric files
```

---

## 3. Step 1 -- Data Loading & Preprocessing

**Step file:** [`src/steps/data_step.py`](src/steps/data_step.py) -- `run_data_step()` (line 9)

### 3.1 Downloading the Raw Dataset

**File:** [`src/data/data_loader.py`](src/data/data_loader.py) -- `get_jigsaw_dataset()`

The dataset source is a community HuggingFace mirror of Jigsaw (`shuttie/jigsaw-unintended-bias`). Both the `train` and `test` splits are loaded:

```
hf://datasets/shuttie/jigsaw-unintended-bias/data/train.csv.gz
hf://datasets/shuttie/jigsaw-unintended-bias/data/test_private_expanded.csv.gz
```

- **Cache location:** `<output_dir>/.cache/`
- **Authentication:** HuggingFace token fetched via [`src/data/data_utils.py`](src/data/data_utils.py) `get_hf_token()`, which checks `HF_TOKEN` env variable (`.env` file) or the local HF token cache.

### 3.2 Preprocessing

**File:** [`src/data/dataset.py`](src/data/dataset.py) -- `download_and_prep_jigsaw()` (line 14)

After loading, the dataset is processed using HuggingFace's Arrow memory-mapped backend:

1. **Toxicity binarization**: `is_toxic = int(target >= 0.5)` -- continuous toxicity scores above 0.5 are labelled as toxic.
2. **Text cleaning**: `None` values in `comment_text` are replaced with `""`.
3. **Identity columns kept as continuous**: All 24 identity columns (e.g., `asian`, `muslim`, `transgender`) are kept as float values in [0, 1] -- they are NOT binarized -- so that subgroup membership can be graded.

The 24 identity columns are defined in [`src/data/dataset.py` line 6](src/data/dataset.py#L6) (`ALL_IDENTITY_COLUMNS`).

### 3.3 Train / Validation / Test Split & Saving

Back in [`src/steps/data_step.py`](src/steps/data_step.py):

1. Both `train` and `test` splits are **shuffled** with a fixed seed (line 26-27).
2. The training split is divided **90/10**: first 90% -> training set, last 10% -> internal validation set (lines 31-33).
3. The Jigsaw `test` split is used as the held-out test set.
4. Slices are taken per CLI arguments (lines 36-41):
   - `train` <- up to `--train_samples` from the 90% training portion
   - `test` <- up to `--eval_samples` from the Jigsaw test split

**Saved to disk** (lines 44-53):
```
outputs/data/
├── train/                ← Arrow dataset (training split, 90% of Jigsaw train)
├── val/                  ← Arrow dataset (validation split, 10% of Jigsaw train)
├── test/                 ← Arrow dataset (Jigsaw test split)
└── identity_columns.json ← Intersection of identity columns found in both splits
```

These Arrow datasets are reused by all subsequent steps without re-downloading.

---

## 4. Step 2 -- Baselines

**Step file:** [`src/steps/baseline_step.py`](src/steps/baseline_step.py) -- `run_baseline_step()` (line 12)

### Data Loading

Uses [`src/steps/utils.py`](src/steps/utils.py) `load_saved_data()` to load the Arrow datasets from `outputs/data/` -- no network call needed.

### 4.1 ML Baseline (TF-IDF + Logistic Regression)

A scikit-learn `Pipeline` with two stages:

```python
Pipeline([
    ('tfidf', TfidfVectorizer(max_features=10000, stop_words='english')),
    ('clf', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42))
])
```

- Trained on `train` split.
- Evaluated on `test` split.
- **Model saved** to `outputs/results/baseline_pipeline.joblib` for reuse in OOD evaluation.

### 4.2 Naive Baseline (Majority Vote)

A `MajorityVoteClassifier` that predicts the most common class (non-toxic) for all inputs.

- Saved to `outputs/results/naive_baseline.joblib` for OOD reuse.

### Output

```
outputs/results/
├── baseline_metrics.csv         ← ML baseline bias metrics
├── naive_baseline_metrics.csv   ← Naive baseline bias metrics
├── baseline_pipeline.joblib     ← Trained ML pipeline
├── naive_baseline.joblib        ← Trained naive model
├── preds_Baseline.csv           ← Per-example predictions (ML)
└── preds_Naive.csv              ← Per-example predictions (Naive)
```

---

## 5. Step 3 -- Raw Transformer Evaluation

**Step file:** [`src/steps/eval_raw_step.py`](src/steps/eval_raw_step.py) -- `run_eval_raw_step()` (line 6)

> **Note:** This step is **excluded** from `--step all` and must be run explicitly with `--step eval-raw`.

For each model in `--models`:

1. **Tokenizer** loaded from HuggingFace with `cache_dir=outputs/.cache/`.
2. **Model** loaded with `num_labels=2` (binary toxicity classification) from HuggingFace cache.
   - This is the **pre-trained model with untrained classification head** -- no fine-tuning.
3. **Inference** run via [`src/steps/utils.py`](src/steps/utils.py) `eval_transformer()` in batches of 32.

### Output

One CSV per model in `outputs/results/`:
```
distilbert-base-uncased_raw_metrics.csv
distilroberta-base_raw_metrics.csv
google_bert_uncased_L-4_H-512_A-8_raw_metrics.csv
```

(`/` in model names replaced by `_`)

---

## 6. Step 4 -- Fine-Tuning Transformers

**Step file:** [`src/steps/finetune_step.py`](src/steps/finetune_step.py) -- `run_finetune_step()` (line 9)

This step does **NOT** run training inline -- it **spawns a subprocess** per model:

```bash
python -m src.train \
  --model_name <model_name> \
  --output_base_dir outputs/finetuned_<safe_name>/ \
  --epochs 1 \
  --batch_size 32 \
  --seed 42 \
  --train_samples <train_samples> \
  --cache_dir <cache_dir> \
  --data_dir <data_dir>
```

**Skip logic** (line 16): If `outputs/finetuned_<safe_name>/small-transformer-toxicity/config.json` already exists, fine-tuning is **skipped** -- the existing checkpoint is reused.

**Data consistency** (line 38-39): The `--data_dir` flag is forwarded so that `src/train.py` reuses the same train/val splits created by the data step, avoiding split inconsistencies.

### The Training Script: `src/train.py`

**File:** [`src/train.py`](src/train.py) -- `main()` (line 87)

#### Data Loading inside train.py

Two loading paths:

1. **Preferred path** (lines 115-141): When `--data_dir` is provided, loads the pre-saved `train/` and `val/` splits from disk. This guarantees the training script uses the same splits as all other pipeline steps.
2. **Fallback path** (lines 143-159): When running standalone without `--data_dir`, downloads from HuggingFace and performs an independent 80/10/10 split.

Tokenization is handled by [`src/data/dataset.py`](src/data/dataset.py) `tokenize_jigsaw_dataset()`:
- Pads/truncates to `max_length=128`.
- Uses HF Arrow memory-mapped `.map()`.

Data is then wrapped in `JigsawDataset` (lines 168-169), which pre-calculates `identity_matrix` as a NumPy array of shape `(N, num_identities)` for efficient metric computation during evaluation.

#### Training Arguments

Lines 178-197 -- key settings:

| Argument | Value | Reason |
|---|---|---|
| `warmup_ratio` | `0.1` | Ramp up LR over first 10% of steps to protect pre-trained weights |
| `lr_scheduler_type` | `"linear"` | Linear decay after warmup (Devlin et al., 2019) |
| `eval_strategy` | `"epoch"` | Evaluate after each epoch |
| `save_strategy` | `"epoch"` | Save checkpoint after each epoch |
| `save_total_limit` | `2` | Keep only the last 2 checkpoints |
| `load_best_model_at_end` | `True` | Restore the best checkpoint at end of training |
| `metric_for_best_model` | `"roc_auc"` | Best model selected by overall ROC-AUC |
| `fp16` | `True` (if CUDA) | Mixed-precision training for speed |

#### Where Checkpoints Are Saved

```
outputs/finetuned_<safe_model_name>/small-transformer-toxicity/
├── checkpoint-<step>/    ← Intermediate checkpoints (max 2 kept)
│   ├── config.json
│   ├── model.safetensors
│   ├── optimizer.pt
│   ├── trainer_state.json
│   └── ...
├── config.json           ← Best model saved here at end of training
├── model.safetensors     ← Best model weights
├── tokenizer_config.json
└── ...
```

#### Metrics During Training

At the end of each epoch, the HuggingFace `Trainer` calls `compute_metrics_wrapper()` -> `compute_metrics()` (line 34):

1. Extracts predicted probabilities from logits (softmax for 2-class, sigmoid for 1-class).
2. Retrieves `identity_matrix` from `val_dataset.identity_matrix` (pre-calculated in `JigsawDataset.__init__()`).
3. Calls `evaluate_models_metrics()` (alias for `evaluate_bias()`) from [`src/evaluator.py`](src/evaluator.py).
4. Returns a dict: `{"roc_auc": <value>, "<identity>_subgroup_auc": ..., "<identity>_bpsn_auc": ..., "<identity>_bnsp_auc": ..., "<identity>_subgroup_fnr": ..., "<identity>_subgroup_fpr": ...}`.

---

## 7. Step 5 -- Fine-Tuned Evaluation

**Step file:** [`src/steps/eval_ft_step.py`](src/steps/eval_ft_step.py) -- `run_eval_ft_step()` (line 10)

For each model:

1. Looks for the fine-tuned model at `outputs/finetuned_<safe_name>/small-transformer-toxicity/`.
2. Prefers the root directory (if `config.json` exists there = best model saved by Trainer).
3. Falls back to the **latest checkpoint** (by step number) if the root doesn't have `config.json` (lines 21-25).
4. Loads the tokenizer from the **original model name** (not the fine-tuned path) and the model weights from the fine-tuned path (lines 28-34).
5. Runs inference + evaluation via `eval_transformer()` in `src/steps/utils.py`.

### Output

```
outputs/results/
├── distilbert-base-uncased_finetuned_metrics.csv
├── distilroberta-base_finetuned_metrics.csv
├── google_bert_uncased_L-4_H-512_A-8_finetuned_metrics.csv
├── preds_distilbert-base-uncased_finetuned.csv
├── preds_distilroberta-base_finetuned.csv
└── preds_google_bert_uncased_L-4_H-512_A-8_finetuned.csv
```

---

## 8. Step 6 -- Out-of-Domain Evaluation

**Step file:** [`src/steps/eval_ood_step.py`](src/steps/eval_ood_step.py) -- `run_eval_ood_step()` (line 182)

Evaluates all fine-tuned models **and** both baselines on the **ToxiGen** dataset to measure generalization beyond the training distribution.

### ToxiGen Dataset Loading

`load_toxigen_dataset()` (line 92) loads from HuggingFace (`skg/toxigen-data`) with multiple fallbacks for label standardization:
- `toxicity_human > 3` -> toxic
- `toxicity_human_annotated`
- `label` (if already binary)
- `toxicity_score >= 0.5`

The standardized dataset is saved to `outputs/data/toxigen_standardized.parquet` for reuse by the LLaMA step.

### Identity Group Extraction

`extract_toxigen_identities_and_evaluate()` (line 42) dynamically parses ToxiGen's `target_groups` field to construct the identity matrix. Groups are extracted from comma-separated strings or Python list literals.

### What Is Evaluated

1. **Baseline models**: ML (TF-IDF + LR) and Naive, loaded from their saved `.joblib` files.
2. **Fine-tuned transformers**: Each model from `--models`, loaded from its checkpoint directory.

### Output

```
outputs/results/
├── ood_toxigen_metrics.csv            ← Combined metrics for all models
├── preds_Baseline_ood.csv             ← Per-example OOD predictions (ML baseline)
├── preds_Naive_ood.csv                ← Per-example OOD predictions (Naive)
├── preds_<safe_name>_finetuned_ood.csv  ← Per-example OOD predictions (transformers)
```

---

## 9. Step 7 -- LLaMA Zero-Shot Evaluation

**Step file:** [`src/steps/llama_step.py`](src/steps/llama_step.py) -- `run_llama_step()` (line 118)

Evaluates a **causal language model** (e.g., `meta-llama/Llama-3.2-1B`) in a **zero-shot** fashion on both the in-distribution (Jigsaw) and out-of-domain (ToxiGen) test sets.

### Prompt Format

For **Instruct models** (with a chat template), the prompt uses the structured chat format:
```
System: You are a fair and unbiased moderator. Your task is to classify whether
        a comment is toxic. [...] Answer strictly with 'Yes' if toxic, or 'No'
        if not toxic.
User:   Comment: '<comment_text>'
```

For **base models** (without a chat template), the fallback prompt is:
```
Comment: "<comment_text>"
Is this comment toxic? Answer:
```

### Scoring Method

**Function:** `get_llama_toxicity_scores()` (line 60)

The model generates logits at the final token position. All vocabulary variants of "Yes" and "No" are aggregated via `logsumexp`, and the toxicity score is computed as:

```
P(toxic) = sigmoid(logsumexp(yes_logits) - logsumexp(no_logits))
```

This gives a continuous score in [0, 1] without requiring text generation.

### Optimizations

- **DataLoader with workers**: Background tokenization via 4 workers on CUDA (line 77-84).
- **`torch.compile`**: Applied on CUDA with compute capability >= 7 (line 149-154).
- **`logits_to_keep=1`**: Only computes logits for the last token position.
- **Periodic checkpointing**: Saves partial results every 10k samples for long runs.

### Dual Evaluation

1. **In-Distribution (ID)**: Evaluated on the Jigsaw test set. Results saved as `<safe_name>_raw_metrics.csv`.
2. **Out-of-Distribution (OOD)**: If `toxigen_standardized.parquet` exists (generated by the `eval-ood` step), LLaMA is also evaluated on ToxiGen and results are appended to `ood_toxigen_metrics.csv`.

### Output

```
outputs/results/
├── meta-llama_Llama-3.2-1B_raw_metrics.csv        ← ID metrics
├── preds_meta-llama_Llama-3.2-1B_llama.csv         ← ID predictions
├── ood_toxigen_metrics.csv                          ← OOD metrics (appended)
└── preds_meta-llama_Llama-3.2-1B_llama_ood.csv     ← OOD predictions
```

---

## 10. Step 8 -- Dataset & Error Analysis

**Step file:** [`src/analysis.py`](src/analysis.py) -- `run_analysis_step()` (line 120)

Computes dataset statistics and samples classification errors for qualitative inspection.

### What It Computes

1. **Jigsaw subgroup statistics**: Per-identity sample counts, toxic/non-toxic breakdown, and toxicity rates on the test set.
2. **ToxiGen subgroup statistics**: Same breakdown for the OOD dataset (if `toxigen_standardized.parquet` exists).
3. **Error analysis**: Samples the top-10 most confident False Positives and False Negatives from the first fine-tuned model's predictions for manual inspection.

### Output

```
outputs/results/
├── dataset_stats.csv           ← Jigsaw test set subgroup distributions
├── dataset_stats_toxigen.csv   ← ToxiGen subgroup distributions
└── error_analysis.csv          ← Sampled FP/FN examples with scores
```

---

## 11. Step 9 -- Report Generation

**Step file:** [`src/steps/report_step.py`](src/steps/report_step.py) -- `run_report_step()` (line 191)

### What It Does

1. Scans `outputs/results/` for all `*_metrics.csv` files and `ood_toxigen_metrics.csv`.
2. Maps filenames back to display names (e.g., `baseline_metrics.csv` -> `"Baseline"`, `<safe_name>_finetuned_metrics.csv` -> `"<model> Finetuned"`).
3. Merges all DataFrames on the `"Identity"` column into a single wide comparison table.
4. Prints 6 comparison sections to stdout via `format_final_report()`:
   - **1. Overall AUC Comparison**
   - **2. Subgroup AUC Comparison**
   - **3. BPSN AUC Comparison** (Over-flagging Detection)
   - **4. BNSP AUC Comparison** (Under-flagging Detection)
   - **5. FNR Comparison**
   - **6. FPR Comparison**
5. Compiles a `final_predictions.csv` merging all per-example predictions from all models across both datasets.
6. Logs all results to a structured `results.csv` append-only log for cross-experiment comparison.

### Output

```
outputs/results/
├── final_report.csv       ← Wide CSV with all models x all metrics
├── final_predictions.csv  ← Per-example predictions from all models
└── results.csv            ← Structured log (appended across runs)
```

---

## 12. Directory Layout & Saved Artifacts

```
outputs/                                      ← --output_dir
│
├── .cache/                                   ← HuggingFace Downloads Cache
│   └── (model weights, tokenizers, datasets -- Arrow format)
│
├── data/                                     ← Preprocessed Dataset Splits
│   ├── train/                                ← Arrow dataset (90% of Jigsaw train)
│   ├── val/                                  ← Arrow dataset (10% of Jigsaw train)
│   ├── test/                                 ← Arrow dataset (Jigsaw test split)
│   ├── identity_columns.json                 ← ["asian", "muslim", ...]
│   └── toxigen_standardized.parquet          ← Standardized ToxiGen (created by eval-ood)
│
├── finetuned_distilbert-base-uncased/        ← Per-model fine-tune output
│   └── small-transformer-toxicity/
│       ├── checkpoint-<step>/                ← Intermediate checkpoint (max 2 kept)
│       ├── config.json                       ← Best model config
│       ├── model.safetensors                 ← Best model weights
│       └── trainer_state.json                ← Training log (loss, metrics per epoch)
│
├── finetuned_distilroberta-base/             ← Same structure per model
│   └── small-transformer-toxicity/
│       └── ...
│
└── results/                                  ← All Metrics & Predictions
    ├── baseline_metrics.csv
    ├── naive_baseline_metrics.csv
    ├── baseline_pipeline.joblib
    ├── naive_baseline.joblib
    ├── distilbert-base-uncased_finetuned_metrics.csv
    ├── distilroberta-base_finetuned_metrics.csv
    ├── google_bert_uncased_L-4_H-512_A-8_finetuned_metrics.csv
    ├── ood_toxigen_metrics.csv
    ├── meta-llama_Llama-3.2-1B_raw_metrics.csv
    ├── dataset_stats.csv
    ├── dataset_stats_toxigen.csv
    ├── error_analysis.csv
    ├── preds_*.csv                           ← Per-example predictions
    ├── final_report.csv                      ← Combined comparison table
    ├── final_predictions.csv                 ← All predictions merged
    └── results.csv                           ← Structured experiment log
```

---

## 13. Evaluation Metrics Deep-Dive

All metric computation goes through [`src/evaluator.py`](src/evaluator.py) -- the core function is `evaluate_bias()` (line 79), aliased as `evaluate_models_metrics`.

### Inputs

| Parameter | Shape | Description |
|---|---|---|
| `y_true` | `(N,)` | Binary ground-truth toxicity labels |
| `y_pred_probs` | `(N,)` | Predicted probability of being toxic |
| `identity_matrix` | `(N, K)` | Continuous identity scores per example |
| `identity_columns` | `list[str]` | Names for each of the K identity columns |
| `threshold` | `float` | Binarization threshold for FNR/FPR (default 0.5) |

### Subgroup Membership

An example is considered **part of a subgroup** if its identity score >= 0.5 (line 108). This uses the continuous annotation values directly from the dataset.

### Metrics Computed

For each identity (one row per identity in the output DataFrame):

| Column | Function | Description |
|---|---|---|
| `1. Overall AUC` | `roc_auc_score(y_true, y_pred_probs)` | ROC-AUC on the entire eval set |
| `2. Overall FNR` | `compute_fnr()` (line 67) | FN / (FN + TP) on the entire eval set |
| `3. Overall FPR` | `compute_fpr()` (line 73) | FP / (FP + TN) on the entire eval set |
| `4. Subgroup AUC` | `compute_subgroup_auc()` (line 6) | ROC-AUC restricted to this identity subgroup |
| `5. BPSN AUC` | `compute_bpsn_auc()` (line 21) | Background Positive, Subgroup Negative AUC -- detects **over-flagging** of a subgroup |
| `6. BNSP AUC` | `compute_bnsp_auc()` (line 44) | Background Negative, Subgroup Positive AUC -- detects **under-flagging** of a subgroup |
| `7. Subgroup FNR` | `compute_fnr()` on subgroup mask | FNR restricted to this identity subgroup |
| `8. Subgroup FPR` | `compute_fpr()` on subgroup mask | FPR restricted to this identity subgroup |

> **Note:** Overall AUC/FNR/FPR values are the *same* on every row -- they describe the whole dataset and are repeated for easy CSV export and comparison.

### BPSN and BNSP AUC Explained

- **BPSN AUC** (Background Positive, Subgroup Negative): Measures whether the model confuses *mentioning an identity* with *being toxic*. A low BPSN AUC means the model over-flags non-toxic comments that mention the identity group.
- **BNSP AUC** (Background Negative, Subgroup Positive): Measures whether the model is too lenient on toxic content mentioning an identity. A low BNSP AUC means the model under-flags toxic comments that mention the identity group.

### Edge Cases

- If a subgroup has fewer than 2 unique label classes, AUC returns `NaN`.
- `compute_fnr` and `compute_fpr` return `NaN` if the denominator is zero.

---

## 14. Understanding the Output

### Per-Step CSVs (`outputs/results/*.csv`)

Each file has `K` rows (one per identity group) and the following columns:

```
Identity | Total Examples | 1. Overall AUC | 2. Overall FNR | 3. Overall FPR |
4. Subgroup AUC | 5. BPSN AUC | 6. BNSP AUC | 7. Subgroup FNR | 8. Subgroup FPR
```

Example interpretation:

```
Identity=muslim, Subgroup AUC=0.61 vs Overall AUC=0.85
→ The model is significantly worse at distinguishing toxic vs. non-toxic content
  when the comment mentions Muslim identity.

Identity=muslim, BPSN AUC=0.55
→ Low BPSN AUC suggests the model confuses mentioning "Muslim" with toxicity
  (over-flagging bias).

Identity=muslim, Subgroup FNR=0.40 vs Overall FNR=0.20
→ The model misses 40% of actual toxic comments mentioning Muslim identity,
  vs only 20% overall — indicating bias (higher false negative rate).
```

### Final Report (`outputs/results/final_report.csv`)

A wide comparison table where each model's metrics are shown side-by-side per identity. Printed to stdout in 6 sections:

```
1. Overall AUC Comparison
2. Subgroup AUC Comparison
3. BPSN AUC Comparison (Over-flagging Detection)
4. BNSP AUC Comparison (Under-flagging Detection)
5. FNR Comparison
6. FPR Comparison
```

This is the primary artifact for the research paper, showing how bias changes between:
- **Naive** baseline (majority vote)
- **Baseline** (TF-IDF + LogReg)
- **Fine-Tuned** transformers (after supervised training)
- **LLaMA** zero-shot via next-token probability
- **OOD** variants of the above (on ToxiGen)

---

## 15. Reproducibility

The pipeline uses multiple layered mechanisms to ensure reproducibility:

| Mechanism | Location | Detail |
|---|---|---|
| Global seed | [`main.py` line 34](main.py#L34) | `set_seed(args.seed)` from `transformers` |
| CuDNN determinism | [`main.py` lines 35-37](main.py#L35-L37) | `deterministic=True`, `benchmark=False` |
| Dataset shuffle seed | [`src/steps/data_step.py`](src/steps/data_step.py) | Seed passed from CLI for train/test shuffle |
| Trainer seed | [`src/train.py` line 192](src/train.py#L192) | `seed=args.seed` passed to `TrainingArguments` |
| Baseline seed | [`src/steps/baseline_step.py`](src/steps/baseline_step.py) | `random_state=42` in `LogisticRegression` |
| Checkpoint skip | [`src/steps/finetune_step.py` line 16](src/steps/finetune_step.py#L16) | If fine-tuned model exists, skip re-training |
| Data split consistency | [`src/steps/finetune_step.py` line 38](src/steps/finetune_step.py#L38) | `--data_dir` forwarded to `src/train.py` |

---

## Appendix: Key Function Reference

| Function | File | Line | Purpose |
|---|---|---|---|
| `main()` | [`main.py`](main.py) | 7 | CLI entry point, orchestrates all steps |
| `run_data_step()` | [`src/steps/data_step.py`](src/steps/data_step.py) | 9 | Load, split, and save dataset to disk |
| `get_jigsaw_dataset()` | [`src/data/data_loader.py`](src/data/data_loader.py) | 11 | Download dataset from HuggingFace mirror |
| `get_hf_token()` | [`src/data/data_utils.py`](src/data/data_utils.py) | 27 | Fetch HuggingFace API token |
| `download_and_prep_jigsaw()` | [`src/data/dataset.py`](src/data/dataset.py) | 14 | Binarize toxicity, clean identities |
| `tokenize_jigsaw_dataset()` | [`src/data/dataset.py`](src/data/dataset.py) | 59 | Tokenize text with HF tokenizer |
| `JigsawDataset` | [`src/data/dataset.py`](src/data/dataset.py) | 84 | PyTorch Dataset wrapper; pre-caches identity_matrix |
| `run_baseline_step()` | [`src/steps/baseline_step.py`](src/steps/baseline_step.py) | 12 | Train and evaluate both baselines |
| `run_eval_raw_step()` | [`src/steps/eval_raw_step.py`](src/steps/eval_raw_step.py) | 6 | Evaluate untrained transformer models |
| `run_finetune_step()` | [`src/steps/finetune_step.py`](src/steps/finetune_step.py) | 9 | Launch fine-training subprocess per model |
| `main()` | [`src/train.py`](src/train.py) | 87 | Training loop via HuggingFace Trainer |
| `compute_metrics()` | [`src/train.py`](src/train.py) | 34 | Metrics hook called by Trainer after each epoch |
| `run_eval_ft_step()` | [`src/steps/eval_ft_step.py`](src/steps/eval_ft_step.py) | 10 | Evaluate fine-tuned model checkpoints |
| `run_eval_ood_step()` | [`src/steps/eval_ood_step.py`](src/steps/eval_ood_step.py) | 182 | OOD evaluation on ToxiGen (all models) |
| `load_toxigen_dataset()` | [`src/steps/eval_ood_step.py`](src/steps/eval_ood_step.py) | 92 | Load and standardize ToxiGen labels |
| `extract_toxigen_identities_and_evaluate()` | [`src/steps/eval_ood_step.py`](src/steps/eval_ood_step.py) | 42 | Parse ToxiGen groups and compute bias metrics |
| `get_llama_toxicity_scores()` | [`src/steps/llama_step.py`](src/steps/llama_step.py) | 60 | Zero-shot scoring via Yes/No token probs |
| `run_llama_step()` | [`src/steps/llama_step.py`](src/steps/llama_step.py) | 118 | LLaMA evaluation (ID + OOD) |
| `run_analysis_step()` | [`src/analysis.py`](src/analysis.py) | 120 | Dataset stats and error sampling |
| `load_saved_data()` | [`src/steps/utils.py`](src/steps/utils.py) | 8 | Load Arrow datasets from disk |
| `get_transformer_predictions()` | [`src/steps/utils.py`](src/steps/utils.py) | 19 | Batch inference for transformer models |
| `eval_transformer()` | [`src/steps/utils.py`](src/steps/utils.py) | 48 | Full evaluate: predict + compute metrics |
| `evaluate_bias()` | [`src/evaluator.py`](src/evaluator.py) | 79 | Core metric computation for all models |
| `compute_subgroup_auc()` | [`src/evaluator.py`](src/evaluator.py) | 6 | ROC-AUC restricted to an identity subgroup |
| `compute_bpsn_auc()` | [`src/evaluator.py`](src/evaluator.py) | 21 | BPSN AUC (over-flagging detection) |
| `compute_bnsp_auc()` | [`src/evaluator.py`](src/evaluator.py) | 44 | BNSP AUC (under-flagging detection) |
| `compute_fnr()` | [`src/evaluator.py`](src/evaluator.py) | 67 | False Negative Rate |
| `compute_fpr()` | [`src/evaluator.py`](src/evaluator.py) | 73 | False Positive Rate |
| `run_report_step()` | [`src/steps/report_step.py`](src/steps/report_step.py) | 191 | Combine all CSVs into final comparison table |
| `format_final_report()` | [`src/steps/report_step.py`](src/steps/report_step.py) | 114 | Print 6-section comparison report to stdout |
| `log_results_to_csv()` | [`src/steps/report_step.py`](src/steps/report_step.py) | 27 | Append results to structured experiment log |
