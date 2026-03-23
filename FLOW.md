# Project Flow — End-to-End Pipeline Documentation

> **Small Transformer Toxicity Bias** is a research pipeline that measures **bias in small pre-trained transformers** using the [Jigsaw Unintended Bias in Toxicity Classification](https://huggingface.co/datasets/shuttie/jigsaw-unintended-bias) dataset. Models are evaluated both *before* and *after* fine-tuning across identity subgroups (race, gender, religion, etc.) using ROC-AUC, Subgroup-AUC, FNR, and FPR.

---

## Table of Contents

1. [High-Level Architecture](#1-high-level-architecture)
2. [Entry Point & CLI](#2-entry-point--cli)
3. [Step 1 — Data Loading & Preprocessing](#3-step-1--data-loading--preprocessing)
4. [Step 2 — Baseline (TF-IDF + Logistic Regression)](#4-step-2--baseline-tf-idf--logistic-regression)
5. [Step 3 — Raw Transformer Evaluation](#5-step-3--raw-transformer-evaluation)
6. [Step 4 — Fine-Tuning Transformers](#6-step-4--fine-tuning-transformers)
7. [Step 5 — Fine-Tuned Evaluation](#7-step-5--fine-tuned-evaluation)
8. [Step 6 — out-of-domain Evaluation](#8-step-6--out-of-domain-evaluation)
9. [Step 7 — LLaMA Zero-Shot Evaluation](#9-step-7--llama-zero-shot-evaluation)
10. [Step 8 — Report Generation](#10-step-8--report-generation)
11. [Directory Layout & Saved Artifacts](#11-directory-layout--saved-artifacts)
12. [Evaluation Metrics Deep-Dive](#12-evaluation-metrics-deep-dive)
13. [Understanding the Output](#13-understanding-the-output)
14. [Reproducibility](#14-reproducibility)

---

## 1. High-Level Architecture

```
main.py  (CLI entry point)
│
├── data           → src/steps/data_step.py       → src/data/dataset.py + src/data/data_loader.py
├── baseline       → src/steps/baseline_step.py   → src/evaluator.py
├── eval-raw       → src/steps/eval_raw_step.py   → src/steps/utils.py + src/evaluator.py
├── finetune       → src/steps/finetune_step.py   ──spawns──▶ python -m src.train
├── eval-finetuned → src/steps/eval_ft_step.py    → src/steps/utils.py + src/evaluator.py
├── eval-ood       → src/steps/eval_ood_step.py   → src/steps/utils.py + src/evaluator.py
├── llama          → src/steps/llama_step.py      → src/evaluator.py
└── report         → src/steps/report_step.py
```

Each step is **independently runnable** via `--step <name>` or all together via `--step all`.

---

## 2. Entry Point & CLI

**File:** [`main.py`](main.py) — `main()` function (line 8)

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
| `--step` | `all` | Which pipeline step to run (`data`, `baseline`, `eval-raw`, `finetune`, `eval-finetuned`, `eval-ood`, `llama`, `report`, `all`) |
| `--output_dir` | `./outputs` | Root directory for all caches, data, models, and results |
| `--models` | 3 small transformers | List of HuggingFace model identifiers to evaluate |
| `--llama_model` | `meta-llama/Llama-3.2-1B` | LLaMA model for zero-shot step |
| `--train_samples` | `20000` | Max training samples used for the baseline and fine-tuning steps. Pass `-1` to use the **full** training set. |
| `--eval_samples` | `5000` | Max evaluation samples used across all evaluation steps. Pass `-1` to evaluate on the **full** evaluation set. |
| `--seed` | `42` | Global random seed for reproducibility |

**Reproducibility setup** happens immediately in [`main.py` lines 29–32](main.py#L29-L32): `set_seed()`, `deterministic=True`, `benchmark=False`.

**Shared directories** are created at [`main.py` lines 35–41](main.py#L35-L41):

```
outputs/
├── .cache/          ← HuggingFace model + dataset cache
├── data/            ← Preprocessed & saved dataset splits
└── results/         ← Per-step CSV metric files
```

---

## 3. Step 1 — Data Loading & Preprocessing

**Step file:** [`src/steps/data_step.py`](src/steps/data_step.py) — `run_data_step()` (line 7)

### 3.1 Downloading the Raw Dataset

**File:** [`src/data/data_loader.py`](src/data/data_loader.py) — `get_jigsaw_dataset()` (line 11)

The dataset source is a community HuggingFace mirror of Jigsaw (`shuttie/jigsaw-unintended-bias`) loaded as raw CSV:

```
hf://datasets/shuttie/jigsaw-unintended-bias/data/train.csv.gz
hf://datasets/shuttie/jigsaw-unintended-bias/data/test_private_expanded.csv.gz
```

- **Cache location:** `<output_dir>/.cache/` (or cluster path `/vol/joberant_nobck/data/NLP_368307701_2526a/<user>/.cache/huggingface` when detected)
- **Authentication:** HuggingFace token fetched via [`src/data/data_utils.py`](src/data/data_utils.py) `get_hf_token()` (line 27), which checks `HF_TOKEN` env variable (`.env` file) or the local HF token cache.

### 3.2 Preprocessing

**File:** [`src/data/dataset.py`](src/data/dataset.py) — `download_and_prep_jigsaw()` (line 14)

After loading, the dataset is processed using HuggingFace's Arrow memory-mapped backend (no full load into RAM):

1. **Toxicity binarization** (line 34): `is_toxic = int(target >= 0.5)` — continuous toxicity scores above 0.5 are labelled as toxic.
2. **Text cleaning** (line 37): `None` values in `comment_text` are replaced with `""`.
3. **Identity columns kept as continuous** (lines 41–42): All 24 identity columns (e.g., `asian`, `muslim`, `transgender`) are kept as float values in [0, 1] — **they are NOT binarized** — so that subgroup membership can be graded.

The 24 identity columns are defined in [`src/data/dataset.py` line 6](src/data/dataset.py#L6) (`ALL_IDENTITY_COLUMNS`).

**Which columns are kept:**
```
['id', 'comment_text', 'target', 'is_toxic', <identity columns>]
```

### 3.3 Train/Validation Split & Saving

Back in [`src/steps/data_step.py`](src/steps/data_step.py):

1. The full dataset is **shuffled** with a fixed `SEED=42` (line 5, 10).
2. **90/10 split**: first 90% → training set, last 10% → validation set (lines 12–14).
3. Slices are taken per CLI arguments (lines 17–22):
   - `baseline_train` ← up to `--train_samples` from the training 90%
   - `eval` ← up to `--eval_samples` from the validation 10%

**Saved to disk** (lines 24–29):
```
outputs/data/
├── baseline_train/       ← Arrow dataset directory (HF Dataset format)
├── eval/                 ← Arrow dataset directory (HF Dataset format)
└── identity_columns.json ← List of identity column names found in the dataset
```

These Arrow datasets are reused by all subsequent steps without re-downloading.

---

## 4. Step 2 — Baseline (TF-IDF + Logistic Regression)

**Step file:** [`src/steps/baseline_step.py`](src/steps/baseline_step.py) — `run_baseline_step()` (line 9)

### Data Loading

Uses [`src/steps/utils.py`](src/steps/utils.py) `load_saved_data()` (line 8) to load the Arrow datasets from `outputs/data/` from disk — no network call needed.

### Model

A scikit-learn `Pipeline` with two stages (lines 24–26):

```python
Pipeline([
    ('tfidf', TfidfVectorizer(max_features=10000, stop_words='english')),
    ('clf', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42))
])
```

- Trained on `baseline_train` split (up to `--train_samples`).
- Evaluated on `eval` split.
- **The model is NOT saved to disk** in this step; it is trained and evaluated in-memory within the same function call.

> **Note:** `src/baseline.py` `train_baseline()` (line 10) is an older standalone script that saves the model via `joblib`. The step pipeline uses `run_baseline_step()` instead, which skips saving.

### Output

Results saved to: `outputs/results/baseline_metrics.csv`

---

## 5. Step 3 — Raw Transformer Evaluation

**Step file:** [`src/steps/eval_raw_step.py`](src/steps/eval_raw_step.py) — `run_eval_raw_step()` (line 6)

For each model in `--models`:

1. **Tokenizer** loaded from HuggingFace with `cache_dir=outputs/.cache/` (line 12).
2. **Model** loaded with `num_labels=2` (binary toxicity classification) from HuggingFace cache (line 15–16).
   - This is the **pre-trained model with untrained classification head** — no fine-tuning.
3. **Inference** run via [`src/steps/utils.py`](src/steps/utils.py) `get_transformer_predictions()` (line 19) then `eval_transformer()` (line 48) in batches of 32.

### Output

One CSV per model in `outputs/results/`:
```
outputs/results/distilbert-base-uncased_raw_metrics.csv
outputs/results/distilroberta-base_raw_metrics.csv
outputs/results/google_bert_uncased_L-4_H-512_A-8_raw_metrics.csv
```

(`/` in model names replaced by `_`)

---

## 6. Step 4 — Fine-Tuning Transformers

**Step file:** [`src/steps/finetune_step.py`](src/steps/finetune_step.py) — `run_finetune_step()` (line 6)

This step does **NOT** run training inline — it **spawns a subprocess** per model (lines 14–22):

```bash
python -m src.train \
  --model_name <model_name> \
  --output_base_dir outputs/finetuned_<safe_name>/ \
  --epochs 1 \
  --batch_size 32 \
  --seed 42 \
  --train_samples <train_samples>
```

**Skip logic** (line 13): If `outputs/finetuned_<safe_name>/small-transformer-toxicity/config.json` already exists, fine-tuning is **skipped** — the existing checkpoint is reused.

### The Training Script: `src/train.py`

**File:** [`src/train.py`](src/train.py) — `main()` (line 80)

#### Data Loading inside train.py

- Calls `download_and_prep_jigsaw("train", cache_dir=...)` (line 101) — downloads/loads from HuggingFace cache.
- Independently shuffles and splits **90/10** on its own (lines 106–115), independent of the saved data splits in `outputs/data/`.
  - If `--train_samples` is provided (and > 0), the dataset is truncated **before** splitting (lines 109–110). Pass `-1` to use the full training set.
- Tokenizes via [`src/data/dataset.py`](src/data/dataset.py) `tokenize_jigsaw_dataset()`.
  - Pads/truncates to `max_length=128`.
  - Uses HF Arrow memory-mapped `.map()` — no full load into RAM.
- Wraps tokenized datasets in `JigsawDataset` (lines 122–123):
  - `JigsawDataset.__init__()` pre-calculates `identity_matrix` as a NumPy array of shape `(N, num_identities)` — this avoids repeated per-step lookups during evaluation.

#### Model Loading

Line 121–126: `AutoModelForSequenceClassification.from_pretrained()` with `num_labels=2` and `cache_dir`.

#### Training Arguments

Lines 129–145 — key settings:

| Argument | Value | Reason |
|---|---|---|
| `eval_strategy` | `"epoch"` | Evaluate after each epoch |
| `save_strategy` | `"epoch"` | Save checkpoint after each epoch |
| `save_total_limit` | `2` | Keep only the last 2 checkpoints (storage safety) |
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

At the end of each epoch, the HuggingFace `Trainer` calls `compute_metrics_wrapper()` (line 155) → `compute_metrics()` (line 29):

1. Extracts predicted probabilities from logits (softmax for 2-class, sigmoid for 1-class).
2. Retrieves `identity_matrix` from `val_dataset.identity_matrix` (pre-calculated in `JigsawDataset.__init__()`).
3. Calls `evaluate_models_metrics()` (alias for `evaluate_bias()`) from [`src/evaluator.py`](src/evaluator.py).
4. Returns a dict: `{"roc_auc": <value>, "<identity>_subgroup_auc": ..., "<identity>_subgroup_fnr": ..., "<identity>_subgroup_fpr": ...}` — logged by the Trainer.

---

## 7. Step 5 — Fine-Tuned Evaluation

**Step file:** [`src/steps/eval_ft_step.py`](src/steps/eval_ft_step.py) — `run_eval_ft_step()` (line 6)

For each model:

1. Looks for the fine-tuned model at `outputs/finetuned_<safe_name>/small-transformer-toxicity/` (line 12).
2. Prefers the root directory (if `config.json` exists there = best model saved by Trainer).
3. Falls back to the **latest checkpoint** (by step number) if the root doesn't have `config.json` (lines 19–22).
4. Loads the tokenizer from the **original model name** (not the fine-tuned path) and the model weights from the fine-tuned path (lines 25–31).
5. Runs inference + evaluation via `eval_transformer()` in `src/steps/utils.py`.

### Output

```
outputs/results/distilbert-base-uncased_finetuned_metrics.csv
outputs/results/distilroberta-base_finetuned_metrics.csv
outputs/results/google_bert_uncased_L-4_H-512_A-8_finetuned_metrics.csv
```

---

## 8. Step 6 — out-of-domain Evaluation

**Step file:** [`src/steps/eval_ood_step.py`](src/steps/eval_ood_step.py) — `run_eval_ood_step()`

Evaluates fine-tuned models on an **out-of-domain** (OOD) test set to measure generalization beyond the training distribution. Uses the same evaluation infrastructure as `eval-finetuned` (`src/steps/utils.py`), but loads a separate held-out dataset (ToxiGen).

- Accepts `--eval_samples` to limit the OOD evaluation set size. Pass `-1` to evaluate on the **full** OOD set.
- Results saved to: `outputs/results/<model>_ood_metrics.csv`

> **Note:** This step is not included in `--step all`; it must be run explicitly with `--step eval-ood`.

---

## 9. Step 7 — LLaMA Zero-Shot Evaluation

**Step file:** [`src/steps/llama_step.py`](src/steps/llama_step.py) — `run_llama_step()` (line 41)

Evaluates a **causal language model** (e.g., `meta-llama/Llama-3.2-1B`) in a **zero-shot** fashion — no fine-tuning.

### Scoring Method

**Function:** `get_llama_toxicity_scores()` (line 9)

Each comment is wrapped in the following prompt:

```
Comment: "<comment_text>"
Is this comment toxic? Answer:
```

The model generates logits at the final token position. The toxicity score is:

```
P(Yes) / (P(Yes) + P(No))
```

where `Yes` and `No` token IDs are found via `tokenizer.encode()` (lines 18–19). This gives a continuous score in [0, 1] without the model needing to generate text.

- Runs in batches of 8 to manage GPU memory.
- Uses `float16` precision on CUDA, `float32` otherwise.

### Output

```
outputs/results/meta-llama_Llama-3.2-1B_raw_metrics.csv
```

---

## 10. Step 8 — Report Generation

**Step file:** [`src/steps/report_step.py`](src/steps/report_step.py) — `run_report_step()` (line 66)

### What it does

1. Scans `outputs/results/` for all `*.csv` files.
2. Maps filenames back to display names:
   - `baseline_metrics.csv` → `"Baseline"`
   - `<safe_name>_raw_metrics.csv` → `"<model> Raw"`
   - `<safe_name>_finetuned_metrics.csv` → `"<model> Finetuned"`
3. Merges all DataFrames on `"Identity"` column into a single wide comparison table.
4. Prints 4 separate comparison sections to stdout via `format_final_report()` (line 4):
   - **Overall AUC Comparison**
   - **Subgroup AUC Comparison**
   - **FNR Comparison**
   - **FPR Comparison**

### Output

```
outputs/results/final_report.csv   ← Wide CSV with all models × all metrics
```

---

## 11. Directory Layout & Saved Artifacts

```
outputs/                                      ← --output_dir
│
├── .cache/                                   ← HuggingFace Downloads Cache
│   └── (model weights, tokenizers, datasets — Arrow format)
│
├── data/                                     ← Preprocessed Dataset Splits
│   ├── baseline_train/                       ← Arrow dataset (20k rows by default)
│   ├── eval/                                 ← Arrow dataset (5k rows by default)
│   └── identity_columns.json                 ← ["asian", "muslim", ...]
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
└── results/                                  ← Metric CSVs
    ├── baseline_metrics.csv
    ├── distilbert-base-uncased_raw_metrics.csv
    ├── distilbert-base-uncased_finetuned_metrics.csv
    ├── distilroberta-base_raw_metrics.csv
    ├── distilroberta-base_finetuned_metrics.csv
    ├── google_bert_uncased_L-4_H-512_A-8_raw_metrics.csv
    ├── google_bert_uncased_L-4_H-512_A-8_finetuned_metrics.csv
    ├── meta-llama_Llama-3.2-1B_raw_metrics.csv
    └── final_report.csv                      ← Combined comparison table
```

---

## 12. Evaluation Metrics Deep-Dive

All metric computation goes through [`src/evaluator.py`](src/evaluator.py) — the core function is `evaluate_bias()`, aliased as `evaluate_models_metrics`.

### Inputs

| Parameter | Shape | Description |
|---|---|---|
| `y_true` | `(N,)` | Binary ground-truth toxicity labels |
| `y_pred_probs` | `(N,)` | Predicted probability of being toxic |
| `identity_matrix` | `(N, K)` | Continuous identity scores per example |
| `identity_columns` | `list[str]` | Names for each of the K identity columns |
| `threshold` | `float` | Binarization threshold for FNR/FPR (default 0.5) |

### Subgroup Membership

An example is considered **part of a subgroup** if its identity score > 0.5 (line 64). This uses the continuous annotation values directly from the dataset.

### Metrics Computed

For each identity (one row per identity in the output DataFrame):

| Column | Function | Description |
|---|---|---|
| `1. Overall AUC` | `roc_auc_score(y_true, y_pred_probs)` | ROC-AUC on the entire eval set |
| `2. Overall FNR` | `compute_fnr()` (line 22) | FN / (FN + TP) on the entire eval set |
| `3. Overall FPR` | `compute_fpr()` (line 29) | FP / (FP + TN) on the entire eval set |
| `4. Subgroup AUC` | `compute_subgroup_auc()` (line 8) | ROC-AUC restricted to this identity subgroup |
| `5. Subgroup FNR` | `compute_fnr()` on subgroup mask | FNR restricted to this identity subgroup |
| `6. Subgroup FPR` | `compute_fpr()` on subgroup mask | FPR restricted to this identity subgroup |

> **Note:** Overall AUC/FNR/FPR values are the *same* on every row — they describe the whole dataset and are repeated for easy CSV export and comparison.

### Edge Cases

- If a subgroup has fewer than 2 unique label classes (e.g., all examples are non-toxic), the AUC returns `NaN` (line 17–18 in `compute_subgroup_auc`).
- Similarly, `compute_fnr` and `compute_fpr` return `NaN` if the denominator is zero (lines 27, 34).

---

## 13. Understanding the Output

### Per-Step CSVs (`outputs/results/*.csv`)

Each file has `K` rows (one per identity group) and the following columns:

```
Identity | Total Examples | 1. Overall AUC | 2. Overall FNR | 3. Overall FPR | 4. Subgroup AUC | 5. Subgroup FNR | 6. Subgroup FPR
```

Example interpretation:

```
Identity=muslim, Subgroup AUC=0.61 vs Overall AUC=0.85
→ The model is significantly worse at distinguishing toxic vs. non-toxic content
  when the comment mentions Muslim identity.

Identity=muslim, Subgroup FNR=0.40 vs Overall FNR=0.20
→ The model misses 40% of actual toxic comments mentioning Muslim identity,
  vs only 20% overall — indicating bias (higher false negative rate for this group).

Identity=muslim, Subgroup FPR=0.10 vs Overall FPR=0.05
→ The model incorrectly flags 10% of non-toxic Muslim-identity comments as toxic,
  vs 5% overall — indicating the model may produce more false alarms on this group.
```

### Final Report (`outputs/results/final_report.csv`)

A wide comparison table where each model's metrics are shown side-by-side per identity. Printed to stdout in 4 sections:

```
1. Overall AUC Comparison
2. Subgroup AUC Comparison
3. FNR Comparison
4. FPR Comparison
```

This is the primary artifact for the research paper/report, showing how bias changes between:
- **Baseline** (TF-IDF + LogReg)
- **Raw** pre-trained transformers (zero-shot classification)
- **Fine-Tuned** transformers (after supervised training)
- **LLaMA** zero-shot via next-token probability

---

## 14. Reproducibility

The pipeline uses multiple layered mechanisms to ensure reproducibility:

| Mechanism | Location | Detail |
|---|---|---|
| Global seed | [`main.py` line 27](main.py#L27) | `set_seed(args.seed)` from `transformers` |
| CuDNN determinism | [`main.py` lines 28–30](main.py#L28-L30) | `deterministic=True`, `benchmark=False` |
| Dataset shuffle seed | [`src/steps/data_step.py`](src/steps/data_step.py) | Fixed `SEED=42` for the 90/10 split |
| Trainer seed | [`src/train.py` line 147](src/train.py#L147) | `seed=args.seed` passed to `TrainingArguments` |
| Baseline seed | [`src/steps/baseline_step.py`](src/steps/baseline_step.py) | `random_state=42` in `LogisticRegression` |
| Checkpoint skip | [`src/steps/finetune_step.py` line 13](src/steps/finetune_step.py#L13) | If fine-tuned model exists, skip re-training |

---

## Appendix: Key Function Reference

| Function | File | Line | Purpose |
|---|---|---|---|
| `main()` | [`main.py`](main.py) | 8 | CLI entry point, orchestrates all steps |
| `run_data_step()` | [`src/steps/data_step.py`](src/steps/data_step.py) | 7 | Load, split, and save dataset to disk |
| `get_jigsaw_dataset()` | [`src/data/data_loader.py`](src/data/data_loader.py) | 11 | Download dataset from HuggingFace mirror |
| `get_hf_token()` | [`src/data/data_utils.py`](src/data/data_utils.py) | 27 | Fetch HuggingFace API token |
| `download_and_prep_jigsaw()` | [`src/data/dataset.py`](src/data/dataset.py) | 14 | Binarize toxicity, clean identities |
| `tokenize_jigsaw_dataset()` | [`src/data/dataset.py`](src/data/dataset.py) | 59 | Tokenize text with HF tokenizer |
| `JigsawDataset` | [`src/data/dataset.py`](src/data/dataset.py) | 84 | PyTorch Dataset wrapper; pre-caches identity_matrix |
| `run_baseline_step()` | [`src/steps/baseline_step.py`](src/steps/baseline_step.py) | 9 | Train and evaluate TF-IDF + LogReg |
| `run_eval_raw_step()` | [`src/steps/eval_raw_step.py`](src/steps/eval_raw_step.py) | 6 | Evaluate untrained transformer models |
| `run_finetune_step()` | [`src/steps/finetune_step.py`](src/steps/finetune_step.py) | 5 | Launch fine-training subprocess per model |
| `main()` | [`src/train.py`](src/train.py) | 80 | Training loop via HuggingFace Trainer |
| `compute_metrics()` | [`src/train.py`](src/train.py) | 29 | Metrics hook called by Trainer after each epoch |
| `run_eval_ft_step()` | [`src/steps/eval_ft_step.py`](src/steps/eval_ft_step.py) | 6 | Evaluate fine-tuned model checkpoints |
| `get_llama_toxicity_scores()` | [`src/steps/llama_step.py`](src/steps/llama_step.py) | 9 | Zero-shot scoring via Yes/No token probs |
| `run_llama_step()` | [`src/steps/llama_step.py`](src/steps/llama_step.py) | 41 | Orchestrate LLaMA evaluation |
| `load_saved_data()` | [`src/steps/utils.py`](src/steps/utils.py) | 8 | Load Arrow datasets from disk |
| `get_transformer_predictions()` | [`src/steps/utils.py`](src/steps/utils.py) | 19 | Batch inference for transformer models |
| `eval_transformer()` | [`src/steps/utils.py`](src/steps/utils.py) | 48 | Full evaluate: predict + compute metrics |
| `evaluate_bias()` | [`src/evaluator.py`](src/evaluator.py) | 37 | Core metric computation for all models |
| `compute_subgroup_auc()` | [`src/evaluator.py`](src/evaluator.py) | 8 | ROC-AUC restricted to an identity subgroup |
| `compute_fnr()` | [`src/evaluator.py`](src/evaluator.py) | 22 | False Negative Rate |
| `compute_fpr()` | [`src/evaluator.py`](src/evaluator.py) | 29 | False Positive Rate |
| `run_report_step()` | [`src/steps/report_step.py`](src/steps/report_step.py) | 66 | Combine all CSVs into final comparison table |
| `format_final_report()` | [`src/steps/report_step.py`](src/steps/report_step.py) | 4 | Print 4-section comparison report to stdout |
| `get_model_pair()` | [`src/model/model_manager.py`](src/model/model_manager.py) | 5 | Load raw + placeholder fine-tuned model pair |
| `train_model()` | [`src/model/model_manager.py`](src/model/model_manager.py) | 26 | Utility wrapper around HuggingFace Trainer |
