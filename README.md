# Harmful Text Detection and Group Bias in Small Transformer Models

## Participants
* **Maxim German** (322542887) - maximgerman1@mail.tau.ac.il
* **Eran Shufaro** (209074731) - shufaru@mail.tau.ac.il
* **Ilay Abramovich** (322271032) - ilaya@mail.tau.ac.il
* **Itay Hazan** (209277367) - itayhazan@mail.tau.ac.il

## Project Description
This project investigates the effectiveness of small pre-trained transformer encoder models (BERT family) when fine-tuned for harmful and toxic text detection. We specifically focus on identifying unintended bias across identity groups -- such as gender, religion, and sexual orientation -- within these models.

While Large Language Models (LLMs) are currently prominent, many deployed moderation systems still rely on smaller transformer-based classifiers due to their stability and ease of deployment. Prior research has shown that these classifiers can exhibit biased behavior toward specific groups even when the text itself is non-toxic.

## Methodology
* **Models**: Naive baseline (majority vote), TF-IDF + Logistic Regression baseline, fine-tuned small encoder-based transformer models (BERT variants), and LLaMA zero-shot evaluation.
* **Datasets**: Google Jigsaw Unintended Bias in Toxicity Classification (in-distribution) and ToxiGen (out-of-domain).
* **Evaluation**: ROC-AUC, Subgroup AUC, BPSN AUC, BNSP AUC, False Negative Rates (FNR), and False Positive Rates (FPR) across identity subgroups.

## Setup

```bash
pip install -r requirements.txt
```

For GPU-accelerated training on the TAU SLURM cluster, install the CUDA-compatible PyTorch build first:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

## Quick Start

Run the full pipeline end-to-end:
```bash
make run-all
```

Or run each step individually:
```bash
make data             # Download, shuffle, split, and cache the datasets
make baseline         # Train and evaluate baselines (TF-IDF + LR, Naive)
make finetune         # Fine-tune transformer models
make eval-finetuned   # Evaluate fine-tuned transformers on Jigsaw
make eval-ood         # Evaluate all models on ToxiGen (out-of-domain)
make llama            # LLaMA zero-shot evaluation (ID + OOD)
make analysis         # Dataset statistics and error sampling
make report           # Aggregate metrics into a final comparison report
```

Additional targets:
```bash
make eval-raw         # Evaluate raw (pre-trained, non-finetuned) transformers
make sanity-check     # Overfit on 100 samples to validate training correctness
make tiny-run         # Quick CPU-friendly validation run (100 samples)
make run-scientific   # 20k train samples, full evaluation set
make run-full         # Full 1.8M dataset (Warning: takes several days)
make test             # Run pytest suite
make lint             # Lint with ruff
```

## CLI Flags

| Flag | Default | Description |
|---|---|---|
| `--step` | `all` | Pipeline step: `data`, `baseline`, `eval-raw`, `finetune`, `eval-finetuned`, `eval-ood`, `llama`, `analysis`, `report`, or `all`. |
| `--output_dir` | `./outputs` | Root directory for caches, data, models, and results. |
| `--models` | 3 small transformers | Space-separated list of HuggingFace model identifiers. |
| `--llama_model` | `meta-llama/Llama-3.2-1B` | LLaMA model for zero-shot evaluation. |
| `--llama_batch_size` | `32` | Batch size for LLaMA inference. |
| `--train_samples` | `20000` | Max training samples (`-1` for full dataset). |
| `--eval_samples` | `5000` | Max evaluation samples (`-1` for full dataset). |
| `--seed` | `42` | Global random seed. |

## SLURM Cluster Usage
```bash
python main.py --step all --output_dir /vol/joberant_nobck/data/NLP_368307701_2526a/<YOUR_USER_NAME>
```

## Citation

```bibtex
@software{German_Small-Transformer-Toxicity-Bias_2026,
  author = {German, Maxim and Shufaro, Eran and Abramovich, Ilay and Hazan, Itay},
  month = {4},
  title = {{Small-Transformer-Toxicity-Bias}},
  url = {https://github.com/cs-tau-projects/Small-Transformer-Toxicity-Bias},
  version = {1.0.0},
  year = {2026}
}
```