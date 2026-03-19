# Harmful Text Detection and Group Bias in Small Transformer Models

## Participants
* **Maxim German** (322542887) - maximgerman1@mail.tau.ac.il
* **Eran Shufaro** (209074731) - shufaru@mail.tau.ac.il
* **Ilay Abramovich** (322271032) - ilaya@mail.tau.ac.il
* **Itay Hazan** (209277367) - itayhazan@mail.tau.ac.il

## Project Description
This project investigates the effectiveness of small pre-trained transformer encoder models (BERT family) when fine-tuned for harmful and toxic text detection. We specifically focus on identifying unintended bias across identity groups—such as gender, religion, and sexual orientation—within these models.

While Large Language Models (LLMs) are currently prominent, many deployed moderation systems still rely on smaller transformer-based classifiers due to their stability and ease of deployment. Prior research has shown that these classifiers can exhibit biased behavior toward specific groups even when the text itself is non-toxic.

## Methodology
* **Models**: Fine-tuning small encoder-based transformer models (BERT variants).
* **Dataset**: Google Jigsaw Unintended Bias in Toxicity Classification dataset.
* **Evaluation**: Performance is measured using ROC-AUC, subgroup AUC, and False Negative Rates (FNR) to identify potential bias gaps.

## How to Run

The pipeline has been designed to support both all-in-one execution and step-by-step execution. Step-by-step execution is highly recommended when running heavy jobs to permit easier debugging and to prevent data loss if a later step (e.g., LLaMA inference) runs out of memory.

All intermediate datasets, models, and results are saved to the `--output_dir` (defaults to `./outputs`), allowing subsequent steps to load them from disk.

### Running Step-by-Step

```bash
# 1. Download, shuffle, split, and cache the datasets
python main.py --step data

# 2. Train and evaluate the TF-IDF + Logistic Regression baseline
python main.py --step baseline

# 3. Evaluate the raw (pre-trained, non-finetuned) Transformer models
python main.py --step eval-raw

# 4. Trigger fine-tuning jobs for the Transformer models
python main.py --step finetune

# 5. Evaluate the newly fine-tuned Transformer models
python main.py --step eval-finetuned

# 6. Evaluate the LLaMA model (requires High-VRAM GPU)
python main.py --step llama

# 7. Aggregate all saved metrics from the above steps and generate final report
python main.py --step report
```

### Running All at Once
To run the entire pipeline end-to-end sequentially:
```bash
python main.py --step all
```
*(Note: even when running `all`, intermediate results are still persisted to disk).*

### University Cluster Usage
When running on the SLURM cluster, make sure to point the output directory to the persistent storage to avoid filling up the limited home directory:

```bash
python main.py --step all --output_dir /vol/joberant_nobck/data/NLP_368307701_2526a/<YOUR_USER_NAME>
```

## Requirements
* **Environment**: Hugging Face Transformers and Datasets ecosystem.
* **Resources**: University-provided computational resources.
* **Access**: All data, weights, and APIs used are publicly accessible.

## AI Usage Disclosure
A generative AI tool (ChatGPT) was used to assist with brainstorming and refining the academic phrasing and structure of the project proposal to ensure conciseness and clarity.

## References
1. Dixon et al. "Measuring and Mitigating Unintended Bias in Text Classification." (2018).
2. Xie et al. "SORRY-Bench: Systematically Evaluating Large Language Model Safety Refusal Behaviors." (2025).