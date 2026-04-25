import logging
import os
import joblib
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from src.data.data_utils import get_hf_token
from src.evaluator import evaluate_bias

logger = logging.getLogger("pipeline")

def eval_transformer_ood(model_name, model, tokenizer, df, device):
    """Evaluates a transformer model on an Out-Of-Domain dataset (ToxiGen)."""
    logger.info(f"Tokenizing OOD data for {model_name}...")
    texts = df["text"].tolist()
    labels = df["label"].tolist()

    batch_size = 32
    all_preds = []

    model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc=f"OOD batches [{model_name}]"):
            batch_texts = texts[i : i + batch_size]
            encoded = tokenizer(batch_texts, padding=True, truncation=True, max_length=256, return_tensors="pt")
            encoded = {k: v.to(device) for k, v in encoded.items()}

            outputs = model(**encoded)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=-1)

            batch_preds = probs[:, 1].cpu().numpy()
            all_preds.extend(batch_preds)

    df["toxicity_score"] = all_preds
    df["model"] = model_name
    return df

def extract_toxigen_identities_and_evaluate(model_name, df_with_preds):
    """Helper to extract ToxiGen subgroups and compute bias metrics."""
    possible_group_cols = ["target_groups", "target_group", "group"]
    found_group_col = next((c for c in possible_group_cols if c in df_with_preds.columns), None)

    identity_cols = []
    identity_matrix_data = {}

    if found_group_col:
        import ast

        for i, row in df_with_preds.iterrows():
            group_val = row[found_group_col]
            if isinstance(group_val, str):
                if group_val.startswith("[") and group_val.endswith("]"):
                    try:
                        groups = ast.literal_eval(group_val)
                    except (ValueError, SyntaxError):
                        groups = [group_val]
                else:
                    groups = [g.strip() for g in group_val.split(",")]
            elif isinstance(group_val, list):
                groups = group_val
            else:
                groups = [str(group_val)]

            for g in groups:
                if g and g.lower() not in ["none", "nan", "null", "unknown"]:
                    if g not in identity_cols:
                        identity_cols.append(g)
                        identity_matrix_data[g] = np.zeros(len(df_with_preds))
                    identity_matrix_data[g][i] = 1.0

    if not identity_cols:
        logger.warning(f"Could not parse identity groups from column '{found_group_col}'. Only overall metrics will be calculated.")
        identity_cols = ["placeholder"]
        identity_matrix = np.zeros((len(df_with_preds), 1))
    else:
        identity_matrix = np.column_stack([identity_matrix_data[g] for g in identity_cols])

    y_true = df_with_preds["label"].to_numpy()
    y_preds = df_with_preds["toxicity_score"].to_numpy()

    metrics_df = evaluate_bias(
        y_true=y_true, y_pred_probs=y_preds, identity_matrix=identity_matrix, identity_columns=identity_cols
    )

    metrics_df.insert(0, "Model", model_name)
    return metrics_df

def load_toxigen_dataset(cache_dir, eval_samples=-1, seed=42):
    """Loads and standardizes labels for the ToxiGen dataset."""
    logger.info("Loading ToxiGen dataset from Hugging Face...")
    try:
        toxigen = load_dataset(
            "skg/toxigen-data", name="train", cache_dir=cache_dir, split="train+test", token=get_hf_token()
        )
    except Exception as e:
        logger.warning(f"Could not load skg/toxigen-data: {e}")
        logger.info("Attempting to load standard 'toxigen/toxigen-data'...")
        try:
            toxigen = load_dataset(
                "toxigen/toxigen-data", name="annotated", cache_dir=cache_dir, split="train+test", token=get_hf_token()
            )
        except Exception as e2:
            logger.error(f"Could not load toxigen/toxigen-data: {e2}", exc_info=True)
            raise e2

    df = toxigen.to_pandas()
    
    # Standardize label column
    if 'toxicity_human' in df.columns:
        df['label'] = (df['toxicity_human'] > 3).astype(int) 
    elif 'toxicity_human_annotated' in df.columns:
        df['label'] = df['toxicity_human_annotated']
    elif 'label' in df.columns:
         pass 
    elif 'toxicity' in df.columns:
         df['label'] = df['toxicity']
    elif 'toxicity_score' in df.columns:
         df['label'] = df['toxicity_score'].apply(lambda x: 1 if x >= 0.5 else 0)
    else:
        raise ValueError(
            f"Could not extract binary labels from ToxiGen dataset. "
            f"Available columns: {list(df.columns)}"
        )

    if eval_samples > 0:
        if len(df) > eval_samples:
            df = df.sample(n=eval_samples, random_state=seed).reset_index(drop=True)
            
    # Standardize input text column globally 
    if 'text' not in df.columns and 'generation' in df.columns:
        df['text'] = df['generation']
    elif 'text' not in df.columns and 'comment_text' in df.columns:
         df['text'] = df['comment_text']
            
    logger.info(f"Loaded {len(df)} samples from ToxiGen for OOD evaluation.")
    return df

def eval_baseline_ood(results_dir, df):
    """Evaluates saved baselines (Logistic Regression and Naive) on ToxiGen."""
    baselines = [
        ("Baseline (TF-IDF + LR)", "baseline_pipeline.joblib", "preds_Baseline_ood.csv"),
        ("Naive (Majority Vote)", "naive_baseline.joblib", "preds_Naive_ood.csv")
    ]
    
    baseline_metrics_list = []

    for model_name, filename, preds_filename in baselines:
        baseline_path = os.path.join(results_dir, filename)
        if not os.path.exists(baseline_path):
            logger.warning(f"Skipping {model_name}: could not find {baseline_path}")
            continue

        logger.info(f"Evaluating {model_name} on OOD data...")
        try:
            model = joblib.load(baseline_path)
            inference_text = df["text"]
            
            X_val = [str(t) if t is not None else "" for t in inference_text]
            y_pred_probs = model.predict_proba(X_val)[:, 1]

            df_with_preds = df.copy()
            df_with_preds['toxicity_score'] = y_pred_probs
            df_with_preds['label'] = df['label']
            
            metrics_df = extract_toxigen_identities_and_evaluate(model_name, df_with_preds)
            baseline_metrics_list.append(metrics_df)
            
            preds_df = df_with_preds[['text', 'toxicity_score']]
            preds_out_path = os.path.join(results_dir, preds_filename)
            preds_df.to_csv(preds_out_path, index=False)
            logger.info(f"Saved {model_name} OOD predictions to {preds_out_path}")
            
        except Exception as e:
            logger.error(f"Error evaluating {model_name} on OOD data: {e}", exc_info=True)
            
    return baseline_metrics_list if baseline_metrics_list else None

def run_eval_ood_step(results_dir, cache_dir, output_dir, models, device, eval_samples=-1, seed=42):
    logger.info("Running OOD Evaluation (ToxiGen)...")
    df = load_toxigen_dataset(cache_dir, eval_samples, seed=seed)
    
    all_metrics = []
    summary_results = []

    # 1. Evaluate Baseline Models (ML & Naive) if they exist
    baseline_metrics_list = eval_baseline_ood(results_dir, df)
    if baseline_metrics_list is not None:
        summary_results.extend(baseline_metrics_list)
        
    # Save the standardized ToxiGen dataset for reuse (e.g., LLaMA step)
    toxigen_save_path = os.path.join(output_dir, "data", "toxigen_standardized.parquet")
    df.to_parquet(toxigen_save_path, index=False)
    logger.info(f"Saved standardized ToxiGen dataset for LLaMA reuse to {toxigen_save_path}")

    # 2. Evaluate Transformer Models
    for base_model_name in tqdm(models, desc="OOD eval models"):
        safe_name = base_model_name.replace("/", "_")
        model_output_base_dir = os.path.join(output_dir, f"finetuned_{safe_name}")
        finetuned_model_dir = os.path.join(model_output_base_dir, "small-transformer-toxicity")

        logger.info(f"Evaluating Fine-Tuned Transformer ({base_model_name}) on OOD data...")
        model_load_path = finetuned_model_dir
        if os.path.exists(finetuned_model_dir):
            if not os.path.exists(os.path.join(finetuned_model_dir, "config.json")):
                checkpoints = [d for d in os.listdir(finetuned_model_dir) if d.startswith("checkpoint-")]
                if checkpoints:
                    checkpoints.sort(key=lambda x: int(x.split("-")[-1]))
                    model_load_path = os.path.join(finetuned_model_dir, checkpoints[-1])

            try:
                tokenizer = AutoTokenizer.from_pretrained(base_model_name, cache_dir=cache_dir)
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token

                ft_model = AutoModelForSequenceClassification.from_pretrained(
                    model_load_path, num_labels=2, cache_dir=cache_dir
                )
                ft_model.to(device)
                
                df_with_preds = eval_transformer_ood(f"Fine-Tuned {base_model_name}", ft_model, tokenizer, df.copy(), device)
                metrics_df = extract_toxigen_identities_and_evaluate(base_model_name, df_with_preds)
                summary_results.append(metrics_df)
                
                preds_df = df_with_preds[['text', 'toxicity_score']]
                preds_out_path = os.path.join(results_dir, f"preds_{safe_name}_finetuned_ood.csv")
                preds_df.to_csv(preds_out_path, index=False)
                logger.info(f"Saved Finetuned {base_model_name} OOD predictions to {preds_out_path}")
                
            except Exception as e:
                logger.error(f"Error evaluating fine-tuned model {base_model_name} on OOD data: {e}", exc_info=True)
        else:
            logger.warning(f"Could not find fine-tuned directory: {finetuned_model_dir}. Please run 'make finetune' first.")

    if summary_results:
        summary_df = pd.concat(summary_results, ignore_index=True)
        out_path = os.path.join(results_dir, "ood_toxigen_metrics.csv")
        summary_df.to_csv(out_path, index=False)
        logger.info(f"Saved detailed OOD metrics to {out_path}")
