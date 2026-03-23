import os
import pandas as pd
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from src.data.data_utils import get_hf_token
import numpy as np
from src.evaluator import evaluate_bias
import joblib

def eval_transformer_ood(model_name, model, tokenizer, df, device):
    """
    Evaluates a transformer model on an Out-Of-Domain dataset (ToxiGen).
    """
    print(f"Tokenizing OOD data for {model_name}...")
    texts = df['text'].tolist()
    labels = df['label'].tolist()
    
    # We evaluate in batches to avoid OOM
    batch_size = 32
    all_preds = []
    
    model.eval()
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            encoded = tokenizer(batch_texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
            encoded = {k: v.to(device) for k, v in encoded.items()}
            
            outputs = model(**encoded)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=-1)
            
            # Assuming label 1 is toxic, label 0 is non-toxic (matches Jigsaw)
            batch_preds = probs[:, 1].cpu().numpy()
            all_preds.extend(batch_preds)
    
    # We no longer calculate AUC here; we just return probabilities
    df['toxicity_score'] = all_preds
    df['model'] = model_name
    return df

def extract_toxigen_identities_and_evaluate(model_name, df_with_preds):
    """
    Helper function to abstract the complex ToxiGen subgroup extraction
    and compute bias metrics.
    """
    # ToxiGen naming varies between versions (skg/toxigen-data vs toxigen/toxigen-data)
    # It may be 'target_groups', 'target_group', or 'group'
    possible_group_cols = ['target_groups', 'target_group', 'group']
    found_group_col = next((c for c in possible_group_cols if c in df_with_preds.columns), None)
    
    identity_cols = []
    identity_matrix_data = {}
    
    if found_group_col:
        import ast
        for i, row in df_with_preds.iterrows():
            group_val = row[found_group_col]
            # Handle cases where it's a string representation of a list
            if isinstance(group_val, str):
                if group_val.startswith('[') and group_val.endswith(']'):
                    try:
                        groups = ast.literal_eval(group_val)
                    except:
                        groups = [group_val]
                else:
                    # Sometimes it's a comma-separated string or just a single string
                    groups = [g.strip() for g in group_val.split(',')]
            elif isinstance(group_val, list):
                groups = group_val
            else:
                groups = [str(group_val)]

            for g in groups:
                if g and g.lower() not in ['none', 'nan', 'null', 'unknown']:
                    if g not in identity_cols:
                        identity_cols.append(g)
                        identity_matrix_data[g] = np.zeros(len(df_with_preds))
                    identity_matrix_data[g][i] = 1.0
                        
    if not identity_cols:
        # Fallback if no target groups found
        print(f"Warning: Could not parse identity groups from column '{found_group_col}'. Only overall metrics will be calculated.")
        identity_cols = ["placeholder"]
        identity_matrix = np.zeros((len(df_with_preds), 1))
    else:
        identity_matrix = np.column_stack([identity_matrix_data[g] for g in identity_cols])
        
    y_true = df_with_preds['label'].to_numpy()
    y_preds = df_with_preds['toxicity_score'].to_numpy()
    
    metrics_df = evaluate_bias(
        y_true=y_true,
        y_pred_probs=y_preds,
        identity_matrix=identity_matrix,
        identity_columns=identity_cols
    )
    
    metrics_df.insert(0, 'Model', model_name)
    return metrics_df

def load_toxigen_dataset(cache_dir, eval_samples=-1):
    """Loads and standardizes labels for the ToxiGen dataset."""
    print("Loading ToxiGen dataset from Hugging Face...")
    try:
        toxigen = load_dataset("skg/toxigen-data", name="train", cache_dir=cache_dir, split="test", token=get_hf_token())
    except Exception as e:
        print(f"Could not load skg/toxigen-data: {e}")
        print("Attempting to load standard 'toxigen/toxigen-data'...")
        try:
             toxigen = load_dataset("toxigen/toxigen-data", name="annotated", cache_dir=cache_dir, split="test", token=get_hf_token())
        except Exception as e2:
             print(f"Could not load toxigen/toxigen-data: {e2}")
             raise e2
             
    df = toxigen.to_pandas()
    
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
        print(f"Warning: Could not identify label column. Available columns: {df.columns}")
        try:
             df['label'] = df['label']
        except KeyError:
             raise ValueError("Could not extract binary labels from ToxiGen dataset.")
             
    if eval_samples > 0:
        if len(df) > eval_samples:
            df = df.sample(n=eval_samples, random_state=42).reset_index(drop=True)
            
    print(f"Loaded {len(df)} samples from ToxiGen for OOD evaluation.")
    return df

def eval_baseline_ood(results_dir, df):
    """Evaluates the saved TF-IDF + Logistic Regression baseline on ToxiGen."""
    baseline_path = os.path.join(results_dir, "baseline_pipeline.joblib")
    if not os.path.exists(baseline_path):
        print(f"\nCould not find baseline pipeline at {baseline_path}. Please run 'make baseline' first to include it in OOD results.")
        return None
        
    print("\n\nEvaluating Baseline (TF-IDF + LR) on OOD data...")
    try:
        pipeline = joblib.load(baseline_path)
        
        # Use 'text' column for inference if it exists, else try 'generation'/'comment_text'
        inference_text = df['text'] if 'text' in df.columns else df.get('generation', df.get('comment_text', None))
        
        if inference_text is not None:
            X_val = [str(t) if t is not None else "" for t in inference_text]
            y_pred_probs = pipeline.predict_proba(X_val)[:, 1]
            
            df_with_preds = df.copy()
            df_with_preds['toxicity_score'] = y_pred_probs
            df_with_preds['label'] = df['label']
            
            return extract_toxigen_identities_and_evaluate('Baseline (TF-IDF + LR)', df_with_preds)
        else:
            print("Warning: Could not find text column in ToxiGen dataset for baseline evaluation.")
            return None
    except Exception as e:
        print(f"Error evaluating baseline on OOD data: {e}")
        import traceback
        traceback.print_exc()
        return None

def run_eval_ood_step(results_dir, cache_dir, output_dir, models, device, eval_samples=-1):
    print("\n--- Running OOD Evaluation (ToxiGen) ---")
    
    df = load_toxigen_dataset(cache_dir, eval_samples)
    
    all_metrics = []
    summary_results = []
    
    # 1. Evaluate Baseline Model if it exists
    baseline_metrics = eval_baseline_ood(results_dir, df)
    if baseline_metrics is not None:
        summary_results.append(baseline_metrics)
        
    # 2. Evaluate Transformer Models
    for base_model_name in models:
        safe_name = base_model_name.replace("/", "_")
        model_output_base_dir = os.path.join(output_dir, f"finetuned_{safe_name}")
        finetuned_model_dir = os.path.join(model_output_base_dir, "small-transformer-toxicity")
        
        print(f"\n\nEvaluating Fine-Tuned Transformer ({base_model_name}) on OOD data...")
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
                
                # Standardize input text column
                if 'text' not in df.columns and 'generation' in df.columns:
                    df['text'] = df['generation']
                elif 'text' not in df.columns and 'comment_text' in df.columns:
                     df['text'] = df['comment_text']
                     
                df_with_preds = eval_transformer_ood(f"Fine-Tuned {base_model_name}", ft_model, tokenizer, df.copy(), device)
                metrics_df = extract_toxigen_identities_and_evaluate(base_model_name, df_with_preds)
                summary_results.append(metrics_df)
                
            except Exception as e:
                print(f"Error evaluating fine-tuned model {base_model_name} on OOD data: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"Could not find fine-tuned directory: {finetuned_model_dir}. Please run 'make finetune' first.")
            
    if summary_results:
        summary_df = pd.concat(summary_results, ignore_index=True)
        out_path = os.path.join(results_dir, "ood_toxigen_metrics.csv")
        summary_df.to_csv(out_path, index=False)
        print(f"\nSaved detailed OOD metrics to {out_path}")
        print(summary_df.head(10).to_string(index=False)) 
