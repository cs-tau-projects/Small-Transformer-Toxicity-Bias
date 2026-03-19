import os
import subprocess
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from tqdm import tqdm

from src.dataset import download_and_prep_jigsaw
from src.evaluator import evaluate_bias

def train_and_eval_baseline(train_ds, val_ds, identity_columns):
    """Trains and evaluates the baseline TF-IDF + Logistic Regression model."""
    print("\n--- Training Baseline (TF-IDF + LogReg) ---")
    X_train = train_ds['comment_text']
    y_train = train_ds['is_toxic']
    
    X_val = val_ds['comment_text']
    y_val = val_ds['is_toxic']
    
    # Extract identity matrix for evaluation
    identities_val = [val_ds[col] for col in identity_columns]
    identity_matrix_val = np.array(identities_val).T
    
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=10000, stop_words='english')),
        ('clf', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42))
    ])
    
    pipeline.fit(X_train, y_train)
    
    print("Evaluating Baseline...")
    y_pred_probs = pipeline.predict_proba(X_val)[:, 1]
    
    metrics_df = evaluate_bias(
        y_true=np.array(y_val),
        y_pred_probs=y_pred_probs,
        identity_matrix=identity_matrix_val,
        identity_columns=identity_columns,
        threshold=0.5
    )
    return metrics_df

def get_transformer_predictions(model, tokenizer, dataset, device, batch_size=32):
    """Generate predictions for a Transformer model on the dataset."""
    model.eval()
    model.to(device)
    all_probs = []
    
    texts = dataset['comment_text']
    
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Inferencing"):
            batch_texts = texts[i:i+batch_size]
            inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            outputs = model(**inputs)
            logits = outputs.logits
            
            if logits.shape[1] == 2:
                probs = torch.softmax(logits, dim=-1)[:, 1]
            else:
                probs = torch.sigmoid(logits)[:, 0]
                
            all_probs.extend(probs.cpu().numpy())
            
    return np.array(all_probs)

def eval_transformer(model_desc, model, tokenizer, val_ds, identity_columns, device):
    """Evaluates a Transformer model given the validation dataset."""
    print(f"\n--- Evaluating {model_desc} ---")
    y_val = val_ds['is_toxic']
    
    identities_val = [val_ds[col] for col in identity_columns]
    identity_matrix_val = np.array(identities_val).T
    
    y_pred_probs = get_transformer_predictions(model, tokenizer, val_ds, device)
    
    metrics_df = evaluate_bias(
        y_true=np.array(y_val),
        y_pred_probs=y_pred_probs,
        identity_matrix=identity_matrix_val,
        identity_columns=identity_columns,
        threshold=0.5
    )
    return metrics_df

def format_final_report(all_results_dict):
    """Combines metrics from all models into a comparative table suitable for an ACL report."""
    print("\n" + "="*80)
    print("FINAL COMPARISON REPORT (ACL Format)")
    print("="*80)
    
    def extract_summary(df, model_name):
        df_copy = df.copy()
        # Some legacy columns might not be there cleanly, so we check carefully
        cols = ['Identity']
        if 'Total Examples' in df.columns:
            cols.append('Total Examples')
        metric_cols = [col for col in df.columns if col.startswith(('1.', '2.', '3.', '4.', '5.', '6.'))]
        
        # We rename the metric columns to have the model name prefixed
        rename_dict = {col: f"{model_name} {col.split('. ')[1]}" for col in metric_cols}
        df_copy = df_copy[cols + metric_cols].rename(columns=rename_dict)
        return df_copy
    
    # Base dataframe to merge on
    if "Baseline" in all_results_dict:
        final_df = all_results_dict["Baseline"][['Identity', 'Total Examples']].copy()
    else:
        # Fallback if no baseline
        first_key = list(all_results_dict.keys())[0]
        final_df = all_results_dict[first_key][['Identity', 'Total Examples']].copy()

    for model_name, df in all_results_dict.items():
        sum_df = extract_summary(df, model_name)
        if 'Total Examples' in sum_df.columns and model_name != "Baseline":
            sum_df = sum_df.drop(columns=['Total Examples'])
        final_df = final_df.merge(sum_df, on='Identity', how='left')
    
    print("\n1. Overall AUC Comparison:")
    auc_cols = ['Identity'] + [c for c in final_df.columns if 'Overall AUC' in c]
    if auc_cols[1:]:  # Ensure matching cols exist
        overall_auc = final_df[auc_cols].head(1).copy()
        overall_auc.loc[0, 'Identity'] = 'Overall Dataset'
        print(overall_auc.to_string(index=False))
    
    print("\n2. Subgroup AUC Comparison:")
    subgroup_cols = ['Identity'] + [c for c in final_df.columns if 'Subgroup AUC' in c]
    subgroup_auc = final_df[subgroup_cols]
    print(subgroup_auc.to_string(index=False))
    
    print("\n3. FNR Comparison (Subgroup and Overall):")
    fnr_cols = ['Identity'] + [c for c in final_df.columns if 'FNR' in c]
    fnr = final_df[fnr_cols]
    print(fnr.to_string(index=False))

    print("\n4. FPR Comparison (Subgroup and Overall):")
    fpr_cols = ['Identity'] + [c for c in final_df.columns if 'FPR' in c]
    fpr = final_df[fpr_cols]
    print(fpr.to_string(index=False))

def main():
    transformer_models = [
        "distilbert-base-uncased",
        "distilroberta-base",
        "google/bert_uncased_L-4_H-512_A-8"
    ]
    llama_model = "meta-llama/Llama-3.2-1B"
    
    output_base_dir = "./outputs"
    cache_dir = os.path.join(output_base_dir, ".cache")
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    
    print(f"Using device: {device}")
    
    # 1. Load Data Splits
    print("\nLoading Dataset...")
    full_ds, identity_columns = download_and_prep_jigsaw("train", cache_dir=cache_dir)
    full_ds = full_ds.shuffle(seed=42)
    
    split_idx = int(0.9 * len(full_ds))
    train_ds = full_ds.select(range(split_idx))
    val_ds = full_ds.select(range(split_idx, len(full_ds)))
    
    eval_ds = val_ds.select(range(min(5000, len(val_ds))))
    baseline_train_ds = train_ds.select(range(min(20000, len(train_ds))))

    all_results_dict = {}

    # 2. RUN BASELINE
    baseline_df = train_and_eval_baseline(baseline_train_ds, eval_ds, identity_columns)
    all_results_dict["Baseline"] = baseline_df
    
    # 3. RUN TRANSFORMER MODELS
    for base_model_name in transformer_models:
        print(f"\n{'='*50}\nProcessing Model: {base_model_name}\n{'='*50}")
        model_output_base_dir = os.path.join(output_base_dir, f"finetuned_{base_model_name.replace('/', '_')}")
        finetuned_model_dir = os.path.join(model_output_base_dir, "small-transformer-toxicity")
        
        # RAW TRANSFORMER
        print(f"\nLoading Raw Pre-trained Transformer ({base_model_name})...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(base_model_name, cache_dir=cache_dir)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            raw_model = AutoModelForSequenceClassification.from_pretrained(
                base_model_name, num_labels=2, cache_dir=cache_dir
            )
            raw_df = eval_transformer(f"Raw {base_model_name}", raw_model, tokenizer, eval_ds, identity_columns, device)
            all_results_dict[f"{base_model_name} Raw"] = raw_df
        except Exception as e:
            print(f"Error evaluating raw model {base_model_name}: {e}")

        # CONDITIONAL TRAINING & EVALUATE FINE-TUNED TRANSFORMER
        if not os.path.exists(finetuned_model_dir) or not any(fname.startswith("checkpoint") or fname == "config.json" for fname in os.listdir(finetuned_model_dir) if os.path.isdir(os.path.join(finetuned_model_dir, fname)) or fname == "config.json"):
            print(f"\nFine-tuned model not found for {base_model_name}. Triggering training script...")
            cmd = [
                "python", "-m", "src.train",
                "--model_name", base_model_name,
                "--output_base_dir", model_output_base_dir,
                "--epochs", "1",
                "--batch_size", "32"
            ]
            subprocess.run(cmd, check=True)
        else:
            print(f"\nFine-tuned model checkpoint found for {base_model_name}. Skipping training.")
            
        print(f"\nLoading Fine-Tuned Transformer ({base_model_name})...")
        model_load_path = finetuned_model_dir
        if os.path.exists(finetuned_model_dir):
            checkpoints = [d for d in os.listdir(finetuned_model_dir) if d.startswith("checkpoint-")]
            if checkpoints:
                checkpoints.sort(key=lambda x: int(x.split("-")[-1]))
                model_load_path = os.path.join(finetuned_model_dir, checkpoints[-1])
            
            try:
                ft_model = AutoModelForSequenceClassification.from_pretrained(model_load_path, num_labels=2)
                ft_df = eval_transformer(f"Fine-Tuned {base_model_name}", ft_model, tokenizer, eval_ds, identity_columns, device)
                all_results_dict[f"{base_model_name} Finetuned"] = ft_df

            except Exception as e:
                print(f"Error loading fine-tuned model {base_model_name}: {e}")

    # 4. LLAMA INFERENCE ONLY
    print(f"\n{'='*50}\nProcessing Inference-Only Model: {llama_model}\n{'='*50}")
    try:
        llama_tokenizer = AutoTokenizer.from_pretrained(llama_model, cache_dir=cache_dir)
        if llama_tokenizer.pad_token is None:
            llama_tokenizer.pad_token = llama_tokenizer.eos_token
        llama_raw = AutoModelForSequenceClassification.from_pretrained(
            llama_model, num_labels=2, cache_dir=cache_dir
        )
        llama_df = eval_transformer(f"Raw {llama_model}", llama_raw, llama_tokenizer, eval_ds, identity_columns, device)
        all_results_dict[f"{llama_model} Raw"] = llama_df
    except Exception as e:
        print(f"Error evaluating Llama model: {e}")

    # 5. FINAL REPORT
    format_final_report(all_results_dict)
    
if __name__ == "__main__":
    main()
