import os
import pandas as pd
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from src.data_utils import get_hf_token
import numpy as np
from src.evaluator import evaluate_bias

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

def run_eval_ood_step(results_dir, cache_dir, output_dir, models, device, eval_samples=-1):
    print("\n--- Running OOD Evaluation (ToxiGen) ---")
    
    print("Loading ToxiGen dataset from Hugging Face...")
    # Loading ToxiGen from HF
    # ToxiGen maps 1 = toxic, 0 = benign. We use the 'test' split.
    # Note: `skg/toxigen-data` is a common source, let's use it.
    try:
        # Using a widely accessible version of toxigen
        toxigen = load_dataset("skg/toxigen-data", name="train", cache_dir=cache_dir, split="test", token=get_hf_token())
    except Exception as e:
        print(f"Could not load skg/toxigen-data: {e}")
        print("Attempting to load standard 'toxigen/toxigen-data'...")
        try:
             toxigen = load_dataset("toxigen/toxigen-data", name="annotated", cache_dir=cache_dir, split="test", token=get_hf_token())
        except Exception as e2:
             print(f"Could not load toxigen/toxigen-data: {e2}")
             raise e2
             
    # Convert to pandas for easier manipulation and evaluation
    df = toxigen.to_pandas()
    
    # Mapping labels to standard 0 and 1 if they are not already.
    # Depending on the dataset version, it might be 'toxicity', 'label', or 'toxicity_human'
    # Defaulting to checking typical column names.
    if 'toxicity_human' in df.columns:
        df['label'] = (df['toxicity_human'] > 3).astype(int) # Usually graded 1-5, so >3 is highly toxic
    elif 'toxicity_human_annotated' in df.columns:
        df['label'] = df['toxicity_human_annotated']
    elif 'label' in df.columns:
         pass # Already labeled
    elif 'toxicity' in df.columns:
         df['label'] = df['toxicity']
    elif 'toxicity_score' in df.columns:
         df['label'] = df['toxicity_score'].apply(lambda x: 1 if x >= 0.5 else 0)
    else:
        # If we can't find a direct label, assume binary label exists somewhere or guess based on features.
        print(f"Warning: Could not identify label column. Available columns: {df.columns}")
        # Default mapping for `skg/toxigen-data` which often uses `label` or `toxicity`
        try:
             df['label'] = df['label']
        except KeyError:
             raise ValueError("Could not extract binary labels from ToxiGen dataset. Please check the dataset schema.")
             
    if eval_samples > 0:
        if len(df) > eval_samples:
            df = df.sample(n=eval_samples, random_state=42).reset_index(drop=True)
            
    print(f"Loaded {len(df)} samples from ToxiGen for OOD evaluation.")
    
    # Evaluate each fine-tuned model
    all_metrics = []
    summary_results = []
    
    for base_model_name in models:
        safe_name = base_model_name.replace("/", "_")
        model_output_base_dir = os.path.join(output_dir, f"finetuned_{safe_name}")
        finetuned_model_dir = os.path.join(model_output_base_dir, "small-transformer-toxicity")
        
        print(f"\nEvaluating Fine-Tuned Transformer ({base_model_name}) on OOD data...")
        model_load_path = finetuned_model_dir
        if os.path.exists(finetuned_model_dir):
            if not os.path.exists(os.path.join(finetuned_model_dir, "config.json")):
                # Fallback to last checkpoint if root doesn't have config
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
                
                # We need to ensure we're matching the expected input column name.
                # Usually toxigen uses 'text' or 'generation'.
                if 'text' not in df.columns and 'generation' in df.columns:
                    df['text'] = df['generation']
                elif 'text' not in df.columns and 'comment_text' in df.columns:
                     df['text'] = df['comment_text']
                     
                df_with_preds = eval_transformer_ood(f"Fine-Tuned {base_model_name}", ft_model, tokenizer, df.copy(), device)
                
                # Toxigen defines targets in lists, usually under 'target_groups' or similar
                # We need to extract unique identities and create a binary matrix.
                identity_cols = []
                identity_matrix_data = {}
                
                if 'target_groups' in df_with_preds.columns:
                    # Some versions have lists of groups like "['black', 'muslim']"
                    import ast
                    for i, row in df_with_preds.iterrows():
                        groups = row['target_groups']
                        if isinstance(groups, str):
                            try:
                                groups = ast.literal_eval(groups)
                            except:
                                groups = [groups]
                        if isinstance(groups, list):
                            for g in groups:
                                if g and g != 'none' and g != 'None':
                                    if g not in identity_cols:
                                        identity_cols.append(g)
                                        identity_matrix_data[g] = np.zeros(len(df_with_preds))
                                    identity_matrix_data[g][i] = 1.0
                                    
                if not identity_cols:
                    # Fallback if target_groups isn't clear
                    print("Warning: Could not parse target_groups. Only overall metrics will be calculated.")
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
                
                metrics_df.insert(0, 'Model', base_model_name)
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
        print(summary_df.head(10).to_string(index=False)) # Print first few for brevity
