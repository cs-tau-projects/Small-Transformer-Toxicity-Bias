import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from src.steps.utils import load_saved_data, eval_transformer

def run_eval_raw_step(data_dir, results_dir, cache_dir, models, device):
    _, eval_ds, identity_columns = load_saved_data(data_dir)
    
    for base_model_name in models:
        print(f"\nLoading Raw Pre-trained Transformer ({base_model_name})...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(base_model_name, cache_dir=cache_dir, token=get_hf_token())
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            raw_model = AutoModelForSequenceClassification.from_pretrained(
                base_model_name, num_labels=2, cache_dir=cache_dir, token=get_hf_token()
            )
            raw_df = eval_transformer(f"Raw {base_model_name}", raw_model, tokenizer, eval_ds, identity_columns, device)
            
            safe_name = base_model_name.replace("/", "_")
            out_path = os.path.join(results_dir, f"{safe_name}_raw_metrics.csv")
            raw_df.to_csv(out_path, index=False)
            print(f"Saved metrics to {out_path}")
        except Exception as e:
            print(f"Error evaluating raw model {base_model_name}: {e}")
