import os
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from src.data.data_utils import get_hf_token
from src.steps.utils import eval_transformer, load_saved_data

def run_eval_raw_step(data_dir, results_dir, cache_dir, models, device):
    _, test_ds, identity_columns = load_saved_data(data_dir)

    for base_model_name in tqdm(models, desc="Evaluating raw models"):
        print(f"\nLoading Raw Pre-trained Transformer ({base_model_name})...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(base_model_name, cache_dir=cache_dir, token=get_hf_token())
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            raw_model = AutoModelForSequenceClassification.from_pretrained(
                base_model_name, num_labels=2, cache_dir=cache_dir, token=get_hf_token()
            )
            
            # Note: eval_transformer returns (metrics_df, y_pred_probs)
            raw_df, y_pred_probs = eval_transformer(f"Raw {base_model_name}", raw_model, tokenizer, test_ds, identity_columns, device)
            
            safe_name = base_model_name.replace("/", "_")
            out_path = os.path.join(results_dir, f"{safe_name}_raw_metrics.csv")
            raw_df.to_csv(out_path, index=False)
            
            preds_df = pd.DataFrame({'comment_text': test_ds['comment_text'], 'toxicity_score': y_pred_probs})
            preds_out_path = os.path.join(results_dir, f"preds_{safe_name}_raw.csv")
            preds_df.to_csv(preds_out_path, index=False)
            
            print(f"Saved metrics to {out_path} and predictions to {preds_out_path}")
        except Exception as e:
            print(f"Error evaluating raw model {base_model_name}: {e}")
