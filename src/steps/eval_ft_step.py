import logging
import os
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from src.steps.utils import eval_transformer, load_saved_data

logger = logging.getLogger("pipeline")

def run_eval_ft_step(data_dir, results_dir, cache_dir, output_dir, models, device):
    _, test_ds, identity_columns = load_saved_data(data_dir)

    for base_model_name in tqdm(models, desc="Evaluating fine-tuned models"):
        safe_name = base_model_name.replace("/", "_")
        model_output_base_dir = os.path.join(output_dir, f"finetuned_{safe_name}")
        finetuned_model_dir = os.path.join(model_output_base_dir, "small-transformer-toxicity")

        logger.info(f"Evaluating Fine-Tuned Transformer ({base_model_name})...")
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
                
                ft_df, y_pred_probs = eval_transformer(f"Fine-Tuned {base_model_name}", ft_model, tokenizer, test_ds, identity_columns, device)
                
                out_path = os.path.join(results_dir, f"{safe_name}_finetuned_metrics.csv")
                ft_df.to_csv(out_path, index=False)
                
                preds_df = pd.DataFrame({'comment_text': test_ds['comment_text'], 'toxicity_score': y_pred_probs})
                preds_out_path = os.path.join(results_dir, f"preds_{safe_name}_finetuned.csv")
                preds_df.to_csv(preds_out_path, index=False)
                
                logger.info(f"Saved metrics to {out_path} and predictions to {preds_out_path}")
            except Exception as e:
                logger.error(f"Error loading fine-tuned model {base_model_name}: {e}", exc_info=True)
        else:
            logger.warning(f"Could not find fine-tuned directory: {finetuned_model_dir}")
