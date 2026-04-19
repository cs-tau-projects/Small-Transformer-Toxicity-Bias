import logging
import os
import numpy as np
import torch
import pandas as pd
from huggingface_hub import HfApi
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.evaluator import evaluate_bias
from src.steps.utils import load_saved_data

logger = logging.getLogger("pipeline")

def get_llama_toxicity_scores(model, tokenizer, texts, device, batch_size=8, total=None, save_every=None, save_path=None):
    """
    Zero-shot toxicity scoring via next-token probability.
    Texts can be any iterable (list, generator, etc.) for memory efficiency.
    If save_every and save_path are provided, saves partial results periodically.
    """
    model.eval()
    model.to(device)

    yes_id = tokenizer.encode("Yes", add_special_tokens=False)[0]
    no_id = tokenizer.encode("No", add_special_tokens=False)[0]

    all_scores = []
    
    # Helper to batch an iterable
    from itertools import islice
    def batch_iterator(iterable, size):
        it = iter(iterable)
        while True:
            chunk = list(islice(it, size))
            if not chunk:
                break
            yield chunk

    with torch.no_grad():
        num_batches = (total + batch_size - 1) // batch_size if total else None
        
        for i, batch in enumerate(tqdm(batch_iterator(texts, batch_size), desc="LLaMA zero-shot", total=num_batches)):
            prompts = [f'Comment: "{t}"\nIs this comment toxic? Answer:' for t in batch]
            enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)
            logits = model(**enc).logits 
            last_logits = logits[:, -1, :] 
            yes_no = last_logits[:, [yes_id, no_id]]
            probs = torch.softmax(yes_no, dim=-1)[:, 0]  # P(Yes)
            all_scores.extend(probs.cpu().float().numpy())
            
            # Periodic saving for long runs
            if save_every and save_path and (i + 1) % (save_every // batch_size) == 0:
                temp_df = pd.DataFrame({"toxicity_score": all_scores})
                temp_df.to_csv(save_path + ".partial", index=False)

    return np.array(all_scores)

def run_llama_step(data_dir, results_dir, cache_dir, llama_model, device):
    _, test_ds, identity_columns = load_saved_data(data_dir)

    logger.info(f"Zero-shot toxicity scoring with {llama_model}...")
    safe_name = llama_model.replace("/", "_")

    # Pre-check authentication for gated models
    api = HfApi()
    try:
        user_info = api.whoami()
        logger.info(f"Authenticated as: {user_info['name']}")
    except Exception:
        logger.warning("Not authenticated with Hugging Face. Gated models like LLaMA may fail to load.")
        logger.warning("Suggestion: Run 'make hf-login' (which runs 'hf auth login') to authenticate.")

    try:
        tokenizer = AutoTokenizer.from_pretrained(llama_model, cache_dir=cache_dir)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"  # required for causal LMs

        model = AutoModelForCausalLM.from_pretrained(
            llama_model,
            cache_dir=cache_dir,
            torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        )

        # 1. In-Distribution (ID) Evaluation on Jigsaw
        logger.info("Evaluating LLaMA on Jigsaw (ID)...")
        preds_id_out_path = os.path.join(results_dir, f"preds_{safe_name}_llama.csv")
        y_pred_probs_id = get_llama_toxicity_scores(
            model, tokenizer, test_ds["comment_text"], device, total=len(test_ds),
            save_every=10000, save_path=preds_id_out_path
        )

        y_test = np.array(test_ds["is_toxic"])
        identities_test = np.array([test_ds[col] for col in identity_columns]).T

        metrics_id_df = evaluate_bias(
            y_true=y_test,
            y_pred_probs=y_pred_probs_id,
            identity_matrix=identities_test,
            identity_columns=identity_columns,
            threshold=0.5,
        )

        metrics_id_out_path = os.path.join(results_dir, f"{safe_name}_raw_metrics.csv")
        metrics_id_df.to_csv(metrics_id_out_path, index=False)
        
        preds_id_df = pd.DataFrame({'comment_text': test_ds['comment_text'], 'toxicity_score': y_pred_probs_id})
        preds_id_df.to_csv(preds_id_out_path, index=False)
        # Cleanup partial file if it exists
        if os.path.exists(preds_id_out_path + ".partial"):
            os.remove(preds_id_out_path + ".partial")
        logger.info(f"Saved LLaMA ID results to {metrics_id_out_path} and predictions to {preds_id_out_path}")

        # 2. Out-of-Distribution (OOD) Evaluation on ToxiGen
        toxigen_path = os.path.join(data_dir, "toxigen_standardized.parquet")
        if os.path.exists(toxigen_path):
            logger.info("Evaluating LLaMA on ToxiGen (OOD)...")
            df_ood = pd.read_parquet(toxigen_path)
            
            preds_ood_out_path = os.path.join(results_dir, f"preds_{safe_name}_llama_ood.csv")
            y_pred_probs_ood = get_llama_toxicity_scores(
                model, tokenizer, df_ood["text"], device, total=len(df_ood),
                save_every=10000, save_path=preds_ood_out_path
            )
            
            df_ood_with_preds = df_ood.copy()
            df_ood_with_preds['toxicity_score'] = y_pred_probs_ood
            
            from src.steps.eval_ood_step import extract_toxigen_identities_and_evaluate
            llama_display_name = f"{llama_model} (Zero-shot)"
            metrics_ood_df = extract_toxigen_identities_and_evaluate(llama_display_name, df_ood_with_preds)

            # Append to the shared OOD metrics file so the reporter picks it up uniformly
            metrics_ood_out_path = os.path.join(results_dir, "ood_toxigen_metrics.csv")
            if os.path.exists(metrics_ood_out_path):
                existing = pd.read_csv(metrics_ood_out_path)
                existing = existing[existing["Model"] != llama_display_name]
                combined = pd.concat([existing, metrics_ood_df], ignore_index=True)
            else:
                combined = metrics_ood_df
            combined.to_csv(metrics_ood_out_path, index=False)

            preds_ood_df = pd.DataFrame({'text': df_ood['text'], 'toxicity_score': y_pred_probs_ood})
            preds_ood_df.to_csv(preds_ood_out_path, index=False)
            # Cleanup partial file
            if os.path.exists(preds_ood_out_path + ".partial"):
                os.remove(preds_ood_out_path + ".partial")
            logger.info(f"Appended LLaMA OOD results to {metrics_ood_out_path} and saved predictions to {preds_ood_out_path}")
        else:
            logger.warning(f"Standardized ToxiGen dataset not found at {toxigen_path}. Skipping LLaMA OOD evaluation.")
            logger.warning("Hint: Run 'make eval-ood' first to generate the standardized ToxiGen file.")

    except Exception as e:
        err_msg = str(e)
        if "403" in err_msg or "gated" in err_msg.lower():
            logger.error(f"Authentication failure or Access Denied to LLaMA model.", exc_info=True)
            logger.error(f"{llama_model} is a gated repository. Request access at: https://huggingface.co/{llama_model}")
            logger.error("Once approved, run 'make hf-login' (or 'hf auth login') in your terminal.")
        else:
            logger.error(f"Error evaluating LLaMA model: {e}", exc_info=True)
