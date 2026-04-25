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
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger("pipeline")

class LlamaPromptDataset(Dataset):
    def __init__(self, texts, tokenizer):
        self.texts = texts
        self.tokenizer = tokenizer
        self.system_prompt = (
            "You are a fair and unbiased moderator. Your task is to classify whether a comment is toxic. "
            "Toxicity includes harassment, hate speech, threats, or severe profanity. "
            "Merely mentioning identities like 'gay', 'Muslim', or 'black' is NOT inherently toxic. "
            "Answer strictly with 'Yes' if toxic, or 'No' if not toxic."
        )

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        # apply_chat_template prepares the exact prompt format for Instruct models
        if hasattr(self.tokenizer, "chat_template") and self.tokenizer.chat_template is not None:
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": f"Comment: '{self.texts[idx]}'"}
            ]
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            # Fallback for base models without a chat template.
            # This matches the exact prompt used previously for the base model.
            return f'Comment: "{self.texts[idx]}"\nIs this comment toxic? Answer:'

def get_yes_no_tokens(tokenizer):
    """Find all variants of 'yes' and 'no' in the vocabulary.

    Explicitly enumerates known BPE surface forms (with and without the
    Ġ / _ space-prefix) so that the space-prefixed tokens the model is
    most likely to generate (e.g. ĠYes, ĠNo) are always included.
    """
    vocab = tokenizer.get_vocab()

    yes_variants = ["Yes", "yes", "YES", "ĠYes", "Ġyes", "ĠYES",
                     "_Yes", "_yes", "_YES"]
    no_variants  = ["No",  "no",  "NO",  "ĠNo",  "Ġno",  "ĠNO",
                     "_No",  "_no",  "_NO"]

    yes_tokens = [vocab[t] for t in yes_variants if t in vocab]
    no_tokens  = [vocab[t] for t in no_variants  if t in vocab]

    # Fallback: encode the space-prefixed word directly
    if not yes_tokens:
        yes_tokens = [tokenizer.encode(" Yes", add_special_tokens=False)[-1]]
    if not no_tokens:
        no_tokens = [tokenizer.encode(" No", add_special_tokens=False)[-1]]

    logger.debug(f"Yes token IDs ({len(yes_tokens)}): {yes_tokens}")
    logger.debug(f"No  token IDs ({len(no_tokens)}): {no_tokens}")
    return yes_tokens, no_tokens

def get_llama_toxicity_scores(model, tokenizer, texts, device, batch_size=32, total=None, save_every=None, save_path=None):
    """
    Zero-shot toxicity scoring via next-token probability with DataLoader optimization.
    """
    model.eval()
    yes_tokens, no_tokens = get_yes_no_tokens(tokenizer)

    dataset = LlamaPromptDataset(texts, tokenizer)
    
    # Custom collate function to tokenize in the background workers
    def collate_fn(batch_prompts):
        return tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=512)

    # Disable tokenizer parallelism to avoid deadlock in DataLoader workers
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Use multiple workers for tokenization to overlap with GPU compute
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4 if device.type == "cuda" else 0,
        pin_memory=True if device.type == "cuda" else False,
        collate_fn=collate_fn
    )

    all_scores = []
    
    with torch.no_grad():
        # Cross-platform autocast for MPS (Mac), CUDA (Titan), or CPU
        device_type = "cuda" if device.type == "cuda" else "cpu"
        # MPS doesn't support generic autocast in all torch versions, so we use it carefully
        with torch.autocast(device_type=device_type, enabled=(device.type == "cuda")):
            for i, enc in enumerate(tqdm(dataloader, desc="LLaMA zero-shot", total=len(dataloader))):
                enc = enc.to(device)
                
                outputs = model(**enc, logits_to_keep=1)
                last_logits = outputs.logits[:, -1, :]
                
                # Extract logits for all Yes/No token variants
                yes_logits = last_logits[:, yes_tokens]
                no_logits = last_logits[:, no_tokens]
                
                # Combine probabilities of multiple variants using logsumexp
                yes_score = torch.logsumexp(yes_logits, dim=-1)
                no_score = torch.logsumexp(no_logits, dim=-1)
                
                # P(Yes) = exp(yes_score) / (exp(yes_score) + exp(no_score)) = sigmoid(yes_score - no_score)
                probs = torch.sigmoid(yes_score - no_score)
                all_scores.extend(probs.cpu().float().numpy())
                
                # Periodic saving for long runs
                if save_every and save_path and (i + 1) % (save_every // batch_size) == 0:
                    temp_df = pd.DataFrame({"toxicity_score": all_scores})
                    temp_df.to_csv(save_path + ".partial", index=False)

    return np.array(all_scores)

def run_llama_step(data_dir, results_dir, cache_dir, llama_model, device, batch_size=32):
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
            torch_dtype=torch.float16 if device.type in ["cuda", "mps"] else torch.float32,
            attn_implementation="sdpa",
        )
        model.to(device)

        # torch.compile can trigger NameError inside dynamo for some
        # transformers/torch version combinations, so we skip it.
        model_to_use = model

        # 1. In-Distribution (ID) Evaluation on Jigsaw
        logger.info("Evaluating LLaMA on Jigsaw (ID)...")
        preds_id_out_path = os.path.join(results_dir, f"preds_{safe_name}_llama.csv")
        y_pred_probs_id = get_llama_toxicity_scores(
            model_to_use, tokenizer, test_ds["comment_text"], device, batch_size=batch_size, total=len(test_ds),
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
                model_to_use, tokenizer, df_ood["text"], device, batch_size=batch_size, total=len(df_ood),
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
