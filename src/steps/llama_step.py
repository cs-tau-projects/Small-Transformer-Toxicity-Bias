import os
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.steps.utils import load_saved_data
from src.evaluator import evaluate_bias

def get_llama_toxicity_scores(model, tokenizer, dataset, device, batch_size=8):
    """
    Zero-shot toxicity scoring via next-token probability.
    Prompt: '...comment... Is this comment toxic? Answer: '
    We take P(token "Yes") / (P("Yes") + P("No")) as the toxicity score.
    """
    model.eval()
    model.to(device)

    yes_id = tokenizer.encode("Yes", add_special_tokens=False)[0]
    no_id  = tokenizer.encode("No",  add_special_tokens=False)[0]

    all_scores = []
    texts = dataset["comment_text"]

    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="LLaMA zero-shot"):
            batch = texts[i : i + batch_size]
            prompts = [
                f'Comment: "{t}"\nIs this comment toxic? Answer:'
                for t in batch
            ]
            enc = tokenizer(prompts, return_tensors="pt", padding=True,
                            truncation=True, max_length=256).to(device)
            logits = model(**enc).logits          # (B, seq_len, vocab)
            last_logits = logits[:, -1, :]        # logits at the final position
            yes_no = last_logits[:, [yes_id, no_id]]
            probs = torch.softmax(yes_no, dim=-1)[:, 0]  # P(Yes)
            all_scores.extend(probs.cpu().float().numpy())

    return np.array(all_scores)

def run_llama_step(data_dir, results_dir, cache_dir, llama_model, device):
    _, eval_ds, identity_columns = load_saved_data(data_dir)

    print(f"\nZero-shot toxicity scoring with {llama_model}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(llama_model, cache_dir=cache_dir)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"   # required for causal LMs

        model = AutoModelForCausalLM.from_pretrained(
            llama_model, cache_dir=cache_dir,
            torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        )

        y_pred_probs = get_llama_toxicity_scores(model, tokenizer, eval_ds, device)

        y_val = np.array(eval_ds["is_toxic"])
        identities_val = np.array([eval_ds[col] for col in identity_columns]).T

        metrics_df = evaluate_bias(
            y_true=y_val,
            y_pred_probs=y_pred_probs,
            identity_matrix=identities_val,
            identity_columns=identity_columns,
            threshold=0.5,
        )

        safe_name = llama_model.replace("/", "_")
        out_path = os.path.join(results_dir, f"{safe_name}_raw_metrics.csv")
        metrics_df.to_csv(out_path, index=False)
        print(f"Saved LLaMA metrics to {out_path}")
    except Exception as e:
        print(f"Error evaluating LLaMA model: {e}")
