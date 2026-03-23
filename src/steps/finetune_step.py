import os
import subprocess
import sys
from tqdm import tqdm

def _extract_training_params(full_config):
    """Extracts training parameters from the global config dictionary."""
    training_config = full_config.get("training", {})
    data_config = full_config.get("data", {})
    
    return {
        "epochs": training_config.get("epochs", 1),
        "learning_rate": training_config.get("learning_rate", 2e-5),
        "batch_size": data_config.get("train_batch_size", 32)
    }

def run_finetune_step(models, output_dir, full_config, seed=42, train_samples=-1):
    # Extract parameters using the helper function
    params = _extract_training_params(full_config)
    model_name_from_config = full_config.get("model", {}).get("name")

    # If a specific model is named in the config, use it.
    # Otherwise, use the list of models passed to the function.
    models_to_run = [model_name_from_config] if model_name_from_config else models

    for base_model_name in tqdm(models_to_run, desc="Fine-tuning models"):
        print(f"\nTriggering fine-tuning for {base_model_name}...")
        safe_name = base_model_name.replace("/", "_")
        model_output_base_dir = os.path.join(output_dir, f"finetuned_{safe_name}")
        finetuned_model_dir = os.path.join(model_output_base_dir, "small-transformer-toxicity")

        if not os.path.exists(os.path.join(finetuned_model_dir, "config.json")):
            cmd = [
                sys.executable, "-m", "src.train",
                "--model_name", base_model_name,
                "--output_base_dir", model_output_base_dir,
                "--epochs", str(params["epochs"]),
                "--batch_size", str(params["batch_size"]),
                "--learning_rate", str(params["learning_rate"]),
                "--seed", str(seed),
                "--train_samples", str(train_samples)
            ]
            
            # Force Hugging Face Trainer to show its internal step progress bar
            env = os.environ.copy()
            env["TQDM_FORCE"] = "1"
            subprocess.run(cmd, check=True, env=env)
        else:
            print(f"Fine-tuned model checkpoint found in {finetuned_model_dir}. Skipping training.")
