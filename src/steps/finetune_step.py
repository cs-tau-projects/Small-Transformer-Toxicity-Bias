import logging
import os
import subprocess
import sys
from tqdm import tqdm

logger = logging.getLogger("pipeline")

def run_finetune_step(models, output_dir, cache_dir=None, seed=42, train_samples=-1, data_dir=None):
    for base_model_name in tqdm(models, desc="Fine-tuning models"):
        logger.info(f"Triggering fine-tuning for {base_model_name}...")
        safe_name = base_model_name.replace("/", "_")
        model_output_base_dir = os.path.join(output_dir, f"finetuned_{safe_name}")
        finetuned_model_dir = os.path.join(model_output_base_dir, "small-transformer-toxicity")
        
        if not os.path.exists(os.path.join(finetuned_model_dir, "config.json")):
            cmd = [
                sys.executable,
                "-m",
                "src.train",
                "--model_name",
                base_model_name,
                "--output_base_dir",
                model_output_base_dir,
                "--epochs",
                "1",
                "--batch_size",
                "32",
                "--seed",
                str(seed),
                "--train_samples",
                str(train_samples),
                "--cache_dir",
                str(cache_dir) if cache_dir else os.path.join(model_output_base_dir, ".cache"),
            ]
            # Forward the pre-saved data splits dir so train.py uses the same
            # 90/10 split as all other pipeline steps (fixes SEED-2 data split inconsistency)
            if data_dir and os.path.isdir(data_dir):
                cmd += ["--data_dir", data_dir]
            # Force Hugging Face Trainer to show its internal step progress bar
            env = os.environ.copy()
            env["TQDM_FORCE"] = "1"
            result = subprocess.run(cmd, check=True, env=env)
            logger.info(f"Fine-tuning subprocess for {base_model_name} finished with return code {result.returncode}")
        else:
            logger.info(f"Fine-tuned model checkpoint found in {finetuned_model_dir}. Skipping training.")
