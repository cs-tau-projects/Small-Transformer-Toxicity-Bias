import os
import subprocess

def run_finetune_step(models, output_dir):
    for base_model_name in models:
        print(f"\nTriggering fine-tuning for {base_model_name}...")
        safe_name = base_model_name.replace("/", "_")
        model_output_base_dir = os.path.join(output_dir, f"finetuned_{safe_name}")
        finetuned_model_dir = os.path.join(model_output_base_dir, "small-transformer-toxicity")
        
        if not os.path.exists(os.path.join(finetuned_model_dir, "config.json")):
            cmd = [
                "python", "-m", "src.train",
                "--model_name", base_model_name,
                "--output_base_dir", model_output_base_dir,
                "--epochs", "1",
                "--batch_size", "32"
            ]
            subprocess.run(cmd, check=True)
        else:
            print(f"Fine-tuned model checkpoint found in {finetuned_model_dir}. Skipping training.")
