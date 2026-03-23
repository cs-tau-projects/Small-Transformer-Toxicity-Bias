import os
import json
from src.data.dataset import download_and_prep_jigsaw

SEED = 42  # Fixed seed for reproducible dataset splits

def run_data_step(cache_dir, data_dir, train_samples=20000, eval_samples=5000):
    print("\nLoading and Splitting Dataset...")
    full_ds, identity_columns = download_and_prep_jigsaw("train", cache_dir=cache_dir)
    full_ds = full_ds.shuffle(seed=SEED)
    
    split_idx = int(0.9 * len(full_ds))
    train_ds = full_ds.select(range(split_idx))
    val_ds = full_ds.select(range(split_idx, len(full_ds)))
    
    # Slice evaluation set
    num_eval = eval_samples if eval_samples > 0 else len(val_ds)
    eval_ds = val_ds.select(range(min(num_eval, len(val_ds))))
    
    # Slice baseline training set
    num_train = train_samples if train_samples > 0 else len(train_ds)
    baseline_train_ds = train_ds.select(range(min(num_train, len(train_ds))))
    
    # Save to disk
    print(f"Saving splits to {data_dir}...")
    baseline_train_ds.save_to_disk(os.path.join(data_dir, "baseline_train"))
    eval_ds.save_to_disk(os.path.join(data_dir, "eval"))
    with open(os.path.join(data_dir, "identity_columns.json"), "w") as f:
        json.dump(identity_columns, f)
