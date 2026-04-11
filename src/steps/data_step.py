import json
import os

from src.data.dataset import download_and_prep_jigsaw

def run_data_step(cache_dir, data_dir, train_samples=20000, eval_samples=5000, seed=42):
    print("\nLoading and Splitting Dataset...")
    full_ds, identity_columns = download_and_prep_jigsaw("train", cache_dir=cache_dir)
    full_ds = full_ds.shuffle(seed=seed)
    
    # Implementing 80/10/10 split
    n = len(full_ds)
    train_end = int(0.8 * n)
    val_end = int(0.9 * n)
    
    train_ds = full_ds.select(range(train_end))
    val_ds = full_ds.select(range(train_end, val_end))
    test_ds = full_ds.select(range(val_end, n))

    # Slice training set if limit is set
    if train_samples > 0:
        train_ds = train_ds.select(range(min(train_samples, len(train_ds))))
        
    # Slice test set if limit is set
    if eval_samples > 0:
        test_ds = test_ds.select(range(min(eval_samples, len(test_ds))))

    # Save to disk
    print(f"Saving splits to {data_dir}...")
    train_ds.save_to_disk(os.path.join(data_dir, "train"))
    val_ds.save_to_disk(os.path.join(data_dir, "val"))
    test_ds.save_to_disk(os.path.join(data_dir, "test"))
    
    # Backward compatibility: save 'baseline_train' and 'eval' as symlinks or copies 
    # to avoid breaking other steps before we update them. Actually, I'll update them now.
    # But for a smoother transition during this turn:
    train_ds.save_to_disk(os.path.join(data_dir, "baseline_train"))
    test_ds.save_to_disk(os.path.join(data_dir, "eval"))
    
    with open(os.path.join(data_dir, "identity_columns.json"), "w") as f:
        json.dump(identity_columns, f)
