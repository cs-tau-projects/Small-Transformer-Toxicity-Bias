import json
import logging
import os

from src.data.dataset import download_and_prep_jigsaw

logger = logging.getLogger("pipeline")

def run_data_step(cache_dir, data_dir, train_samples=20000, eval_samples=5000, seed=42):
    # Check if all expected files already exist to skip redundant processing
    expected_paths = [
        os.path.join(data_dir, "train"),
        os.path.join(data_dir, "val"),
        os.path.join(data_dir, "test"),
        os.path.join(data_dir, "baseline_train"),
        os.path.join(data_dir, "eval"),
        os.path.join(data_dir, "identity_columns.json")
    ]
    
    if all(os.path.exists(p) for p in expected_paths):
        logger.info(f"Data splits already available in {data_dir}. Skipping loading and splitting.")
        return

    logger.info("Loading and Splitting Dataset...")
    train_ds, train_id_cols = download_and_prep_jigsaw("train", cache_dir=cache_dir)
    test_ds, test_id_cols = download_and_prep_jigsaw("test", cache_dir=cache_dir)
    
    train_ds = train_ds.shuffle(seed=seed)
    test_ds = test_ds.shuffle(seed=seed)
    
    # Create an internal validation split from the train dataset (10%)
    n_train = len(train_ds)
    val_idx = int(0.9 * n_train)
    val_ds = train_ds.select(range(val_idx, n_train))
    train_ds = train_ds.select(range(val_idx))
    
    # Slice training set if limit is set
    if train_samples > 0:
        train_ds = train_ds.select(range(min(train_samples, len(train_ds))))
        
    # Slice test set if limit is set
    if eval_samples > 0:
        test_ds = test_ds.select(range(min(eval_samples, len(test_ds))))

    # Save to disk
    logger.info(f"Saving splits to {data_dir}...")
    train_ds.save_to_disk(os.path.join(data_dir, "train"))
    val_ds.save_to_disk(os.path.join(data_dir, "val"))
    test_ds.save_to_disk(os.path.join(data_dir, "test"))
    
    # Backward compatibility: save 'baseline_train' and 'eval' as symlinks or copies 
    # to avoid breaking other steps before we update them. Actually, I'll update them now.
    # But for a smoother transition during this turn:
    train_ds.save_to_disk(os.path.join(data_dir, "baseline_train"))
    test_ds.save_to_disk(os.path.join(data_dir, "eval"))
    
    # Maintain common identity columns
    identity_columns = list(set(train_id_cols).intersection(set(test_id_cols)))
    
    with open(os.path.join(data_dir, "identity_columns.json"), "w") as f:
        json.dump(identity_columns, f)

    logger.info(f"Data step complete — train: {len(train_ds)}, val: {len(val_ds)}, test: {len(test_ds)}, identities: {len(identity_columns)}")
