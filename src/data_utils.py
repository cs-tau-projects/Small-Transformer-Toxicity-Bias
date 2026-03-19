import os
import getpass
from datasets import load_dataset

ALL_IDENTITY_COLUMNS = [
    "asian", "atheist", "bisexual", "black", "buddhist", "christian", "female", 
    "heterosexual", "hindu", "homosexual_gay_or_lesbian", "intellectual_or_learning_disability", 
    "jewish", "latino", "male", "muslim", "other_disability", "other_gender", 
    "other_race_or_ethnicity", "other_religion", "other_sexual_orientation", 
    "physical_disability", "psychiatric_or_mental_illness", "transgender", "white"
]

def get_huggingface_cache_dir():
    try:
        username = getpass.getuser()
    except Exception:
        username = "unknown_user"
        
    cluster_base = "/vol/joberant_nobck/data/NLP_368307701_2526a"
    if os.path.exists(cluster_base):
        cache_dir = f"{cluster_base}/{username}/.cache/huggingface"
    else:
        cache_dir = "./.hf_cache"
    return cache_dir

def load_jigsaw_data(split="train", threshold=0.5):
    """
    Loads Jigsaw Unintended Bias dataset and returns comment_text, toxicity label, and identity metadata.
    Ensures memory efficiency by using Hugging Face datasets.
    """
    cache_dir = get_huggingface_cache_dir()
    
    data_files = {}
    if split == "train":
        data_files["train"] = "hf://datasets/shuttie/jigsaw-unintended-bias/data/train.csv.gz"
    elif split == "test":
        data_files["test"] = "hf://datasets/shuttie/jigsaw-unintended-bias/data/test_private_expanded.csv.gz"
        
    dataset = load_dataset("csv", data_files=data_files, cache_dir=cache_dir)
    ds = dataset[split]
    
    kept_identities = [col for col in ALL_IDENTITY_COLUMNS if col in ds.column_names]
    
    def process_batch(examples):
        is_toxic_batch = [int((target or 0) >= threshold) for target in examples['target']]
        result = {'is_toxic': is_toxic_batch}
        for col in kept_identities:
            result[col] = [float(val or 0.0) for val in examples[col]]
        return result

    ds = ds.map(
        process_batch, 
        batched=True,
        desc=f"Processing targets and identities for {split}"
    )
    
    keep_cols = ['id', 'comment_text', 'target', 'is_toxic'] + kept_identities
    ds = ds.select_columns([c for c in keep_cols if c in ds.column_names])
    
    return ds, kept_identities
