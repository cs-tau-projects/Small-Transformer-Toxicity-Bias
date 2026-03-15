"""
This script loads the Google Jigsaw Unintended Bias in Toxicity Classification
dataset from Hugging Face without storing it fully on disk, using the `streaming` feature.
"""
import sys
from datasets import load_dataset

def get_jigsaw_dataset(split="train"):
    """
    Loads the Google Jigsaw Unintended Bias dataset.
    Returns the Hugging Face Dataset object.
    """
    print(f"Loading '{split}' split of Google Jigsaw Unintended Bias dataset...")
    
    try:
        # Load the raw csv.gz from a community mirror to bypass Kaggle manual download
        # and avoid broken dataset builder scripts.
        data_files = {}
        if split == "train":
            data_files["train"] = "hf://datasets/shuttie/jigsaw-unintended-bias/data/train.csv.gz"
        elif split == "test":
            data_files["test"] = "hf://datasets/shuttie/jigsaw-unintended-bias/data/test_private_expanded.csv.gz"
            
        dataset = load_dataset("csv", data_files=data_files)
        
        # Access the dataset split
        split_data = dataset[split]
        
        print(f"Successfully loaded dataset split '{split}'. Total examples: {len(split_data)}")
        return split_data

    except Exception as e:
        print(f"Error loading dataset: {e}", file=sys.stderr)
        raise e

if __name__ == "__main__":
    get_jigsaw_dataset()
