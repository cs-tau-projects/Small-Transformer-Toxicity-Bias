"""
This script loads the Google Jigsaw Unintended Bias in Toxicity Classification
dataset from Hugging Face.
"""
import getpass
from datasets import load_dataset
from src.data_utils import get_hf_token

def get_jigsaw_dataset(split="train", cache_dir=None):
    """
    Loads the Google Jigsaw Unintended Bias dataset.
    Returns the Hugging Face Dataset object.
    """
    print(f"Loading '{split}' split of Google Jigsaw Unintended Bias dataset...")
    
    if cache_dir is None:
        try:
            username = getpass.getuser()
        except Exception:
            username = "unknown_user"
            
        cluster_base = "/vol/joberant_nobck/data/NLP_368307701_2526a"
        if os.path.exists(cluster_base):
            cache_dir = f"{cluster_base}/{username}/.cache/huggingface"
        else:
            cache_dir = "./.hf_cache"
            
    print(f"Using cache directory: {cache_dir}")
    
    try:
        # Load the raw csv.gz from a community mirror to bypass Kaggle manual download
        # and avoid broken dataset builder scripts.
        data_files = {}
        if split == "train":
            data_files["train"] = "hf://datasets/shuttie/jigsaw-unintended-bias/data/train.csv.gz"
        elif split == "test":
            data_files["test"] = "hf://datasets/shuttie/jigsaw-unintended-bias/data/test_private_expanded.csv.gz"
            
        dataset = load_dataset("csv", data_files=data_files, cache_dir=cache_dir, token=get_hf_token())
        
        # Access the dataset split
        split_data = dataset[split]
        
        print(f"Successfully loaded dataset split '{split}'. Total examples: {len(split_data)}")
        return split_data

    except Exception as e:
        print(f"Error loading dataset: {e}", file=sys.stderr)
        raise e

if __name__ == "__main__":
    get_jigsaw_dataset()
