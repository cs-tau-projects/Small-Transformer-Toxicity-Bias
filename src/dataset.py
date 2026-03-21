import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

ALL_IDENTITY_COLUMNS = [
    "asian", "atheist", "bisexual", "black", "buddhist", "christian", "female", 
    "heterosexual", "hindu", "homosexual_gay_or_lesbian", "intellectual_or_learning_disability", 
    "jewish", "latino", "male", "muslim", "other_disability", "other_gender", 
    "other_race_or_ethnicity", "other_religion", "other_sexual_orientation", 
    "physical_disability", "psychiatric_or_mental_illness", "transgender", "white"
]

def download_and_prep_jigsaw(split='train', threshold=0.5, cache_dir=None):
    """
    Downloads the Jigsaw Unintended Bias dataset and prepares it for standard toxicity classification.
    Maintains continuous identity values for subgroup AUC calculation instead of binarizing.
    """
    from .data_loader import get_jigsaw_dataset
    print(f"Loading split '{split}' of Jigsaw Unintended Bias dataset...")
    try:
        # Load from HuggingFace via custom loader
        ds = get_jigsaw_dataset(split, cache_dir=cache_dir)
    except Exception as e:
        print(f"Failed to load dataset. Error: {e}")
        raise e

    # Discover dynamically which identity columns were kept
    kept_identities = [col for col in ALL_IDENTITY_COLUMNS if col in ds.column_names]

    # Process directly using Arrow rather than Pandas
    def process_batch(examples):
        # Binarize toxicity target
        is_toxic_batch = [int((target or 0) >= threshold) for target in examples['target']]
        
        # Ensure comment_text is always a string (sklearn/transformers crash on None)
        comments = [str(t) if t is not None else "" for t in examples["comment_text"]]
        
        # Keep identities as continuous, fill nas with 0.0
        result = {'is_toxic': is_toxic_batch, 'comment_text': comments}
        for col in kept_identities:
            result[col] = [float(val or 0.0) for val in examples[col]]
            
        return result

    print("Processing targets and identities (memory-mapped)...")
    ds = ds.map(
        process_batch, 
        batched=True,
        desc="Processing targets and identities"
    )

    # Subselect columns to save memory
    keep_cols = ['id', 'comment_text', 'target', 'is_toxic'] + kept_identities
    ds = ds.select_columns([c for c in keep_cols if c in ds.column_names])
    
    return ds, kept_identities

def tokenize_jigsaw_dataset(dataset, tokenizer_name: str, max_length: int = 128, cache_dir: str = None):
    """
    Tokenizes the dataset eagerly using HF's memory-mapped Arrow backend.
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, cache_dir=cache_dir)
    
    def tokenize_function(examples):
        # Note: None values are already handled by process_batch during prep, 
        # but the guard is kept here for safety in case dataset is loaded differently.
        return tokenizer(
            [str(t) if t is not None else "" for t in examples["comment_text"]],
            padding="max_length",
            truncation=True,
            max_length=max_length
        )
        
    print(f"Tokenizing dataset using {tokenizer_name} (memory-mapped)...")
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        desc="Tokenizing dataset"
    )
    
    return tokenized_dataset

class JigsawDataset(Dataset):
    """
    PyTorch Dataset wrapper for the Jigsaw dataset.
    Yields pre-tokenized inputs and evaluation targets.
    """
    def __init__(self, dataset, identity_columns):
        self.dataset = dataset
        self.identity_columns = identity_columns
        
        # Pre-calculate identity matrix to avoid repeated lookups during evaluation
        identities_list = []
        for col in self.identity_columns:
            identities_list.append(self.dataset[col])
        self.identity_matrix = np.array(identities_list).T

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        result = {
            'input_ids': torch.tensor(item['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(item['attention_mask'], dtype=torch.long),
            'label': torch.tensor(item['is_toxic'], dtype=torch.long)
        }
        
        # Package continuous identity probabilities into a single tensor for Subgroup AUC
        identity_probs = [float(item.get(col, 0.0)) for col in self.identity_columns]
        result['identity_probs'] = torch.tensor(identity_probs, dtype=torch.float32)
        
        return result
