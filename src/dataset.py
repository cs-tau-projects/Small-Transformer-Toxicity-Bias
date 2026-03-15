import pandas as pd
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

def download_and_prep_jigsaw(split='train', threshold=0.5):
    """
    Downloads the Jigsaw Unintended Bias dataset and prepares it for standard toxicity classification.
    Converts continuous toxicity and identity targets to binary indicators based on the threshold.
    """
    from .data_loader import get_jigsaw_dataset
    print(f"Loading split '{split}' of Jigsaw Unintended Bias dataset...")
    try:
        # Load from HuggingFace via custom loader
        ds = get_jigsaw_dataset(split)
    except Exception as e:
        print(f"Failed to load dataset. Error: {e}")
        raise e

    # Convert to pandas for easier vectorized manipulation
    df = ds.to_pandas()
    
    # Binarize the toxicity target
    df['is_toxic'] = (df['target'] >= threshold).astype(int)
    
    # Binarize identity columns, handling existing nans by filling with 0
    # Dynamically keep identities that actually exist in the dataframe
    kept_identities = []
    for col in ALL_IDENTITY_COLUMNS:
        if col in df.columns:
            df[col] = (df[col].fillna(0) >= threshold).astype(int)
            kept_identities.append(col)
            
    # Subselect columns to save memory
    keep_cols = ['id', 'comment_text', 'target', 'is_toxic'] + kept_identities
    df = df[keep_cols]
    
    return df

class JigsawDataset(Dataset):
    """
    PyTorch Dataset wrapper for the Jigsaw dataset.
    Given a dataframe (from download_and_prep_jigsaw) and a HuggingFace tokenizer,
    this yields tokenized inputs and labels, along with identity flags for evaluation.
    """
    def __init__(self, df: pd.DataFrame, tokenizer_name: str, max_length: int = 128):
        self.df = df.reset_index(drop=True)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        
        self.texts = self.df['comment_text'].tolist()
        self.labels = self.df['is_toxic'].tolist()
        
        # Discover dynamically which identity columns were kept
        self.identity_columns = [col for col in ALL_IDENTITY_COLUMNS if col in self.df.columns]
        self.identities = {col: self.df[col].tolist() for col in self.identity_columns}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        item = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }
        
        for col in self.identity_columns:
            item[col] = torch.tensor(self.identities[col][idx], dtype=torch.long)
            
        return item
