import torch
from unittest.mock import patch
from src.data_loader import get_jigsaw_dataset
from src.dataset import download_and_prep_jigsaw, tokenize_jigsaw_dataset, JigsawDataset

def test_jigsaw_dataset_loading():
    """
    Tests that the Jigsaw Unintended Bias dataset correctly loads from the custom source.
    """
    train_data = get_jigsaw_dataset(split="train")
    
    assert train_data is not None, "Dataset returned None"
    assert len(train_data) > 0, "Dataset should not be empty"
    
    # Check that basic expected columns exist in the first item
    first_item = train_data[0]
    expected_columns = ['comment_text', 'target', 'male', 'female', 'black', 'white']
    for col in expected_columns:
        assert col in first_item, f"Expected column '{col}' missing from dataset"

@patch('src.data_loader.get_jigsaw_dataset')
def test_dataset_prep_and_tokenization(mock_get_jigsaw):
    """
    Tests the prep and tokenization pipeline locally.
    """
    # Create a small dataset to avoid running processing on the whole 1.8M rows in the test
    # Load raw locally (cached from the first test)
    raw_ds = get_jigsaw_dataset(split="train")
    small_ds = raw_ds.select(range(20))
    mock_get_jigsaw.return_value = small_ds
    
    ds, identities = download_and_prep_jigsaw(split="train")
    
    assert 'is_toxic' in ds.column_names
    assert len(identities) > 0
    assert 'target' in ds.column_names, "Should maintain continuous target"
    assert identities[0] in ds.column_names, "Should maintain continuous identities"
    
    # Test tokenization
    tok_ds = tokenize_jigsaw_dataset(ds, "bert-base-uncased", max_length=16)
    
    assert 'input_ids' in tok_ds.column_names
    assert 'attention_mask' in tok_ds.column_names
    
    # Test dataset wrapper
    pytorch_ds = JigsawDataset(tok_ds, identities)
    assert len(pytorch_ds) == 20
    
    item = pytorch_ds[0]
    
    assert 'input_ids' in item
    assert 'label' in item
    assert 'identity_probs' in item
    assert isinstance(item['input_ids'], torch.Tensor)
    assert isinstance(item['identity_probs'], torch.Tensor)
    assert len(item['identity_probs']) == len(identities)
    
    # Ensure float typing for probabilities
    assert item['identity_probs'].dtype == torch.float32
