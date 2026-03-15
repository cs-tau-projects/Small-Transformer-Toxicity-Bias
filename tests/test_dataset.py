import pytest
from src.data_loader import get_jigsaw_dataset

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
        
    # Validate the data using filtering
    toxic_examples = train_data.filter(lambda x: x['target'] > 0).select(range(1))
    assert len(toxic_examples) > 0, "No toxic examples found in the dataset"
    
    toxic = toxic_examples[0]
    assert toxic['target'] > 0, "Target toxicity should ideally be greater than 0"
    
    # Check identity columns presence and type implicitly by accessing them
    identities = [
        'male', 'female', 'transgender', 'other_gender', 'heterosexual', 
        'homosexual_gay_or_lesbian', 'christian', 'jewish', 'muslim', 'hindu', 
        'buddhist', 'atheist', 'black', 'white', 'asian', 'latino', 
        'other_race_or_ethnicity', 'physical_disability', 
        'intellectual_or_learning_disability', 'psychiatric_or_mental_illness', 
        'other_disability'
    ]
    
    found_any = False
    for ident in identities:
        if ident in toxic and toxic[ident] is not None:
            found_any = True
            break
            
    assert found_any, "Expected at least some identity columns to be properly parsed"
