"""
This script loads the Google Jigsaw Unintended Bias in Toxicity Classification
dataset from Hugging Face without storing it fully on disk, using the `streaming` feature.
"""
import sys
from datasets import load_dataset

def main():
    print("Loading Google Jigsaw Unintended Bias dataset in streaming mode...")
    
    try:
        # Load dataset in streaming mode which iterably downloads data items on the fly
        # Note: We use a mirror here since the official google dataset requires a manual Kaggle login/download
        dataset = load_dataset("james-burton/jigsaw_unintended_bias100K", streaming=True)
        
        # Access the training split as an iterator
        train_iter = iter(dataset['train'])
        
        print("Successfully loaded dataset iterator. Fetching 3 toxic examples:\n")
        
        # Print 3 examples where toxicity is not 0
        count = 0
        while count < 3:
            example = next(train_iter)
            if example.get('target') == 0:
                continue
                
            print(f"--- Example {count+1} ---")
            print(f"Text: {example.get('comment_text')}")
            print(f"Toxicity label (target): {example.get('target')}")
            
            # Define some common identity categories from this dataset
            identities = ['male', 'female', 'transgender', 'other_gender', 'heterosexual', 'homosexual_gay_or_lesbian', 'christian', 'jewish', 'muslim', 'hindu', 'buddhist', 'atheist', 'black', 'white', 'asian', 'latino', 'other_race_or_ethnicity', 'physical_disability', 'intellectual_or_learning_disability', 'psychiatric_or_mental_illness', 'other_disability']
            
            # Extract non-zero identity mentions
            found_identities = []
            for identity in identities:
                val = example.get(identity)
                if val is not None and val > 0:
                    found_identities.append(f"{identity} ({val:.2f})")
            
            if found_identities:
                print("Identity mentions:", ", ".join(found_identities))
            else:
                print("Identity mentions: None or Not annotated")
                
            print("\n")
            count += 1 

    except Exception as e:
        print(f"Error loading dataset: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
