import getpass
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments

def get_model_pair(model_name: str, num_labels: int = 2):
    """
    Returns a 'Raw' model (pre-trained weights, no fine-tuning) and a 'Fine-tuned' model placeholder.
    """
    # Set cache directory for TAU storage efficiency
    user_name = getpass.getuser()
    cache_dir = f"/vol/joberant_nobck/data/NLP_368307701_2526a/{user_name}/huggingface_cache"
    
    # Load the raw model
    raw_model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        cache_dir=cache_dir
    )
    

    # Placeholder for the fine-tuned model that will be populated after training
    finetuned_model = None
    
    return raw_model, finetuned_model

def train_model(model, train_dataset, eval_dataset, output_dir: str, **kwargs):
    """
    Trains the model using the Hugging Face Trainer API.
    Incorporates safety mechanisms (save_total_limit, load_best_model_at_end) for TAU storage concern.
    """
    # Default arguments to protect TAU storage
    training_args_dict = {
        "output_dir": output_dir,
        "save_total_limit": 2,
        "load_best_model_at_end": True,
        "eval_strategy": "epoch",
        "save_strategy": "epoch"
    }
    
    # Update with custom user args
    training_args_dict.update(kwargs)
    
    training_args = TrainingArguments(**training_args_dict)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset
    )
    
    trainer.train()
    
    # Returns the fine-tuned model (best model loaded due to load_best_model_at_end=True)
    return trainer.model
