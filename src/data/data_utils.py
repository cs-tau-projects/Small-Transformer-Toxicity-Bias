import getpass
import os

from datasets import load_dataset
from huggingface_hub import get_token

ALL_IDENTITY_COLUMNS = [
    "asian",
    "atheist",
    "bisexual",
    "black",
    "buddhist",
    "christian",
    "female",
    "heterosexual",
    "hindu",
    "homosexual_gay_or_lesbian",
    "intellectual_or_learning_disability",
    "jewish",
    "latino",
    "male",
    "muslim",
    "other_disability",
    "other_gender",
    "other_race_or_ethnicity",
    "other_religion",
    "other_sexual_orientation",
    "physical_disability",
    "psychiatric_or_mental_illness",
    "transgender",
    "white",
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


def get_hf_token():
    """
    Returns the Hugging Face token.
    Checks:
    1. Environment variable HF_TOKEN (via load_dotenv if available)
    2. Local cache (via huggingface_hub.get_token)
    """
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    return get_token()

