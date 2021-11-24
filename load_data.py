from datasets import load_dataset

split = "train"
cache_dir = "./data_cache"

dataset_dialogue = load_dataset(
    "doc2dial",
    name="dialogue_domain",  # this is the name of the dataset for the second subtask, dialog generation
    split=split,
    ignore_verifications=True,
    cache_dir=cache_dir,
)

dataset_document = load_dataset(
    "doc2dial",
    name="document_domain",  # this is the name of the dataset for the second subtask, dialog generation
    split=split,
    ignore_verifications=True,
    cache_dir=cache_dir,
)
