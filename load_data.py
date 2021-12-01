from datasets import load_dataset
import numpy as np

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


def get_doc_from_id(row=0):
    turn = 0
    # print the turn - dict containing info
    print('First Turn: ', dataset_dialogue['turns'][row][turn], '\n')
    # print doc_id - is the title of the relevant document
    id = dataset_dialogue['doc_id'][row]
    print('Dialogue doc_id: ', id)
    # confirm is the same as in the document dataset
    ind = dataset_document['doc_id'].index(id)
    print('Document doc_id: ', dataset_document['doc_id'][ind], '\n')
    # print span_id - the span of relevant text in the relevant document (just take the first span for now)
    span = 0
    sp_id = dataset_dialogue['turns'][row][turn]['references'][span]['sp_id']
    print('Document sp_id: ', dataset_document['spans'][ind][int(sp_id)], '\n')
    print('Document sp_id_title: ', dataset_document['spans'][ind][int(sp_id)]['title'])
    print('Document sp_id_text: ', dataset_document['spans'][ind][int(sp_id)]['text_sec'])

get_doc_from_id()
