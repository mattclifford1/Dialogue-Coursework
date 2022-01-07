from datasets import load_dataset
import numpy as np
import utils

split = "train"
cache_dir = "./data_cache"


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

if __name__ == '__main__':
    train_data = utils.load_doc2dial_dataset(name="doc2dial_rc", split="train")
    '''
    train_data has keys 'id', 'title', 'context', 'question', 'answers', 'domain'
    'id': id of the data_entry
    'title': title of the context
    'context': text of document
    'question': full dialogue between user and agent up until relevant question
    'answers': text response
    'domain': relevant document domain
    '''
    print(train_data)
