
from datasets import load_dataset, Dataset
import pandas as pd
from tqdm import tqdm
# from utils import load_doc2dial_dataset

def load_doc2dial_dataset(name='dialogue_domain', split='train'):
  cache_dir = "./data_cache"

  return load_dataset(
      "doc2dial",
      name=name,
      split=split,
      ignore_verifications=True,
      cache_dir=cache_dir,
  )

def add_spans(sample):
    dial_id, turn_id = sample['id'].split('_')

    dial = train_dialogues.filter(lambda ex: ex['dial_id'] == dial_id)[0]

    doc_id = dial['doc_id']   ## corresponding document id
    span_ids = []             ## corresponding span ids
    for tr in dial['turns']:
        if tr['turn_id'] == int(turn_id)+1:
            for sp in tr['references']:
                span_ids.append(sp['sp_id'])

            break
    doc = documents.filter(lambda ex: ex['doc_id'] == doc_id)[0]

    spans = {}
    answer_spans = {}
    for span in doc['spans']:
        spans[span['id_sp']] = span['text_sp']
        if span['id_sp'] in span_ids:
            answer_spans[span['id_sp']] = span['text_sp']

    sample['spans'] = spans
    sample['answers']['spans'] = answer_spans

    return sample

if __name__ == '__main__':
    train_dialogues = load_doc2dial_dataset(name="dialogue_domain", split="train")
    val_dialogues = load_doc2dial_dataset(name="dialogue_domain", split="validation")

    documents = load_doc2dial_dataset(name="document_domain", split="train")

    train_data = load_doc2dial_dataset(name="doc2dial_rc", split="train")
    val_data = load_doc2dial_dataset(name="doc2dial_rc", split="validation")

    # new_train_data = pd.DataFrame(columns=['id', 'question', 'context', 'answers', 'spans',  'domain', 'title'])

    # for ex in tqdm(train_data):
    #     new_train_data = new_train_data.append(add_spans(ex, train_dialogues), ignore_index=True)

    # new_train_data.to_csv('doc2dial_rc_train.csv')
    # print(new_train_data.head())

    ### Update
    new_data_train = train_data.map(add_spans)
    print(new_data_train[0])
    new_train_data.to_csv('doc2dial_rc_train.csv')
