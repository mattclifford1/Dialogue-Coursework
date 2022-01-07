import json
import os
import pandas as pd
from datasets import load_dataset

def load_own_rc_data(split='train'):
    if split == 'train':
        return pd.read_csv('doc2dial_rc_train.csv')
    elif split == 'validation':
        return pd.read_csv('doc2dial_rc_val.csv')
    else:
        raise ValueError('split needs to be train or validation not: '+str(split))

## A function to load a specific doc2dial dataset
def load_doc2dial_dataset(name='dialogue_domain', split='train'):
  cache_dir = "./data_cache"
  return load_dataset(
      "doc2dial",
      name=name,
      split=split,
      ignore_verifications=True,
      cache_dir=cache_dir,
  )

def save_and_test_preds(preds, file):
    # test on metrics
    with open(file, 'w') as outfile:
        json.dump(preds, outfile)
    cmd = 'python sharedtask_utils.py --task subtask1 --prediction_json '+file
    os.system(cmd)

def print_example_val_data():
    new_val_data = pd.read_csv('doc2dial_rc_val.csv')
    print(new_val_data.head())
    ## Example
    new_val_data_slice = new_val_data.sample(10, random_state=2)

    for i in range(len(new_val_data_slice)):
      print("QUESTION: ", new_val_data_slice["question"].iloc[i][5:])
      print("CONTEXT: ", new_val_data_slice['context'].iloc[i])
      print("ANSWERS: ", new_val_data_slice['answers'].iloc[i])
      print("SPANS: ", new_val_data_slice['spans'].iloc[i])
      print("DOMAIN: ", new_val_data_slice['domain'].iloc[i])
      # print("ANSWER:  ", train_predict_doc2vec(new_val_data_slice, i)["prediction_text"])
      print('==========================================')
      print('==========================================')
      print('==========================================')

def add_spans(sample, dialogues):
    dial_id, turn_id = sample['id'].split('_')

    dials = dialogues.filter(lambda ex: ex['dial_id'] == dial_id)[0]

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
    train_data = load_doc2dial_dataset(name="doc2dial_rc", split="train")
    val_data = load_doc2dial_dataset(name="doc2dial_rc", split="validation")
    print_example_val_data()
