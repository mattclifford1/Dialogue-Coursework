from datasets import load_dataset, Dataset
import pandas as pd
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
import nltk
import ast
from tqdm import tqdm
import utils

nltk.download('punkt')

def train_predict_doc2vec(dataset, index):
  instance = dataset.iloc[index]
  docs = ast.literal_eval(instance['spans'])

  tokenized_docs = {}
  for i, d in docs.items():
      tokenized_docs[int(i)] = word_tokenize(d.lower())

  tagged_data = []
  for i, d in tokenized_docs.items():
    tagged_data.append(TaggedDocument(d, [i]))

  model = Doc2Vec(tagged_data, vector_size=80, window=5, min_count=1, workers=-1, epochs = 100)

  test_doc = instance['question'][5:]

  predictions = model.docvecs.most_similar(positive=[model.infer_vector(test_doc)], topn=30)
  # print(predictions)

  ids = []
  count = 0
  sum_score = 0
  pred = 0
  tol_zone = 5
  while True:

    id, score = predictions[pred]
    ids.append(id)
    count += 1
    sum_score += score

    pred += 1
    next_id, next_score = predictions[pred]

    possible_next_ids = list(range(id-tol_zone, id+tol_zone+1))
    # print(possible_next_ids)
    if next_id not in possible_next_ids:
      break

  sorted_ids = sorted(ids)
  text = ''
  for i in sorted_ids:
    text += docs.get(str(i), '')

  final_score = sum_score / count
  return {"id":instance["id"], "prediction_text":text, "no_answer_probability":1.0-final_score}

if __name__ == '__main__':
    # get data
    new_val_data = pd.read_csv('doc2dial_rc_val.csv')
    # predict on all val data
    preds = []
    new_val_data_slice = new_val_data[:]
    for i in tqdm(range(len(new_val_data_slice))):
      preds.append(train_predict_doc2vec(new_val_data_slice, i))

    utils.save_and_test_preds(preds, 'predictions_subtask1_cosine_simple.json')
