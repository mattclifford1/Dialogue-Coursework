import numpy as np
from sklearn.naive_bayes import BernoulliNB
import utils
from word_counts import word_counter
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import ast
import os
import pickle
from sklearn.utils.class_weight import compute_sample_weight

def train_clf(X_train, y_train):
    clf = BernoulliNB()
    sample_weighting = compute_sample_weight(class_weight='balanced', y=y_train)
    clf.fit(X_train, y_train, sample_weight=sample_weighting)
    return clf


if __name__ == '__main__':
    train_data = utils.load_own_rc_data(split="train")
    val_data = utils.load_own_rc_data(split="validation")

    print('Getting word count data')
    counter = word_counter(train_data)
    train = counter.convert_data(train_data)
    val = counter.convert_data(val_data)

    # train a binary classifier per span within a context
    print('Training')
    train_path = 'NB_classifiers.pickle'
    if not os.path.exists(train_path):
        clfs = {}
        for context in tqdm(train.keys()):
            X = np.array(train[context]['X'])
            Y = np.array(train[context]['Y'])
            clfs[context] = {}
            for span in range(Y.shape[1]):
                clfs[context][span] = train_clf(X, Y[:, span])
        # with open(train_path, 'wb') as file:
        #     pickle.dump(clfs, file)
    else:
        print('Loading model: ', train_path)
        with open(train_path, 'rb') as file:
            clfs = pickle.load(file)

    # validate training
    preds = []
    print('Evaluating')
    skip_inds = counter.no_context_availble(val_data)
    for row in tqdm(range(len(val_data))):
        if row in skip_inds:
            continue
        docs = ast.literal_eval(val_data['spans'][row])
        context = val_data['context'][row]
        X, Y = counter.convert_row(val_data, row)
        X = np.array(X)
        # get predicted spans for current X
        pred_spans = []
        for span in range(len(Y)):
            y_pred = clfs[context][span].predict(X.reshape(1, -1))
            if y_pred[0] == 1:
                pred_spans.append(span+1)
        # build text spans
        text = ''
        for i in pred_spans:
            text += docs.get(str(i), '')
        if len(pred_spans) == 0:
            no_ans_prob = 1
        else:
            no_ans_prob = 0
        preds.append({"id":val_data["id"][row], "prediction_text":text, "no_answer_probability":no_ans_prob})
    utils.save_and_test_preds(preds, 'predictions_subtask1_NB.json')
