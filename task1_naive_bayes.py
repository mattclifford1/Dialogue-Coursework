import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
import utils
from word_counts import get_train_val
from tqdm import tqdm
from sklearn.metrics import accuracy_score

def train_clf(X_train, y_train):
    clf = BernoulliNB()
    clf.fit(X_train, y_train)
    return clf


if __name__ == '__main__':
    train_data = utils.load_own_rc_data(split="train")
    val_data = utils.load_own_rc_data(split="validation")

    print('Getting word count data')
    train, val = get_train_val(train_data, val_data)

    # train a binary classifier per span within a context
    clfs = {}
    print('Training')
    for context in tqdm(train.keys()):
        X = np.array(train[context]['X'])
        Y = np.array(train[context]['Y'])
        clfs[context] = {}
        for span in range(Y.shape[1]):
            clfs[context][span] = train_clf(X, Y[:, span])

    # validate training
    scores = []
    print('Evaluating')
    for context in tqdm(val.keys()):
        if val[context]['X'] == []:  # no examples for this context
            continue
        X = np.array(val[context]['X'])
        Y = np.array(val[context]['Y'])
        for span in range(Y.shape[1]):
            y_pred = clfs[context][span].predict(X)
            y_true = Y[:, span]
            acc = accuracy_score(y_true, y_pred)
            scores.append(acc)
    print('AVG score: ', sum(scores)/len(scores)*100, '% out of ', len(scores), ' examples')
