import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
import utils
from word_counts import get_unique_words_in_questions, get_spans_per_doc, count_questions_and_spans


train_data = utils.load_own_rc_data(split="train")
val_data = utils.load_own_rc_data(split="validation")

train, val = get_train_val(train_data, val_data)
