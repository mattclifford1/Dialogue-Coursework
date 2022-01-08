import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
import utils
from word_counts import get_unique_words_in_questions, get_spans_per_doc, count_questions_and_spans



train_data = utils.load_own_rc_data(split="train")
val_data = utils.load_own_rc_data(split="validation")

unique_words = get_unique_words_in_questions(train_data)
num_spans = get_spans_per_doc(train_data)

train_data = count_questions_and_spans(train_data, unique_words, num_spans)
val_data = count_questions_and_spans(val_data, unique_words, num_spans)
