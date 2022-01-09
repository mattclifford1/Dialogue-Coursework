'''
get word counts for questions
with corresponding span occurance for use with NB classifiers
'''
import utils
from tqdm import tqdm
import ast
from data_investigation import get_inds_of_new_entries

def get_first_question(question_str):
    rm_user = question_str[5:] # get rid of starting user
    # tab char split the turns
    final_question = rm_user.split('\t')[0] # question in reverse ordering
    return final_question

def get_unique_words_in_questions(data):
    # get unique words in questions
    unique_words = {}
    count = 0
    for row in range(len(data)):
        questions = data['question'][row]
        first_question = get_first_question(questions)
        words = first_question.split(' ')
        for word in words:
            if word not in unique_words.keys(): # dont make duplicated
                unique_words[word] = count
                count += 1
    return unique_words

def count_words(count_str, unique_words):
    counts = [0]*len(unique_words.keys())
    for word in count_str.split(' '):
        if word in unique_words.keys():
            # add to the index represented by that word
            counts[unique_words[word]] += 1
    return counts

def get_unique_contexts(data):
    contexts = []
    for row in range(len(data)):
        doc = data['context'][row]
        if doc not in contexts:
            contexts.append(doc)
    return contexts

def get_spans_per_context(data):
    contexts = get_unique_contexts(data)
    # set up data storage
    spans_in_context = {}
    for context in contexts:
        spans_in_context[context] = 0
    # loop through to find the max of the spans
    for row in range(len(data)):
        context = data['context'][row]
        num_spans = get_num_spans(data['spans'][row])
        if num_spans > spans_in_context[context]:
            spans_in_context[context] = num_spans
    return spans_in_context

def get_num_spans(span_str):
    span_dict = ast.literal_eval(span_str)
    return len(span_dict.keys())

def one_hot_spans(answers_str, max_spans):
    spans = ast.literal_eval(answers_str)['spans'].keys()
    one_hot = [0]*max_spans
    for span in spans:
        one_hot[int(span)-1] = 1
    return one_hot

def count_questions_and_spans(data, unique_words, num_spans):
    '''
    convert data into dict of domains containing word count and span matricies
    '''
    # set up data storage
    count_data = {}
    for doc in num_spans.keys():
        count_data[doc] = {'X':[], 'Y':[]}
    for row in tqdm(range(len(data))):
        # get data
        doc = data['domain'][row]
        answers = data['answers'][row]
        first_question = get_first_question(data['question'][row])
        # append in the format we want
        count_data[doc]['X'].append(count_words(first_question, unique_words))
        count_data[doc]['Y'].append(one_hot_spans(answers, num_spans[doc]))


class word_counter():
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset
        self.unique_words = get_unique_words_in_questions(self.base_dataset)
        self.num_spans = get_spans_per_context(self.base_dataset)

    def convert_data(self, data):
        dont_use_inds = get_inds_of_new_entries(self.base_dataset, data)
        # set up data storage
        new_data = {}
        for context in self.num_spans.keys():
            new_data[context] = {'X':[], 'Y':[]}
        # get new_data data
        for row in tqdm(range(len(data))):
            if row in dont_use_inds:
                continue
            context = data['context'][row]
            answers = data['answers'][row]
            first_question = get_first_question(data['question'][row])
            # append in the format we want
            new_data[context]['X'].append(count_words(first_question, self.unique_words))
            new_data[context]['Y'].append(one_hot_spans(answers, self.num_spans[context]))
        return new_data

    def no_context_availble(self, data):
        return get_inds_of_new_entries(self.base_dataset, data)

    def convert_row(self, data, row):
        context = data['context'][row]
        answers = data['answers'][row]
        first_question = get_first_question(data['question'][row])
        X = count_words(first_question, self.unique_words)
        Y = one_hot_spans(answers, self.num_spans[context])
        return X, Y




if __name__ == '__main__':
    train_data = utils.load_own_rc_data(split="train")
    val_data = utils.load_own_rc_data(split="validation")

    counter = word_counter(train_data)
    # train = counter.convert_data(train_data)
    # val = counter.convert_data(val_data)

    skip_inds = counter.no_context_availble(val_data)
    for row in tqdm(range(len(val_data))):
        if row in skip_inds:
            continue
        docs = ast.literal_eval(val_data['spans'][row])
        context = val_data['context'][row]
        # X, Y = counter.convert_row(val_data, row)
        fq = get_first_question(val_data['question'][row])
        a = val_data['answers'][row]
        print(a)
        print(len(count_words(fq, counter.unique_words)))
        print(len(one_hot_spans(a, counter.num_spans[context])))

    # unique_words = get_unique_words_in_questions(train_data)
    # num_spans = get_spans_per_context(train_data)
    # print(num_spans)
    #
    # count_questions_and_spans(train_data, unique_words, num_spans)
