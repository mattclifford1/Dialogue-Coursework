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

def get_train_val(train_data, val_data):
    dont_use_val_inds = get_inds_of_new_entries(train_data, val_data)
    unique_words = get_unique_words_in_questions(train_data)
    num_spans = get_spans_per_context(train_data)
    # set up data storage
    train = {}
    val = {}
    for context in num_spans.keys():
        train[context] = {'X':[], 'Y':[]}
        val[context] = {'X':[], 'Y':[]}
    # get train data
    for row in tqdm(range(len(train_data))):
        context = train_data['context'][row]
        answers = train_data['answers'][row]
        first_question = get_first_question(train_data['question'][row])
        # append in the format we want
        train[context]['X'].append(count_words(first_question, unique_words))
        train[context]['Y'].append(one_hot_spans(answers, num_spans[context]))
    # get val data
    for row in tqdm(range(len(val_data))):
        if row in dont_use_val_inds:
            continue
        context = val_data['context'][row]
        answers = val_data['answers'][row]
        first_question = get_first_question(val_data['question'][row])
        # append in the format we want
        val[context]['X'].append(count_words(first_question, unique_words))
        val[context]['Y'].append(one_hot_spans(answers, num_spans[context]))

    return train, val

if __name__ == '__main__':
    train_data = utils.load_own_rc_data(split="train")
    val_data = utils.load_own_rc_data(split="validation")

    get_train_val(train_data, val_data)


    # unique_words = get_unique_words_in_questions(train_data)
    # num_spans = get_spans_per_context(train_data)
    # print(num_spans)
    #
    # count_questions_and_spans(train_data, unique_words, num_spans)
