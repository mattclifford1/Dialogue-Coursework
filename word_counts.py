'''
get word counts for questions
with corresponding span occurance for use with NB classifiers
'''
import utils
from tqdm import tqdm
import ast

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

def get_doc_types(data):
    docs = []
    for row in range(len(data)):
        doc = data['domain'][row]
        if doc not in docs:
            docs.append(doc)
    return docs

def get_spans_per_doc(data):
    docs = get_doc_types(data)
    # set up data storage
    spans_in_doc = {}
    for doc in docs:
        spans_in_doc[doc] = 0
    # loop through to find the max of the spans
    for row in range(len(data)):
        doc = data['domain'][row]
        num_spans = get_num_spans(data['spans'][row])
        if num_spans > spans_in_doc[doc]:
            spans_in_doc[doc] = num_spans
    return spans_in_doc

def get_num_spans(span_str):
    span_dict = ast.literal_eval(span_str)
    return len(span_dict.keys())

def one_hot_spans(answers_str, max_spans):
    spans = ast.literal_eval(answers_str)['spans'].keys()
    one_hot = [0]*max_spans
    for span in spans:
        if max_spans < int(span): # some val spans are bigger?
            one_hot[int(span)-1] += 1
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


if __name__ == '__main__':
    train_data = utils.load_own_rc_data(split="train")

    unique_words = get_unique_words_in_questions(train_data)
    num_spans = get_spans_per_doc(train_data)

    count_questions_and_spans(train_data, unique_words, num_spans)
