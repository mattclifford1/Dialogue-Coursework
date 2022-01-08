'''
investigate data, look at errors in the data and how train/val differ
'''
import utils
from tqdm import tqdm


def count_column(data, col='context'):
    # get number of contexts
    count = {}
    for row in range(len(data)):
        entry = data[col][row]
        if entry in count.keys():
            count[entry] += 1
        else:
            count[entry] = 1
    # print('DOC COUNTS: ', doc_count)
    # print(col, ' COUNT: ', len(count.keys()))
    return count

def get_not_in_list(data, list_of_values, col='context'):
    '''
    find all the values of col not in list given
    '''
    new_entries = []
    inds = []
    for row in range(len(data)):
        entry = data[col][row]
        if entry not in list_of_values:
            inds.append(row)
            if entry not in new_entries:
                new_entries.append(entry)
    # print('number of new unique entries in ', col, ': ', len(new_entries))
    return new_entries, inds

def count_unique_spans(data):
    from word_counts import get_num_spans
    spans_count = {}
    for row in range(len(data)):
        span_str = data['spans'][row]
        spans = get_num_spans(span_str)
        if spans in spans_count.keys():
            spans_count[spans] += 1
        else:
            spans_count[spans] = 1
    # print('DOC COUNTS: ', doc_count)
    print('UNIQUE SPANS COUNT: ', len(spans_count.keys()))


def get_duplicated_titles(data):
    '''
    there are more titles than contexts, so we see if these are mistakes
    '''
    context_titles = {}
    duplicates = []
    for row in range(len(data)):
        context = data['context'][row]
        title = data['title'][row]
        if context in context_titles.keys():
            if context_titles[context] != title:
                duplicates.append([context_titles[context], title])
        else:
            context_titles[context] = title

    for dup in duplicates:
        print(dup)
        print(' ')
    print('DUPLICATES: ', len(duplicates))


def get_inds_of_new_entries(refernce_data, new_data, col='context'):
    '''
    returns list of indicies where the specified 'col' has entries in 'new_data'
    that don't occur in 'refernce_data'
    '''
    count = count_column(refernce_data, col)
    _, inds = get_not_in_list(new_data, count.keys(), col)
    # print('Percent diff: ', len(inds)/len(new_data)*100)
    return inds



if __name__ == '__main__':
    train_data = utils.load_own_rc_data(split="train")
    val_data = utils.load_own_rc_data(split="validation")

    # # get contexts not in training data but not validation
    # count_context = count_column(train_data, col='context')
    # new_entries, inds = get_not_in_list(val_data, count_context.keys(), col='context')

    print('seeing what contexts are in validation that arent in train')
    get_inds_of_new_entries(train_data, val_data)
    print(' ')
    print('seeing what contexts are in train that arent in validation')
    get_inds_of_new_entries(val_data, train_data)

    # count_unique_spans(train_data)
    # get_duplicated_titles(train_data)
