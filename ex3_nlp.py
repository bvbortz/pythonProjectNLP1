# -*- coding: utf-8 -*-
import numpy as np
from data_loader import SentimentTreeBank

def main():
    """
    Add Documentation here
    """
    # test_get_one_hot()
    test_average_one_hots()
    pass  # Replace Pass with Your Code


def test_average_one_hots():
    dataset = SentimentTreeBank()
    # get train set
    sent = dataset.get_train_set()[2]
    print(average_one_hots(dataset.sentences[2], get_word_to_ind(sent.text)))
    pass


def test_get_one_hot():
    hot = get_one_hot(5, 1)
    print(hot)


def get_one_hot(size, ind):
    """
    this method returns a one-hot vector of the given size, where the 1 is placed in the ind entry.
    :param size: the size of the vector
    :param ind: the entry index to turn to 1
    :return: numpy ndarray which represents the one-hot vector
    """
    one_hot = np.zeros(size)
    one_hot[ind] = 1
    return one_hot

def average_one_hots(sent, word_to_ind):
    """
    this method gets a sentence, and a mapping between words to indices, and returns the average
    one-hot embedding of the tokens in the sentence.
    :param sent: a sentence object.
    :param word_to_ind: a mapping between words to indices
    :return:
    """
    vec_size = len(word_to_ind)
    average_vec = np.zeros(vec_size)
    for word in sent.text:
        average_vec += get_one_hot(vec_size, word_to_ind[word])
    average_vec = average_vec / vec_size
    return average_vec

def get_word_to_ind(words_list):
    """
    this function gets a list of words, and returns a mapping between
    words to their index.
    :param words_list: a list of words
    :return: the dictionary mapping words to the index
    """
    word_set = set()
    word_to_index = dict()
    cur_index = 0
    for word in words_list:
        if word not in word_set:
            word_set.add(word)
            word_to_index[word] = cur_index
            cur_index += 1
    return word_to_index

if __name__ == '__main__':
    main()
