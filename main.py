import spacy
import numpy as np
from datasets import load_dataset
from collections import Counter

def bigram(dataset):
    result = list()
    cnt_dict_tuples = dict()
    cnt_dict = dict()
    sentence = list()
    length = 0
    for text in dataset['text']:
        doc = nlp(text)
        cnt = 0
        i = 0
        while i < len(doc)-1:
            if doc[i].is_alpha:
                length += 1
                if length % 100 == 0:
                    print(length)
                if cnt == 0:
                    cnt += 1
                    cur_key = ('START', doc[i].lemma_)
                    update_dict(cnt_dict_tuples, cur_key)
                    update_dict(cnt_dict, 'START')
                else:
                    tmp = i
                    while i < len(doc)-1 and not doc[i+1].is_alpha:
                        i += 1
                    if i >= len(doc)-1:
                        break
                    cur_key = (doc[tmp].lemma_, doc[i+1].lemma_)
                    update_dict(cnt_dict_tuples, cur_key)
                    update_dict(cnt_dict, doc[tmp].lemma_)
                i += 1
    return cnt_dict_tuples, cnt_dict, length


def update_dict(cnt_dict, cur_key):
    if cur_key in cnt_dict:
        cnt_dict[cur_key] += 1
    else:
        cnt_dict[cur_key] = 1


def unigram(dataset):
    cnt_dict = dict()
    length = 0
    for text in dataset['text']:
        doc = nlp(text)
        for token in doc:
            if token.is_alpha:
                # sentence.append(token)
                length += 1
                update_dict(cnt_dict, token.lemma_)
                if length % 100 == 0:
                    print(length)
    # for word in sentence:
    #     cnt[word] += 1
    return cnt_dict, length


def get_probability_unigram(sentence, cnt_dict, length):
    p = 1
    for token in sentence:
        if token.is_alpha:
            cnt = cnt_dict[token.lemma_]
            p += np.log(cnt / length)
    return p


def get_probability_bigram(sentence, cnt_dict, cnt_dict_tuples, length):
    p = 1
    cnt = 0
    i = 0
    while i < len(sentence) - 1:
        if sentence[i].is_alpha:
            if cnt == 0:
                cnt += 1
                cur_key = ('START', sentence[i].lemma_)
                p *= cnt_dict_tuples[cur_key] / cnt_dict['START']
            else:
                tmp = i
                while i < len(sentence) - 1 and not sentence[i + 1].is_alpha:
                    i += 1
                if i >= len(sentence) - 1:
                    break
                cur_key = (sentence[tmp].lemma_, sentence[i + 1].lemma_)
                p += np.log(cnt_dict_tuples[cur_key] / sentence[tmp].lemma_)
        i += 1
    return p


nlp = spacy.load("en_core_web_sm")
dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
cnt_dict_tuples, cnt_dict, length = bigram(dataset)
print(get_probability_bigram(nlp("I have a house in the"), cnt_dict, cnt_dict_tuples, length))
