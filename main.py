import spacy
import numpy as np
from datasets import load_dataset
import datetime

def bigram(dataset):
    cnt_dict_tuples = dict()
    cnt_dict = dict()
    length = 0
    for text in dataset['text']:
        doc = nlp(text)
        cnt = 0
        i = 0
        while i < len(doc)-1:
            if doc[i].is_alpha:
                length += 1
                # if length % 100 == 0:
                #     print(length)
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
                # if length % 100 == 0:
                #     print(length)
    # for word in sentence:
    #     cnt[word] += 1
    return cnt_dict, length


def get_probability_unigram(sentence, cnt_dict, length):
    p = 0
    for token in sentence:
        if token.is_alpha:
            cnt = cnt_dict[token.lemma_]
            p *= cnt / length
    return p


def get_probability_bigram(sentence, cnt_dict, cnt_dict_tuples, length):
    p = 1
    cnt = 0
    num_of_pairs = 0
    i = 0
    while i < len(sentence) - 1:
        if sentence[i].is_alpha:
            if cnt == 0:
                cnt += 1
                num_of_pairs += 1
                cur_key = ('START', sentence[i].lemma_)
                if cur_key not in cnt_dict_tuples:
                    return 0, num_of_pairs
                p *= cnt_dict_tuples[cur_key] / cnt_dict['START']
            else:
                tmp = i
                while i < len(sentence) - 1 and not sentence[i + 1].is_alpha:
                    i += 1
                if i >= len(sentence) - 1:
                    break
                num_of_pairs += 1
                cur_key = (sentence[tmp].lemma_, sentence[i + 1].lemma_)
                if cur_key not in cnt_dict_tuples:
                    return 0, num_of_pairs
                p *= cnt_dict_tuples[cur_key] / cnt_dict[sentence[tmp].lemma_]
        i += 1
    return p, num_of_pairs


def complete_sentence_probability_bigram(sentence, cnt_dict, cnt_dict_tuples, length):
    p = 1
    cnt = 0
    num_of_pairs = 0
    i = 0
    while i < len(sentence) - 1:
        if sentence[i].is_alpha:
            if cnt == 0:
                cnt += 1
                num_of_pairs += 1
                cur_key = ('START', sentence[i].lemma_)
                if cur_key not in cnt_dict_tuples:
                    return 0, num_of_pairs
                p *= cnt_dict_tuples[cur_key] / cnt_dict['START']
            else:
                tmp = i
                while i < len(sentence) - 1 and not sentence[i + 1].is_alpha:
                    i += 1
                if i >= len(sentence) - 1:
                    break
                num_of_pairs += 1
                cur_key = (sentence[tmp].lemma_, sentence[i + 1].lemma_)
                if cur_key not in cnt_dict_tuples:
                    return 0, num_of_pairs
                p *= cnt_dict_tuples[cur_key] / cnt_dict[sentence[tmp].lemma_]
                last_index = i + 1
        i += 1
    max_word = ''
    max_prob = 0
    for text in dataset['text']:
        doc = nlp(text)
        for token in doc:
            if token.is_alpha:
                cur_key = (sentence[last_index].lemma_, token.lemma_)
                if cur_key not in cnt_dict_tuples:
                    continue
                new_prob = p * cnt_dict_tuples[cur_key] / cnt_dict[sentence[last_index].lemma_]
                if new_prob > max_prob:
                    max_prob = new_prob
                    max_word = token
    print(f"max word is {max_word} and max probability in log space is {np.log(max_prob)}")
    return max_prob, max_word

def get_probability_interpulation(sentence, cnt_dict_bigram, cnt_dict_tuples, cnt_dict_unigram, l1, l2, length):
    p = 1
    cnt = 0
    num_of_pairs = 0
    i = 0
    while i < len(sentence) - 1:
        if sentence[i].is_alpha:
            if cnt == 0:
                cnt += 1
                num_of_pairs += 1
                cur_key = ('START', sentence[i].lemma_)
                if cur_key not in cnt_dict_tuples:
                    p2 = 0
                else:
                    p2 = cnt_dict_tuples[cur_key] / cnt_dict_bigram['START']
                if sentence[i].lemma_ not in cnt_dict_unigram:
                    p1 = 0
                else:
                    p1 = cnt_dict_unigram[sentence[i].lemma_] / length

            else:
                tmp = i
                while i < len(sentence) - 1 and not sentence[i + 1].is_alpha:
                    i += 1
                if i >= len(sentence) - 1:
                    break
                num_of_pairs += 1
                cur_key = (sentence[tmp].lemma_, sentence[i + 1].lemma_)
                if cur_key not in cnt_dict_tuples:
                    p2 = 0
                else:
                    p2 = cnt_dict_tuples[cur_key] / cnt_dict_bigram[sentence[tmp].lemma_]
                if sentence[i + 1].lemma_ not in cnt_dict_unigram:
                    p1 = 0
                else:
                    p1 = cnt_dict_unigram[sentence[i + 1].lemma_] / length
            p *= l1 * p1 + l2 * p2
        i += 1
    return p, num_of_pairs


nlp = spacy.load("en_core_web_sm")
dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
print("load has finished")
print("2")
print("train unigram")
start = datetime.datetime.now().timestamp()
cnt_dict_unigram, length_unigram = unigram(dataset)
end = datetime.datetime.now().timestamp()
print(f"finished in {end - start} seconds")
print("train bigram")
start = datetime.datetime.now().timestamp()
cnt_dict_tuples, cnt_dict_bigram, length_bigram = bigram(dataset)
end = datetime.datetime.now().timestamp()
print(f"finished in {end - start} seconds")
# print(get_probability_bigram(nlp("I have a house in the"), cnt_dict, cnt_dict_tuples, length))
max_prob = -np.inf
max_word = ''
cnt = 0
start = datetime.datetime.now().timestamp()
# complete_sentence_probability_bigram(nlp(f"I have a house in "), cnt_dict_bigram, cnt_dict_tuples, length_bigram)
# for text in dataset['text']:
#     doc = nlp(text)
#     for token in doc:
#         if token.is_alpha:
#             # if cnt % 10000 == 0:
#             #     print(cnt)
#             #     print(max_word)
#             cnt += 1
#             cur_prob = get_probability_bigram(nlp(f"I have a house in {token}"), cnt_dict_bigram, cnt_dict_tuples, length_bigram)[0]
#             if cur_prob > max_prob:
#                 max_prob = cur_prob
#                 max_word = token
# print(f"max word is {max_word} and max probability in log space is {np.log(max_prob)}")
end = datetime.datetime.now().timestamp()
print(f"finished in {end - start} seconds")
print('3')
prop1, num_of_pairs1 = get_probability_bigram(nlp("Brad Pitt was born in Oklahoma"), cnt_dict_bigram, cnt_dict_tuples, length_bigram)
prop2, num_of_pairs2 = get_probability_bigram(nlp("The actor was born in USA"), cnt_dict_bigram, cnt_dict_tuples, length_bigram)
perplexity = np.exp(-(np.log(prop1) + np.log(prop2)) / (num_of_pairs1 + num_of_pairs2))

print(f"probability in log space for a is {np.log(prop1)} for b {np.log(prop2)} and the perplexity is {perplexity}")
print('4')
lambda1 = 1 / 3
lambda2 = 1 - lambda1
prop_inter1 = get_probability_interpulation(nlp("Brad Pitt was born in Oklahoma"), cnt_dict_bigram,
                                          cnt_dict_tuples, cnt_dict_unigram, lambda1, lambda2, length_unigram)[0]
prop_inter2 = get_probability_interpulation(nlp("The actor was born in USA"), cnt_dict_bigram,
                                          cnt_dict_tuples, cnt_dict_unigram, lambda1, lambda2, length_unigram)[0]

perplexity = np.exp(-(np.log(prop_inter1) + np.log(prop_inter2)) / (num_of_pairs1 + num_of_pairs2))
print(f"probability in log space for a is {np.log(prop_inter1)} for b {np.log(prop_inter2)} and the perplexity is {perplexity}")
