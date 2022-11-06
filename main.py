import spacy
import numpy as np
from datasets import load_dataset
from collections import Counter

def bigram(doc):
    result = list()
    sentence = list()

    for token in doc:
        if token.is_alpha:
            sentence.append(token)

    for wold in range(len(sentence) - 1):
        first_word = sentence[wold]
        second_word = sentence[wold + 1]
        elemnt = [first_word, second_word]
        result.append(elemnt)

    return result

def unigram(dataset):
    cnt = Counter()
    cnt_dict = dict()
    sentence = list()
    length = 0
    for text in dataset['text']:
        doc = nlp(text)
        for token in doc:
            if token.is_alpha:
                # sentence.append(token)
                length += 1
                if token.lemma_ in cnt_dict:
                    cnt_dict[token.lemma_] += 1
                else:
                    cnt_dict[token.lemma_] = 1
                cnt[token.lemma_] += 1
                if length % 100 == 0:
                    print(length)
    # for word in sentence:
    #     cnt[word] += 1
    return lambda x: cnt_dict[x] / length


def get_probability_unigram(sentence, lambda_model):
    p = 1
    for word in sentence:
        p *= lambda_model(word)
    return p


nlp = spacy.load("en_core_web_sm")
dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
dod_test = nlp("i want to make a cake ")
lambda_model_unigram = unigram(dataset)
print(get_probability_unigram(nlp("I have a house in the US"), lambda_model_unigram))
