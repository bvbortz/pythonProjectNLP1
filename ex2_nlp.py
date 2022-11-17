# -*- coding: utf-8 -*-
from nltk.corpus import brown
import nltk
import numpy as np
import spacy


def clean_tag(tag):
    minus_find = tag.find('-')
    plus_find = tag.find('+')
    if minus_find != -1 and plus_find != -1:
        tag = tag[:min(plus_find, minus_find)]
    elif minus_find != -1 or plus_find != -1:
        tag = tag[:max(plus_find, minus_find)]
    return tag

def update_2d_dict(dict, first_key, second_key):
    if first_key in dict:
        if second_key in dict[first_key]:
            dict[first_key][second_key] += 1
        else:
            dict[first_key][second_key] = 1
    else:
        dict[first_key] = {second_key: 1}

def update_1d_dict(dict, key):
    if key in dict:
        dict[key] += 1
    else:
        dict[key] = 1


class Baseline_tagger:
    def __init__(self):
        self.cnt_dict = dict()

    def train(self, train):
        for sentence in train:
            for word, tag in sentence:
                tag = self.clean_tag(tag)
                if word in self.cnt_dict:
                    if tag in self.cnt_dict[word]:
                        self.cnt_dict[word][tag] += 1
                    else:
                        self.cnt_dict[word][tag] = 1
                else:
                    self.cnt_dict[word] = {tag: 1}

    def clean_tag(self, tag):
        minus_find = tag.find('-')
        plus_find = tag.find('+')
        if minus_find != -1 and plus_find != -1:
            tag = tag[:min(plus_find, minus_find)]
        elif minus_find != -1 or plus_find != -1:
            tag = tag[:max(plus_find, minus_find)]
        return tag

    def predict(self, test):
        tag_predicted = list()
        for sentence in test:
            for word, tag in sentence:
                tag = self.clean_tag(tag)
                if word in self.cnt_dict:
                    tag_predicted.append(max(self.cnt_dict[word], key=self.cnt_dict[word].get))
                else:
                    tag_predicted.append('NN')
        return tag_predicted

    def accuracy(self, test):
        cnt_correct_known = 0
        cnt_correct_unknown = 0
        cnt_known = 0
        cnt_total = 0
        for sentence in test:
            for word, tag in sentence:
                tag = self.clean_tag(tag)
                if word in self.cnt_dict:
                    cnt_known += 1
                    if max(self.cnt_dict[word], key=self.cnt_dict[word].get) == tag:
                        cnt_correct_known += 1
                else:
                    if 'NN' == tag:
                        cnt_correct_unknown += 1
                cnt_total += 1
        return cnt_correct_known / cnt_known, \
               cnt_correct_unknown / (cnt_total - cnt_known), \
               (cnt_correct_unknown + cnt_correct_known) / cnt_total

class bigram_tag_and_prev_tag:
    def __init__(self):
        self.bi_dict = dict()
        self.cnt_dict = dict()
    def train(self, train):
        for sentence in train:
            for i, (word, tag) in enumerate(sentence):
                tag = clean_tag(tag)
                if i == 0:  # first tag for each sentence is 'START'
                    first_tag = "START"
                    continue
                second_tag = tag
                update_2d_dict(self.bi_dict, first_tag, second_tag)
                update_1d_dict(self.cnt_dict, first_tag)
                if i == len(sentence) - 1:
                    update_2d_dict(self.bi_dict, second_tag, "STOP")
                    update_1d_dict(self.cnt_dict, second_tag)
                first_tag = tag

    def bigram_prob(self, tag, prev_tag):
        if prev_tag not in self.bi_dict:
            return 0
        if tag not in self.bi_dict[prev_tag]:
            return 0
        return self.bi_dict[prev_tag][tag] / self.cnt_dict[prev_tag]


class bigram_word_given_tag:
    def __init__(self):
        self.bi_dict = dict()
        self.cnt_dict = dict()

    def train(self, train):
        for sentence in train:
            for i, (word, tag) in enumerate(sentence):
                tag = clean_tag(tag)
                update_2d_dict(self.bi_dict, tag, word)
                update_1d_dict(self.cnt_dict, tag)

    def bigram_prob(self, tag, prev_tag):
        if prev_tag not in self.bi_dict:
            return 0
        if tag not in self.bi_dict[prev_tag]:
            return 0
        return self.bi_dict[prev_tag][tag] / self.cnt_dict[prev_tag]


class bigram_HMM:
    def __init__(self):
        self.word_given_tag = bigram_word_given_tag()
        self.tag_given_prev_tag = bigram_tag_and_prev_tag()

    def train(self, train):
        self.tag_given_prev_tag.train(train)
        self.word_given_tag.train(train)

    def get_keys(self, k):
        if k == 0:
            return ["START"]
        else:
            return self.word_given_tag.cnt_dict.keys()

    def predict(self, test):
        pi_dict = list()
        bp_dict = list()
        pi_dict.append("START")
        for sentence in test:
            for i, (word, tag) in enumerate(sentence):
                pi_dict.append(dict())
                bp_dict.append(dict())
                for v in self.get_keys(i+1):
                    max_num = -1
                    for w in self.get_keys(i):
                        cur_val = pi_dict[i][w] * self.tag_given_prev_tag.bigram_prob(v, w) * \
                            self.word_given_tag.bigram_prob(word, v)
                        if cur_val > max_num:
                            max_num = cur_val
                            max_arg = w
                    pi_dict[i][v] = max_num
                    bp_dict[i][v] = max_arg
            predicted_tags = [""]*len(sentence)
            max_num = -1
            for v in self.get_keys(len(sentence)):
                cur_val = pi_dict[len(sentence)-1][v] * self.tag_given_prev_tag.bigram_prob("STOP", v)
                if cur_val > max_num:
                    max_num = cur_val
                    max_arg = v
            predicted_tags[len(sentence)-1] = max_arg
            for k in range(len(sentence)-2, -1, -1):
                predicted_tags[k] = bp_dict[k+1][predicted_tags[k+1]]
            print(f"predicted tags are {predicted_tags}")




                    


nltk.download('brown')
news_text = brown.tagged_sents(categories='news')
train_test_ratio = int(len(news_text) * 0.9)
test = news_text[train_test_ratio:]
train = news_text[:train_test_ratio]
bs = bigram_HMM()
bs.train(train)
bs.predict(test)
