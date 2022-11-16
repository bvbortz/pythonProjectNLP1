# -*- coding: utf-8 -*-
from nltk.corpus import brown
import nltk
import numpy as np
import spacy

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
        cnt_correct = 0
        cnt_total = 0
        for sentence in test:
            for word, tag in sentence:
                tag = self.clean_tag(tag)
                if word in self.cnt_dict:
                    if max(self.cnt_dict[word], key=self.cnt_dict[word].get) == tag:
                        cnt_correct += 1
                else:
                    if 'NN' == tag:
                        cnt_correct += 1
                cnt_total += 1
        return cnt_correct / cnt_total


class bigram_HMM:
    def __init__(self):
        self.prev_tag_and_tag = dict()
        self.word_and_tag = dict()
        self.baseline = Baseline_tagger()

    def train(self, train):
        self.baseline.train(train)


nltk.download('brown')
news_text = brown.tagged_sents(categories='news')
train_test_ratio = int(len(news_text) * 0.9)
test = news_text[train_test_ratio:]
train = news_text[:train_test_ratio]
bs = Baseline_tagger()
bs.train(train)
print(bs.accuracy(test))
