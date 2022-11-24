# -*- coding: utf-8 -*-
import re

from nltk.corpus import brown
import nltk
import numpy as np

def map_pseudo_words(dataset):
    map_dict = dict()
    cnt_dict = dict()
    for sentence in dataset:
        for word, tag in sentence:
            update_1d_dict(cnt_dict, word)
    cnt = 0
    cnt_total = 0
    for word in cnt_dict:
        if cnt_dict[word] < 5:
            cnt_total += 1
            # print(word)
            if word[-3:] == 'ing':
                map_dict[word] = "present_verb"
            elif word[-2:] == "ed":
                map_dict[word] = "past_verb"
            elif word[-4:] == "tion":
                map_dict[word] = "inter_noun"
            elif word[-3:] == "ful":
                map_dict[word] = "ful_suffix"
            elif word[-3:] == "ive":
                map_dict[word] = "ive_suffix"
            elif word[-3:] == "ity":
                map_dict[word] = "ity_suffix"
            elif word[-3:] == "nce":
                map_dict[word] = "nce_suffix"
            elif word[-3:] == "n't":
                map_dict[word] = "negative_suffix"
            elif word[-2:] == "ly":
                map_dict[word] = "ly_suffix"
            elif word[-2:] == "er":
                map_dict[word] = "er_suffix"
            elif word[-2:] == "or":
                map_dict[word] = "or_suffix"
            elif word[-2:] == "ic":
                map_dict[word] = "ic_suffix"
            elif word[-2:] == "al":
                map_dict[word] = "al_suffix"
            elif word[-2:] == "'s" or word[-2:] == "s'":
                map_dict[word] = "possessive_suffix"
            elif bool(re.search(r'\d', word)):
                if word.isdigit() and len(word) == 4:
                    map_dict[word] = "four_digit_num"
                elif word[-2:] == "th" or word[-2:] == "st" or word[-2:] == "nd" or word[-2:] == "nd":
                    map_dict[word] = "digit_and_suffix"
                elif word.isdigit() and len(word) == 2:
                    map_dict[word] = "two_digit_num"
                elif word.isupper:
                    map_dict[word] = "upper_and_digit"
                elif word.find(',') != -1:
                    map_dict[word] = "comma_and_digit"
                elif word.find('-') != -1:
                    map_dict[word] = "dash_and_digit"
                elif word.find('/') != -1:
                    map_dict[word] = "slash_and_digit"
                elif word.find('.') != -1:
                    map_dict[word] = "dot_and_digit"
                elif word.find("$") != -1:
                    map_dict[word] = "dollar_and_digit"
                else:
                    map_dict[word] = "other_num"
            elif word[-1] == '.' or word.isupper():
                map_dict[word] = "acronym"
            elif word[-1] == 's':
                map_dict[word] = "multiple"
            elif word.find("-") != -1:
                map_dict[word] = "double_word"
            elif word[0].isupper():
                map_dict[word] = "init_upper"
            else:
                map_dict[word] = "other"
                cnt += 1
    print(f"the were {cnt} words mapped to other")
    print(f"the were {cnt_total} words mapped")
    return map_dict


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

def count_true(predicted_tags, true_tags):
    if len(predicted_tags) != len(true_tags):
        print("watch the length")
    cnt = 0
    for i in range(min(len(predicted_tags), len(true_tags))):
        if predicted_tags[i] == true_tags[i]:
            cnt += 1
    return cnt

class Baseline_tagger:
    def __init__(self):
        self.cnt_dict = dict()

    def train(self, train):
        for sentence in train:
            for word, tag in sentence:
                tag = clean_tag(tag)
                update_2d_dict(self.cnt_dict, word, tag)

    def predict(self, test):
        tag_predicted = list()
        for sentence in test:
            for word, tag in sentence:
                tag = clean_tag(tag)
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
                tag = clean_tag(tag)
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
    def __init__(self, map_dict=dict()):
        self.bi_dict = dict()
        self.cnt_dict = dict()
        self.map_dict = map_dict
    def train(self, train):
        for sentence in train:
            for i, (word, tag) in enumerate(sentence):

                tag = clean_tag(tag)
                if i == 0:  # first tag for each sentence is 'START'
                    first_tag = "START"
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
    def __init__(self, delta=0, map_dict=dict()):
        self.bi_dict = dict()
        self.cnt_dict = dict()
        self.all_words = set()
        self.delta = delta
        self.map_dict = map_dict

    def train(self, train):
        for sentence in train:
            for i, (word, tag) in enumerate(sentence):
                if word in self.map_dict:
                    word = self.map_dict[word]
                self.all_words.add(word)
                tag = clean_tag(tag)
                update_2d_dict(self.bi_dict, tag, word)
                update_1d_dict(self.cnt_dict, tag)

    def bigram_prob(self, word, tag):
        if word in self.map_dict:
            word = self.map_dict[word]
        if tag not in self.bi_dict:
            return 0
        if word not in self.all_words and tag == "NN":
            return 1
        if word not in self.bi_dict[tag]:
            return 0
        return (self.bi_dict[tag][word] + self.delta) / (self.cnt_dict[tag] + len(self.all_words))



class bigram_HMM:
    def __init__(self, delta=0, map_dict=dict()):
        self.baseline = Baseline_tagger()
        self.word_given_tag = bigram_word_given_tag(delta=delta, map_dict=map_dict)
        self.tag_given_prev_tag = bigram_tag_and_prev_tag()
        self.map_dict = map_dict

    def train(self, train):
        self.baseline.train(train)
        self.tag_given_prev_tag.train(train)
        self.word_given_tag.train(train)

    def get_keys(self, k):
        if k == 0:
            return ["START"]
        else:
            return self.word_given_tag.cnt_dict.keys()

    def predict(self, test):
        cnt = 0
        cnt_total = 0
        cnt_accurate = 0
        for sentence in test:
            pi_dict = list()
            bp_dict = list()
            pi_dict.append({"START": 1})
            bp_dict.append({"START": 1})
            true_tags = list()
            for i, (word, tag) in enumerate(sentence):
                tag = clean_tag(tag)
                true_tags.append(tag)
                pi_dict.append(dict())
                bp_dict.append(dict())
                for v in self.get_keys(i+1):
                    # if word not in self.baseline.cnt_dict:
                    #     pi_dict[i + 1][v] = 1
                    #     bp_dict[i + 1][v] = "NN"
                    max_num = -1
                    for w in self.get_keys(i):
                        cur_val = pi_dict[i][w] * self.tag_given_prev_tag.bigram_prob(v, w) * \
                            self.word_given_tag.bigram_prob(word, v)
                        if cur_val > max_num:
                            max_num = cur_val
                            max_arg = w
                    pi_dict[i+1][v] = max_num
                    bp_dict[i+1][v] = max_arg
            if len(sentence) <= 1:
                continue
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
            print(f"true tags are {true_tags}")
            print(f"cnt is {cnt}")
            cnt_accurate += count_true(predicted_tags[1:], true_tags)
            cnt_total += len(true_tags)
            cnt += 1
        return cnt_accurate / cnt_total
    def wiki_predict(self, test):
        all_states = list(self.get_keys(2))
        states = list(self.get_keys(2))
        pred_and_true_list = list()
        states.append("START")
        cnt_correct_known = 0
        cnt_correct_unknown = 0
        cnt_known = 0
        cnt_total = 0
        for sentence in test:
            words = list()
            true_tags = list()
            if len(sentence) <= 1:
                continue
            t1 = np.zeros((len(states), len(sentence)+1))
            t2 = np.zeros((len(states), len(sentence) + 1))
            t1[len(states)-1, 0] = 1
            for i, (word, tag) in enumerate(sentence):
                tag = clean_tag(tag)
                if tag not in self.word_given_tag.cnt_dict:
                    all_states.append(tag)
                if word in self.map_dict:
                    word = self.map_dict[word]
                words.append(word)
                tag = clean_tag(tag)
                true_tags.append(tag)
                max_val = -1
                for j in range(len(states)):
                    for k in range(len(states)):
                        cur_val = t1[k, i] * self.tag_given_prev_tag.bigram_prob(states[j], states[k])
                        if cur_val > max_val:
                            max_val = cur_val
                            max_arg = k
                    t1[j, i+1] = self.word_given_tag.bigram_prob(word, states[j]) * max_val
                    t2[j, i+1] = max_arg
            z = np.zeros(len(sentence))
            predicted_tags = [""] * len(sentence)
            z[len(sentence)-1] = np.argmax(t1[:, len(sentence)])
            predicted_tags[len(sentence)-1] = states[int(z[len(sentence)-1])]
            for i in range(len(sentence)-1, 0, -1):
                z[i-1] = t2[int(z[i]), i+1]
                predicted_tags[i-1] = states[int(z[i-1])]
            pred_and_true_list.append((predicted_tags, true_tags))
            cnt_correct_known, cnt_correct_unknown, cnt_known, cnt_total = self.count_true(predicted_tags, true_tags,
                                                                                           words, cnt_correct_known,
                                                                                           cnt_correct_unknown,
                                                                                           cnt_known, cnt_total)
        self.confusion_mat = np.zeros((len(all_states), len(all_states)))
        # print(all_states)
        for predicted_tags, true_tags in pred_and_true_list:
            for i in range(len(predicted_tags)):
                predicted_index = all_states.index(predicted_tags[i])
                true_index = all_states.index(true_tags[i])
                self.confusion_mat[true_index, predicted_index] += 1
            # print(f"predicted tags are {predicted_tags}")
            # print(f"true tags are {true_tags}")

        # print(self.confusion_mat)
        return cnt_correct_known / cnt_known, \
               cnt_correct_unknown / (cnt_total - cnt_known), \
               (cnt_correct_unknown + cnt_correct_known) / cnt_total

    def count_true(self, predicted_tags, true_tags, words, cnt_correct_known, cnt_correct_unknown, cnt_known, cnt_total):

        for i in range(min(len(predicted_tags), len(true_tags))):
            cnt_total += 1
            if words[i] not in self.baseline.cnt_dict:
                if predicted_tags[i] == true_tags[i]:
                    cnt_correct_unknown += 1
            else:
                cnt_known += 1
                if predicted_tags[i] == true_tags[i]:
                    cnt_correct_known += 1
        return cnt_correct_known, cnt_correct_unknown, cnt_known, cnt_total




                    


nltk.download('brown')
news_text = brown.tagged_sents(categories='news')
train_test_ratio = int(len(news_text) * 0.9)
test = news_text[train_test_ratio:]
train = news_text[:train_test_ratio]
baseline = Baseline_tagger()
baseline.train(train)
bs = bigram_HMM()
bs.train(train)
print(baseline.accuracy(test))
print(bs.wiki_predict(test))
add_one_bs = bigram_HMM(1)
add_one_bs.train(train)
print(add_one_bs.wiki_predict(test))
map_dict = map_pseudo_words(news_text)
pseudo_word_HMM = bigram_HMM(1, map_dict)
pseudo_word_HMM.train(train)
print(pseudo_word_HMM.wiki_predict(test))
