import pickle
import datetime
import numpy as np
import spacy
from datasets import load_dataset


class Unigram:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.num_words = 0
        self.dic = dict()

    def set_unigram(self, dataset):
        """
        trains unigram on the given dataset
        """
        for text in dataset['text']:
            doc = self.nlp(text)
            for token in doc:
                if token.is_alpha:
                    if token.lemma_ in self.dic:
                        self.dic[token.lemma_] = self.dic[token.lemma_] + 1
                    else:
                        self.dic[token.lemma_] = 1
                    self.num_words += 1

    def unigram_word(self, word):
        """
        finds probability of a word in the trained model
        """
        if word not in self.dic:
            return 0
        return self.dic[word] / self.num_words

    def unigram_log_prob(self, word):
        val = self.unigram_word(word)
        if val == 0:
            return -np.inf
        return np.log(val)


class Bigram:
    def __init__(self):
        self.max_next_op = dict()
        self.bidict = dict()
        self.nlp = spacy.load("en_core_web_sm")

    def set_bigram(self, dataset):
        """
        trains bigram on the given dataset
        """
        for text in dataset['text']:
            doc = self.nlp(text)
            for i, token in enumerate(doc):
                if i == 0:  # first word for each sentence is 'START'
                    first_word = "START"
                    continue
                if token.is_alpha:
                    second_word = token.lemma_
                    if first_word in self.bidict:
                        if second_word in self.bidict[first_word]:
                            self.bidict[first_word][second_word] += 1
                        else:
                            self.bidict[first_word][second_word] = 1
                    else:
                        self.bidict[first_word] = {second_word: 1}
                    if first_word in self.max_next_op:
                        self.max_next_op[first_word] += 1
                    else:
                        self.max_next_op[first_word] = 1
                    first_word = token.lemma_


    def bigram_word(self, first_word, second_word):
        """
        find the probability of the second word given the first word in the trained model
        """
        if first_word not in self.bidict:
            return 0
        if second_word not in self.bidict[first_word]:
            return 0
        return self.bidict[first_word][second_word]/self.max_next_op[first_word]

    def bigram_word_log(self, first_word, second_word):
        val = self.bigram_word(first_word, second_word)
        if val == 0:
            return -np.inf
        return np.log(val)

    def best_word(self, first_word):
        """
        find the best suitable word after the given word based on the train model
        :param first_word:
        :return:
        """
        return max(self.bidict[first_word], key=self.bidict[first_word].get)


def linear_model(unigram, bigram, unilambda, bilambda, sentence):
    """
    performs linear interpolation smoothing between the given trained unigram and bigram models and
    returns log probability of the given sentence
    """
    prob = 0
    biprob = bilambda * bigram.bigram_word("START", sentence[0].lemma_)
    uniprob = unilambda * unigram.unigram_word(sentence[0].lemma_)
    prob += np.log(biprob + uniprob)
    for word in range(len(sentence) - 1):
        biprob = bilambda * bigram.bigram_word(sentence[word].lemma_, sentence[word + 1].lemma_)
        uniprob = unilambda * unigram.unigram_word(sentence[word + 1].lemma_)
        prob += np.log(biprob + uniprob)
    return prob

def pickel_instance(filename, instance):
    f = open(filename, "wb")
    pickle.dump(instance, f)

def load_pickel(filename):
    f = open(filename, "rb")
    return pickle.load(f)

nlp = spacy.load("en_core_web_sm")
start = datetime.datetime.now().timestamp()
dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split="train")
unigram = Unigram()
unigram.set_unigram(dataset)
bigram = Bigram()
bigram.set_bigram(dataset)
pickel_instance("unigramins", unigram)
pickel_instance("bigramins", bigram)
end = datetime.datetime.now().timestamp()
print(f"finished in {end - start} seconds")
# unigram = load_pickel("unigramins")
# bigram = load_pickel("bigramins")
for token in nlp("in"):
    print("I have a house in " + bigram.best_word(token.lemma_))

sentence1 = nlp("Brad Pitt was born in Oklahoma")
sentence2 = nlp("The actor was born in USA")
prob1 = 0
prob2 = 0
prob1 += bigram.bigram_word_log("START", sentence1.doc[0].lemma_)
prob2 += bigram.bigram_word_log("START", sentence2.doc[0].lemma_)
for word in range(len(sentence1)-1):
    prob1 += bigram.bigram_word_log(sentence1[word].lemma_, sentence1[word+1].lemma_)
for word in range(len(sentence2) - 1):
    prob2 += bigram.bigram_word_log(sentence2[word].lemma_, sentence2[word + 1].lemma_)
print("the prob of the first sentence is : ")
print(prob1)
print("the prob of the second sentence is : " + str(prob2))
perplexity = np.exp(-(prob1 + prob2) / (len(sentence1) + len(sentence2)))
print(f"the perplexity is {perplexity}")
unilamda = 1/3
bilamda = 2/3
prob1 = linear_model(unigram, bigram, unilamda, bilamda, sentence1)
prob2 = linear_model(unigram, bigram, unilamda, bilamda, sentence2)
perplexity = np.exp(-(prob1 + prob2) / (len(sentence1) + len(sentence2)))
print(f"the prob of the first sentence with interpolation is: {prob1}")
print(f"the prob of the second sentence with interpolation is: {prob2}")
print(f"the perplexity is {perplexity}")