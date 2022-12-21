import nltk
from nltk.corpus import dependency_treebank

def word_bigrams(v1, v2, sentence):
    res = list()
    for w1 in sentence:
        for w2 in sentence:
            if v1 == w1 and v2 == w2:
                res.append(1)
            else:
                res.append(0)
    return res


nltk.download('dependency_treebank')
sentences = dependency_treebank.parsed_sents()
train_test_ratio = int(len(sentences) * 0.9)
test = sentences[train_test_ratio:]
train = sentences[:train_test_ratio]