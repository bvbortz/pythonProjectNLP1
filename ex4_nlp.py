import nltk
from nltk.corpus import dependency_treebank

def word_bigrams(v1, v2, sentence, fiture="word"):
    res = list()
    for node1 in sentence.nodes:
        for node2 in sentence.nodes:
            if v1[fiture] == node1[fiture] and v2[fiture] == node2[fiture]:
                res.append(1)
            else:
                res.append(0)
    return res


nltk.download('dependency_treebank')
sentences = dependency_treebank.parsed_sents()
train_test_ratio = int(len(sentences) * 0.9)
test = sentences[train_test_ratio:]
train = sentences[:train_test_ratio]