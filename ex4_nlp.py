import nltk
from nltk.corpus import dependency_treebank
import numpy as np
from collections import defaultdict, namedtuple
from networkx import DiGraph
from networkx.algorithms import minimum_spanning_arborescence

words_dict = {None: 0}
tags_dict = {'TOP': 0}

def test_feature_function():
    pass

def min_spanning_arborescence_nx(arcs, sink):
    """
    Wrapper for the networkX min_spanning_tree to follow the original API
    :param arcs: list of Arc tuples
    :param sink: unused argument. We assume that 0 is the only possible root over the set of edges given to
     the algorithm.
    """
    G = DiGraph()
    for arc in arcs:
        G.add_edge(arc.head, arc.tail, weight=arc.weight)
    ARB = minimum_spanning_arborescence(G)
    result = {}
    headtail2arc = {(a.head, a.tail): a for a in arcs}
    for edge in ARB.edges:
        tail = edge[1]
        result[tail] = headtail2arc[(edge[0], edge[1])]
    return result


def feature_function(v1, v2, sentence=None):
    """
    param v1 : first word
    param v2 : second word
    param sentence : a sentence
    return : feature vector of Word bigrams
        """
    total_features = len(words_dict) ** 2 + len(tags_dict) ** 2
    res = np.zeros(total_features)
    w1 = v1['word']
    w2 = v2['word']
    if w1 in words_dict and w2 in words_dict:
        res[words_dict[w1] * len(words_dict) + words_dict[w2]] = 1
    t1 = v1["tag"]
    t2 = v2["tag"]
    if t1 in tags_dict and t2 in tags_dict:
        res[len(words_dict) ** 2 + tags_dict[t1] * len(tags_dict) + tags_dict[t2]] = 1
    return res


def set_dicts(corpus):
    """ Initialize dictionary for the words and tags's indexes in
        the feature vector """
    for tree in corpus:
        for i in range(1, len(tree.nodes)):
            word = tree.nodes[i]["word"]
            tag = tree.nodes[i]["tag"]
            if word not in words_dict and word is not None:
                words_dict[word] = len(words_dict)
            if tag not in tags_dict and tag is not None:
                tags_dict[tag] = len(tags_dict)

def perceptron(feature_size, num_iter, train, lr):
    """
    for r =1 to N number of iterration
        for i =1 to M number of trainnig set
            T' =Chu_Liu_Edmonds_algorithem

    """
    teta = np.zeros(feature_size)
    teta_sum = np.zeros(feature_size)
    for r in range(num_iter):
        for i, tree in enumerate(train):
            opt_tree = min_spanning_arborescence_nx(get_arcs(tree, teta))
            teta +=lr * (sum_edges(tree, tree.root, feature_size) - sum_edges(opt_tree, opt_tree.root, feature_size))
            teta_sum += teta
    return teta_sum / (num_iter * len(train))


def get_arcs(tree, teta):
    arcs = list()
    for node1 in tree.nodes:
        for node2 in tree.nodes:
            arcs.append((node1, -feature_function(node1, node2) * teta, node2))
    return arcs

def sum_edges(tree, root, size):
    if len(root['deps']['']) == 0:
        return np.zeros(size)
    sum = 0
    for child in root['deps']['']:
        sum += feature_function(root, tree.nodes[child]) + sum_edges(tree, tree.nodes[child], size)
    return sum

nltk.download('dependency_treebank')
sentences = dependency_treebank.parsed_sents()
train_test_ratio = int(len(sentences) * 0.9)
test = sentences[train_test_ratio:]
train = sentences[:train_test_ratio]
set_dicts(sentences)
total_features = len(words_dict) ** 2 + len(tags_dict) ** 2
sum1 = sum_edges(test[0], test[0].root, size=total_features)
print(np.sum(sum1))