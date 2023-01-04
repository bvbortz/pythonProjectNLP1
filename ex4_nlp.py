from itertools import permutations

from nltk.corpus import dependency_treebank
from nltk import download
import numpy as np
from collections import defaultdict, namedtuple
from networkx import DiGraph
from networkx.algorithms import minimum_spanning_arborescence
import datetime
words_dict = {None: 0}
tags_dict = {'TOP': 0}
Arc = namedtuple('Arc', ('tail', 'weight', 'head'))
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
    res = np.zeros(total_features, dtype=np.uint8)
    w1 = v1['word']
    w2 = v2['word']
    if w1 in words_dict and w2 in words_dict:
        res[words_dict[w1] * len(words_dict) + words_dict[w2]] = 1
    t1 = v1["tag"]
    t2 = v2["tag"]
    if t1 in tags_dict and t2 in tags_dict:
        res[len(words_dict) ** 2 + tags_dict[t1] * len(tags_dict) + tags_dict[t2]] = 1
    return res

def calc_weight(node1, node2, teta):
    res = 0
    w1 = node1['word']
    w2 = node2['word']
    if w1 in words_dict and w2 in words_dict:
        res += teta[words_dict[w1] * len(words_dict) + words_dict[w2]]
    t1 = node1["tag"]
    t2 = node2["tag"]
    if t1 in tags_dict and t2 in tags_dict:
        res += teta[len(words_dict) ** 2 + tags_dict[t1] * len(tags_dict) + tags_dict[t2]]
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
    with open("/cs/usr/bvbortz/PycharmProjects/pythonProjectNLP1/check_print.txt", 'a+') as check_printing:
        print(f"started train on {len(train)} trees")
        check_printing.write(f"started train on {len(train)} trees\n")
        teta = np.zeros(feature_size)
        teta_sum = np.zeros(feature_size)
        for r in range(num_iter):
            for i, tree in enumerate(train):
                start_time = datetime.datetime.now().timestamp()
                opt_tree = min_spanning_arborescence_nx(get_arcs(tree, teta), 0)
                teta += lr * (sum_edges(tree, tree.root, feature_size) - sum_opt_edges(tree, opt_tree, feature_size))
                teta_sum += teta
                end_time = datetime.datetime.now().timestamp()
                check_printing.write(f"trained on {i} trees taken {end_time-start_time}\n")
                print(f"trained on {i} trees taken {end_time-start_time}")
        return teta_sum / (num_iter * len(train))

def sum_opt_edges(tree, opt_tree, size):
    sum = np.zeros(size, dtype=np.uint8)
    for arc in opt_tree.values():
        sum += feature_function(tree.nodes[arc.tail], tree.nodes[arc.head])
        # sum += int(arc.weight)
    return sum

def get_arcs(tree, teta):
    # arcs = list()
    # length = len(tree.nodes)
    # for i in range(1, length):
    #     arcs.append(Arc(i, -feature_function(tree.nodes[0], tree.nodes[i]) @ teta.T, 0))
    #     for j in range(1, length):
    #         if i != j:
    #             arcs.append(Arc(i, -feature_function(tree.nodes[i], tree.nodes[j]) @ teta.T, j))
    # return arcs
    return [Arc(node1['address'], -calc_weight(node1, node2, teta), node2['address']) for node1, node2 in permutations(tree.nodes.values(), 2)]


def sum_edges2(tree, root, size):
    if len(root['deps']['']) == 0:
        return np.zeros(size, dtype=np.uint8)
    sum = 0
    for child in root['deps']['']:
        sum += feature_function(root, tree.nodes[child]) + sum_edges(tree, tree.nodes[child], size)
    return sum

def sum_edges(tree, root, size):
    sum = np.zeros(size, dtype=np.uint8)
    for node in tree.nodes.values():
        for child in node['deps']['']:
            sum += feature_function(node, tree.nodes[child])
    return sum



def attachment_score(true_tree, pred_tree):
    total_equal_edges = 0
    for i in range(len(true_tree.nodes)):
        for j in range(true_tree.nodes[i]['deps']['']):
            for arc in pred_tree.values():
                if arc.tail == i and arc.head == j:
                    total_equal_edges += 1
    return total_equal_edges / len(true_tree.nodes)


def evaluate(test, teta):
    sum_score = 0
    for i, tree in enumerate(test):
        opt_tree = min_spanning_arborescence_nx(get_arcs(tree, teta), 0)
        sum_score += attachment_score(tree, opt_tree)
    return sum_score / len(test)

def new_train():
    download('dependency_treebank')
    sentences = dependency_treebank.parsed_sents()
    train_test_ratio = int(len(sentences) * 0.9)
    test = sentences[train_test_ratio:]
    train = sentences[:train_test_ratio]
    set_dicts(sentences)
    total_features = len(words_dict) ** 2 + len(tags_dict) ** 2
    teta_star = perceptron(total_features, 2, train, 1)
    np.save("teta_star.npy", teta_star)
    score = evaluate(test, teta_star)
    print(f"the attachment_score is {score}")

def old_train():
    download('dependency_treebank')
    sentences = dependency_treebank.parsed_sents()
    train_test_ratio = int(len(sentences) * 0.9)
    test = sentences[train_test_ratio:]
    train = sentences[:train_test_ratio]
    set_dicts(sentences)
    total_features = len(words_dict) ** 2 + len(tags_dict) ** 2
    teta_star = np.load("teta_star.npy")
    if len(teta_star) != total_features:
        print("saved file is not full")
        return
    score = evaluate(test, teta_star)
    print(f"the attachment_score is {score}")

new_train()