import nltk
from nltk.corpus import dependency_treebank

words_dict = {}
tags_dict = {}

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

def perceptron(feature_size, num_iter, train):
    """
    for r =1 to N number of iterration
        for i =1 to M number of trainnig set
            T' =Chu_Liu_Edmonds_algorithem

    """
    teta = np.zeros(feature_size)
    for r in num_iter:
        for i, tree in enumerate(train):
            pass


nltk.download('dependency_treebank')
sentences = dependency_treebank.parsed_sents()
train_test_ratio = int(len(sentences) * 0.9)
test = sentences[train_test_ratio:]
train = sentences[:train_test_ratio]