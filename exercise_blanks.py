import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset, Dataset
import operator
import data_loader
import pickle
import tqdm
import time
import plotly.graph_objects as go

# ------------------------------------------- Constants ----------------------------------------

SEQ_LEN = 52
W2V_EMBEDDING_DIM = 300

ONEHOT_AVERAGE = "onehot_average"
W2V_AVERAGE = "w2v_average"
W2V_SEQUENCE = "w2v_sequence"

TRAIN = "train"
VAL = "val"
TEST = "test"


# ------------------------------------------ Helper methods and classes --------------------------

def get_available_device():
    """
    Allows training on GPU if available. Can help with running things faster when a GPU with cuda is
    available but not a most...
    Given a device, one can use module.to(device)
    and criterion.to(device) so that all the computations will be done on the GPU.
    """
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def save_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_model(model, path, epoch, optimizer):
    """
    Utility function for saving checkpoint of a model, so training or evaluation can be executed later on.
    :param model: torch module representing the model
    :param optimizer: torch optimizer used for training the module
    :param path: path to save the checkpoint into
    """
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()}, path)


def load(model, path, optimizer):
    """
    Loads the state (weights, paramters...) of a model which was saved with save_model
    :param model: should be the same model as the one which was saved in the path
    :param path: path to the saved checkpoint
    :param optimizer: should be the same optimizer as the one which was saved in the path
    """
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    return model, optimizer, epoch


# ------------------------------------------ Data utilities ----------------------------------------

def load_word2vec():
    """ Load Word2Vec Vectors
        Return:
            wv_from_bin: All 3 million embeddings, each lengh 300
    """
    import gensim.downloader as api
    wv_from_bin = api.load("word2vec-google-news-300")
    # vocab = list(wv_from_bin.vocab.keys())
    vocab = list(wv_from_bin.index_to_key)
    # print(wv_from_bin.vocab[vocab[0]])
    print("Loaded vocab size %i" % len(vocab))
    return wv_from_bin


def create_or_load_slim_w2v(words_list, cache_w2v=True):#TODO explain in readme changed the defualt from Flase to True
    """
    returns word2vec dict only for words which appear in the dataset.
    :param words_list: list of words to use for the w2v dict
    :param cache_w2v: whether to save locally the small w2v dictionary
    :return: dictionary which maps the known words to their vectors
    """
    print("i got to create_or_load_slim_w2v")
    w2v_path = "w2v_dict.pkl"
    if not os.path.exists(w2v_path):
        full_w2v = load_word2vec()
        w2v_emb_dict = {k: full_w2v[k] for k in words_list if k in full_w2v}
        if cache_w2v:
            save_pickle(w2v_emb_dict, w2v_path)
    else:
        w2v_emb_dict = load_pickle(w2v_path)
    return w2v_emb_dict


def get_w2v_average(sent, word_to_vec, embedding_dim): # TODO make sure this function works
    """
    This method gets a sentence and returns the average word embedding of the words consisting
    the sentence.
    :param sent: the sentence object
    :param word_to_vec: a dictionary mapping words to their vector embeddings
    :param embedding_dim: the dimension of the word embedding vectors
    :return The average embedding vector as numpy ndarray.
    """
    new_emb = np.zeros(embedding_dim)
    for i in range(len(sent.text)):
        if sent.text[i] in word_to_vec:
            new_emb += word_to_vec[sent.text[i]]
    new_emb /= len(sent.text)
    return new_emb


def get_one_hot(size, ind):
    """
    this method returns a one-hot vector of the given size, where the 1 is placed in the ind entry.
    :param size: the size of the vector
    :param ind: the entry index to turn to 1
    :return: numpy ndarray which represents the one-hot vector
    """
    one_hot = np.zeros(size)
    one_hot[ind] = 1
    return one_hot


def average_one_hots(sent, word_to_ind):
    """
    this method gets a sentence, and a mapping between words to indices, and returns the average
    one-hot embedding of the tokens in the sentence.
    :param sent: a sentence object.
    :param word_to_ind: a mapping between words to indices
    :return:
    """
    vec_size = len(word_to_ind)
    average_vec = np.zeros(vec_size)
    for word in sent.text:
        average_vec += get_one_hot(vec_size, word_to_ind[word])
    average_vec = average_vec / len(sent.text)
    return average_vec


def get_word_to_ind(words_list):
    """
    this function gets a list of words, and returns a mapping between
    words to their index.
    :param words_list: a list of words
    :return: the dictionary mapping words to the index
    """
    word_set = set()
    word_to_index = dict()
    cur_index = 0
    for word in words_list:
        if word not in word_set:
            word_set.add(word)
            word_to_index[word] = cur_index
            cur_index += 1
    return word_to_index


def sentence_to_embedding(sent, word_to_vec, seq_len, embedding_dim=300):
    """
    this method gets a sentence and a word to vector mapping, and returns a list containing the
    words embeddings of the tokens in the sentence.
    :param sent: a sentence object
    :param word_to_vec: a word to vector mapping.
    :param seq_len: the fixed length for which the sentence will be mapped to.
    :param embedding_dim: the dimension of the w2v embedding
    :return: numpy ndarray of shape (seq_len, embedding_dim) with the representation of the sentence
    """
    emb = np.zeros((seq_len, embedding_dim))
    for i in range(len(sent.text)):
        if i == seq_len:
            break
        if sent.text[i] in word_to_vec:
            emb[i, :] = word_to_vec[sent.text[i]]
    return emb



class OnlineDataset(Dataset):
    """
    A pytorch dataset which generates model inputs on the fly from sentences of SentimentTreeBank
    """

    def __init__(self, sent_data, sent_func, sent_func_kwargs):
        """
        :param sent_data: list of sentences from SentimentTreeBank
        :param sent_func: Function which converts a sentence to an input datapoint
        :param sent_func_kwargs: fixed keyword arguments for the state_func
        """
        self.data = sent_data
        self.sent_func = sent_func
        self.sent_func_kwargs = sent_func_kwargs

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sent = self.data[idx]
        sent_emb = self.sent_func(sent, **self.sent_func_kwargs)
        sent_label = sent.sentiment_class
        return sent_emb, sent_label


class DataManager():
    """
    Utility class for handling all data management task. Can be used to get iterators for training and
    evaluation.
    """

    def __init__(self, data_type=ONEHOT_AVERAGE, use_sub_phrases=True, dataset_path="stanfordSentimentTreebank", batch_size=50,
                 embedding_dim=None):
        """
        builds the data manager used for training and evaluation.
        :param data_type: one of ONEHOT_AVERAGE, W2V_AVERAGE and W2V_SEQUENCE
        :param use_sub_phrases: if true, training data will include all sub-phrases plus the full sentences
        :param dataset_path: path to the dataset directory
        :param batch_size: number of examples per batch
        :param embedding_dim: relevant only for the W2V data types.
        """

        # load the dataset
        self.sentiment_dataset = data_loader.SentimentTreeBank(dataset_path, split_words=True)
        # map data splits to sentences lists
        self.sentences = {}
        if use_sub_phrases:
            self.sentences[TRAIN] = self.sentiment_dataset.get_train_set_phrases()
        else:
            self.sentences[TRAIN] = self.sentiment_dataset.get_train_set()

        self.sentences[VAL] = self.sentiment_dataset.get_validation_set()
        self.sentences[TEST] = self.sentiment_dataset.get_test_set()

        # map data splits to sentence input preperation functions
        words_list = list(self.sentiment_dataset.get_word_counts().keys())
        if data_type == ONEHOT_AVERAGE:
            self.sent_func = average_one_hots
            self.sent_func_kwargs = {"word_to_ind": get_word_to_ind(words_list)}
        elif data_type == W2V_SEQUENCE:
            self.sent_func = sentence_to_embedding

            self.sent_func_kwargs = {"seq_len": SEQ_LEN,
                                     "word_to_vec": create_or_load_slim_w2v(words_list),
                                     "embedding_dim": embedding_dim
                                     }
        elif data_type == W2V_AVERAGE:
            self.sent_func = get_w2v_average
            words_list = list(self.sentiment_dataset.get_word_counts().keys())
            self.sent_func_kwargs = {"word_to_vec": create_or_load_slim_w2v(words_list),
                                     "embedding_dim": embedding_dim
                                     }
        else:
            raise ValueError("invalid data_type: {}".format(data_type))
        # map data splits to torch datasets and iterators
        self.torch_datasets = {k: OnlineDataset(sentences, self.sent_func, self.sent_func_kwargs) for
                               k, sentences in self.sentences.items()}
        self.torch_iterators = {k: DataLoader(dataset, batch_size=batch_size, shuffle=k == TRAIN)
                                for k, dataset in self.torch_datasets.items()}

    def get_torch_iterator(self, data_subset=TRAIN):
        """
        :param data_subset: one of TRAIN VAL and TEST
        :return: torch batches iterator for this part of the datset
        """
        return self.torch_iterators[data_subset]

    def get_labels(self, data_subset=TRAIN):
        """
        :param data_subset: one of TRAIN VAL and TEST
        :return: numpy array with the labels of the requested part of the datset in the same order of the
        examples.
        """
        return np.array([sent.sentiment_class for sent in self.sentences[data_subset]])

    def get_input_shape(self):
        """
        :return: the shape of a single example from this dataset (only of x, ignoring y the label).
        """
        return self.torch_datasets[TRAIN][0][0].shape




# ------------------------------------ Models ----------------------------------------------------

class LSTM(nn.Module):
    """
    An LSTM for sentiment analysis with architecture as described in the exercise description.
    """
    def __init__(self, embedding_dim, hidden_dim, n_layers, dropout):
        super(LSTM, self).__init__()
        self.num_layers = n_layers
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(embedding_dim,
                            hidden_dim,
                            num_layers=n_layers,
                            bidirectional=True,
                            batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, 1)
    def forward(self, text):
        batch_size = text.shape[0]
        h_0, c_0 = self.init_hidden(batch_size)
        # x = self.lstm(text, (2 * self.num_layers, self.hidden_dim))
        x, (h_n, c_n) = self.lstm(text.float(), (h_0, c_0))
        dropped = self.dropout(x[:, -1, :])
        output = self.fc(dropped)
        return output

    def init_hidden(self, batch_size):
        h, c = (Variable(torch.zeros(self.num_layers*2, batch_size, self.hidden_dim)),
                Variable(torch.zeros(self.num_layers * 2, batch_size, self.hidden_dim)))
        return h, c

    def predict(self, text):
        batch_size = text.shape[0]
        h_0, c_0 = self.init_hidden(batch_size)
        # x = self.lstm(text, (2 * self.num_layers, self.hidden_dim))
        x, (h_n, c_n) = self.lstm(text.float(), (h_0, c_0))
        output = self.fc(x[:, -1, :])
        return F.sigmoid(output)


class LogLinear(nn.Module):
    """
    general class for the log-linear models for sentiment analysis.
    """
    def __init__(self, embedding_dim):
        super(LogLinear, self).__init__()
        self.embedding_dim = embedding_dim
        self.fc = torch.nn.Linear(embedding_dim[0], 1)
        return

    def forward(self, x):
        flat_x = torch.flatten(x, 1).float()
        return self.fc(flat_x)

    def predict(self, x):
        return F.sigmoid(self.forward(x))


# ------------------------- training functions -------------


def binary_accuracy(preds, y):
    """
    This method returns the accuracy of the predictions, relative to the labels.
    You can choose whether to use numpy arrays or tensors here.
    :param preds: a vector of predictions
    :param y: a vector of true labels
    :return: scalar value - (<number of accurate predictions> / <number of examples>)
    """
    # true_cnt = 0
    # for i in range(len(preds)):
    #     if np.round(preds[i]) == y[i]:
    #         true_cnt += 1
    # preds = preds.detach().numpy()
    y = y.reshape(preds.shape)
    return torch.sum(torch.round(preds) == y).float() / len(preds)


def train_epoch(model, data_iterator, optimizer, criterion):
    """
    This method operates one epoch (pass over the whole train set) of training of the given model,
    and returns the accuracy and loss for this epoch
    :param model: the model we're currently training
    :param data_iterator: an iterator, iterating over the training data for the model.
    :param optimizer: the optimizer object for the training process.
    :param criterion: the criterion object for the training process.
    """
    accuracy = 0.0
    running_loss = 0.0
    data_iterator_len = 0
    for data in data_iterator:
        inputs, labels = data
        optimizer.zero_grad()

        # outputs = model(inputs.float())
        outputs = model(inputs)
        # outputs = outputs.reshape(labels.shape)
        labels = labels.reshape(outputs.shape)
        accuracy += binary_accuracy(outputs, labels)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        data_iterator_len += 1
    return accuracy/data_iterator_len, running_loss / data_iterator_len


def evaluate(model, data_iterator, criterion):
    """
    evaluate the model performance on the given data
    :param model: one of our models..
    :param data_iterator: torch data iterator for the relevant subset
    :param criterion: the loss criterion used for evaluation
    :return: tuple of (average loss over all examples, average accuracy over all examples)
    """
    accuracy = 0.0
    running_loss = 0.0
    data_iterator_len = 0
    for data in data_iterator:
        inputs, labels = data

        outputs = model(inputs)
        labels = labels.reshape(outputs.shape)
        accuracy += binary_accuracy(outputs, labels)
        loss = criterion(outputs, labels)
        loss.backward()

        running_loss += loss.item()
        data_iterator_len += 1
    return accuracy / data_iterator_len, running_loss / data_iterator_len


def get_predictions_for_data(model, data_iter):
    """

    This function should iterate over all batches of examples from data_iter and return all of the models
    predictions as a numpy ndarray or torch tensor (or list if you prefer). the prediction should be in the
    same order of the examples returned by data_iter.
    :param model: one of the models you implemented in the exercise
    :param data_iter: torch iterator as given by the DataManager
    :return:
    """
    predictions = list()
    for data in data_iter:
        inputs, labels = data
        outputs = model.predict(inputs)
        predictions.append(outputs)
    return predictions


def train_model(model, data_manager, n_epochs, lr, weight_decay=0.):
    """
    Runs the full training procedure for the given model. The optimization should be done using the Adam
    optimizer with all parameters but learning rate and weight decay set to default.
    :param model: module of one of the models implemented in the exercise
    :param data_manager: the DataManager object
    :param n_epochs: number of times to go over the whole training set
    :param lr: learning rate to be used for optimization
    :param weight_decay: parameter for l2 regularization
    """
    accuracy = list()
    train_loss = list()
    val_accuracy = list()
    val_train_loss = list()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    for epoch in range(n_epochs):
        print(f"starting the {epoch+1} epoch")
        start = time.time()
        acc, loss = train_epoch(model, data_manager.get_torch_iterator(data_subset=TRAIN), optimizer, criterion)
        end = time.time()
        accuracy.append(acc)
        train_loss.append(loss)
        print(f"finshed the {epoch + 1} and it took {end - start} epoch train_acc = {acc} , train_loss = {loss}")
        start = time.time()
        val_acc, val_loss = evaluate(model, data_manager.get_torch_iterator(data_subset=VAL), criterion)
        end = time.time()
        print(f"finshed the {epoch+1} and it took {end - start} epoch val_acc = {val_acc} , val_loss = {val_loss}")
        val_accuracy.append(val_acc)
        val_train_loss.append(val_loss)
    return val_accuracy, val_train_loss, accuracy, train_loss


def train_log_linear_with_one_hot():
    """
    Here comes your code for training and evaluation of the log linear model with one hot representation.
    """
    data_manager = DataManager(batch_size=64)
    model = LogLinear(data_manager.get_input_shape())
    val_acc, val_loss, train_accuracy, train_loss = train_model(model, data_manager, 20, lr=0.01, weight_decay=0.001)
    return model, val_acc, val_loss, train_accuracy, train_loss


def train_log_linear_with_w2v():
    """
    Here comes your code for training and evaluation of the log linear model with word embeddings
    representation.
    """
    data_manager = DataManager(data_type=W2V_AVERAGE, batch_size=64, embedding_dim=W2V_EMBEDDING_DIM)
    model = LogLinear(data_manager.get_input_shape())
    val_acc, val_loss, train_accuracy, train_loss = train_model(model, data_manager, 20, lr=0.01, weight_decay=0.001)
    return model, val_acc, val_loss, train_accuracy, train_loss



def train_lstm_with_w2v():
    """
    Here comes your code for training and evaluation of the LSTM model.
    """
    data_manager = DataManager(data_type=W2V_SEQUENCE, batch_size=64, embedding_dim=W2V_EMBEDDING_DIM)
    model = LSTM( embedding_dim=W2V_EMBEDDING_DIM, hidden_dim=SEQ_LEN, n_layers=100, dropout=0.5)
    val_acc, val_loss, train_accuracy, train_loss = train_model(model, data_manager, 4, lr=0.001, weight_decay=0.0001)
    return model, val_acc, val_loss, train_accuracy, train_loss


def run_part(part):
    if part == 5:
        return train_log_linear_with_one_hot()
    if part == 7:
        return train_log_linear_with_w2v()
    if part == 8:
        return train_lstm_with_w2v()

if __name__ == '__main__':
    model, val_acc, val_loss, train_accuracy, train_loss = run_part(8)
    fig = go.Figure([go.Scatter(name="Validetion Loss", y=val_loss),
                     go.Scatter(name="Train Loss", y=train_loss)])
    fig.show()
    fig2 = go.Figure([go.Scatter(name="Validetion acc", y=val_acc),
                     go.Scatter(name="Train acc", y=train_accuracy)])
    fig2.show()
    # train_log_linear_with_w2v()
    # train_lstm_with_w2v()