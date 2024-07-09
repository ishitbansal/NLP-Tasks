import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import nltk
import string
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix, f1_score
from model import FFNN_Tagger, LSTM_Tagger
from dataset import POSDatatset_FFNN, POSDatatset_LSTM

START_TOKEN = "<s>"
END_TOKEN = "</s>"
UNKNOWN_TOKEN = "<unk>"
PAD_TOKEN = "<pad>"


class POS_Tagger:
    def __init__(self, type='ffnn'):
        self.type = type

    def train(self, train_dataset, embedding_dim=64, hidden_dim=128, p=2, s=3, model=None):

        self.p = p
        self.s = s
        self.train_dataset = train_dataset
        self.model = model

        train_dataloader = DataLoader(
            train_dataset, batch_size=32, shuffle=True)

        input_dim = len(train_dataset.vocab)
        output_dim = len(train_dataset.tags)

        if (model == None):
            if (self.type == 'ffnn'):
                self.model = FFNN_Tagger(
                    input_dim, embedding_dim, hidden_dim, output_dim, p, s)
            elif (self.type == 'lstm'):
                self.model = LSTM_Tagger(
                    input_dim, embedding_dim, hidden_dim, 1, output_dim)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        num_epochs = 5
        for epoch in range(num_epochs):
            total_loss = 0.0
            for inputs, labels in train_dataloader:
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            # print(f'Epoch {epoch+1}, Loss: {total_loss:.4f}')

        self.training_args = {'vocab': train_dataset.vocab, 'tags': train_dataset.tags,
                              'words_index': train_dataset.words_index, 'tags_index': train_dataset.tags_index}

    def evaluate(self, test_dataset):
        test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)
        total_correct = 0
        total_samples = 0
        y_true = []
        y_predicted = []
        with torch.no_grad():
            for inputs, labels in test_dataloader:
                outputs = self.model(inputs)
                if (self.type == 'ffnn'):
                    _, predicted = outputs.max(1)
                    correct_labels = labels.argmax(1)
                    samples = labels.size(0)
                elif self.type == 'lstm':
                    _, predicted = outputs.max(2)
                    correct_labels = labels.argmax(2)
                    samples = labels.size(0)*labels.size(1)
                y_true.extend(correct_labels.view(-1))
                y_predicted.extend(predicted.view(-1))
                total_correct += (predicted == correct_labels).sum().item()
                total_samples += samples
        accuracy = accuracy_score(y_true, y_predicted)
        recall = recall_score(y_true, y_predicted, average='weighted')
        f1_micro = f1_score(y_true, y_predicted, average='micro')
        f1_macro = f1_score(y_true, y_predicted, average='macro')
        confusion_mat = confusion_matrix(y_true, y_predicted)
        return accuracy, recall, f1_micro, f1_macro, confusion_mat

    def predict(self, sentence):
        words = nltk.word_tokenize(sentence)
        words = [word.lower()
                 for word in words if word not in string.punctuation]
        sente = []
        for word in words:
            sente.append(self.train_dataset.words_index.get(
                word, self.train_dataset.words_index[UNKNOWN_TOKEN]))
        tokens = sente

        X = []

        if self.type == 'ffnn':
            for i in range(len(tokens)):
                x = []
                for j in range(i-self.p, i+self.s+1):
                    if (j < 0):
                        x.append(self.train_dataset.words_index[START_TOKEN])
                    elif (j >= len(tokens)):
                        x.append(self.train_dataset.words_index[END_TOKEN])
                    else:
                        x.append(tokens[j])
                X.append(x)

            X = torch.tensor(X)
            output = self.model(X)
            _, predicted = output.max(1)

            index_tags = {}
            index_words = {}

            for k, v in self.train_dataset.words_index.items():
                index_words[v] = k
            for k, v in self.train_dataset.tags_index.items():
                index_tags[v] = k

            for i in range(len(X)):
                print(words[i], index_tags[int(predicted[i])])

        elif self.type == 'lstm':
            X.append(tokens)
            X = torch.tensor(X)

            output = self.model(X)
            _, predicted = output.max(2)

            index_tags = {}
            index_words = {}

            for k, v in self.train_dataset.words_index.items():
                index_words[v] = k
            for k, v in self.train_dataset.tags_one_hot.items():
                index_tags[np.array(v).argmax()] = k

            for i in range(len(tokens)):
                print(words[i], index_tags[int(predicted[0][i])])


if __name__ == "__main__":

    type = sys.argv[1]

    train_filepath = "./ud-treebanks-v2.13/UD_English-Atis/en_atis-ud-train.conllu"
    test_filepath = "./ud-treebanks-v2.13/UD_English-Atis/en_atis-ud-test.conllu"
    dev_filepath = "./ud-treebanks-v2.13/UD_English-Atis/en_atis-ud-dev.conllu"

    if (type == '-f'):

        p = 2
        s = 3
        train_dataset = POSDatatset_FFNN(train_filepath, p, s)
        training_args = {'vocab': train_dataset.vocab, 'tags': train_dataset.tags,
                         'words_index': train_dataset.words_index, 'tags_index': train_dataset.tags_index}
        dev_dataset = POSDatatset_FFNN(dev_filepath, p, s, training_args)

        input_dim = len(train_dataset.vocab)
        output_dim = len(train_dataset.tags)
        embedding_dim = 256
        hidden_dim = 128
        hidden_layers = 4
        activation = nn.ReLU()

        model = FFNN_Tagger(input_dim, embedding_dim, hidden_dim,
                            output_dim, p, s, hidden_layers, activation)
        tagger = POS_Tagger()
        tagger.train(train_dataset, model=model)

        torch.save(tagger, 'ffnn_model.pth')

    elif type == '-r':

        train_dataset = POSDatatset_LSTM(train_filepath)
        training_args = {'vocab': train_dataset.vocab, 'tags': train_dataset.tags, 'words_index': train_dataset.words_index,
                         'tags_index': train_dataset.tags_index, 'tags_one_hot': train_dataset.tags_one_hot}
        dev_dataset = POSDatatset_LSTM(dev_filepath, training_args)

        input_dim = len(train_dataset.vocab)
        output_dim = len(train_dataset.tags)
        embedding_dim = 64
        hidden_dim = 128
        stacks = 2
        bidirectional = False

        model = LSTM_Tagger(input_dim, embedding_dim, hidden_dim,
                            stacks, output_dim, bidirectional)
        tagger = POS_Tagger('lstm')
        tagger.train(train_dataset, dev_dataset, model=model)

        torch.save(tagger, 'lstm_model.pth')
