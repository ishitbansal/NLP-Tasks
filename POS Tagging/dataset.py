import torch
import numpy as np
from conllu import parse
from sklearn.preprocessing import LabelEncoder

START_TOKEN = "<s>"
END_TOKEN = "</s>"
UNKNOWN_TOKEN = "<unk>"
PAD_TOKEN = "<pad>"


class POSDatatset_FFNN:
    def __init__(self, file_path, p=2, s=3, training_args=None):
        self.training_args = training_args
        if training_args != None:
            self.vocab = training_args['vocab']
            self.tags = training_args['tags']
            self.words_index = training_args['words_index']
            self.tags_index = training_args['tags_index']
        self.read_conllu(file_path)
        self.make_sequences(p, s)

    def read_conllu(self, file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            data = f.read()
            sentences = parse(data)

        if self.training_args == None:

            freq = {}
            tags = set()
            vocab = set()
            for sentence in sentences:
                for token in sentence:
                    vocab.add(token['form'])
                    tags.add(token['upostag'])
                    freq[token['form']] = freq.get(token['form'], 0)+1

            freq_cutoff = 3
            removed_words = set()
            for word in vocab:
                if (freq[word] < freq_cutoff):
                    removed_words.add(word)

            for word in removed_words:
                vocab.remove(word)

            vocab.add(START_TOKEN)
            vocab.add(END_TOKEN)
            vocab.add(UNKNOWN_TOKEN)

            label_encoder = LabelEncoder()
            self.tags = list(tags)
            self.vocab = list(vocab)
            encoded_tags = label_encoder.fit_transform(self.tags)
            encoded_words = label_encoder.fit_transform(self.vocab)

            self.tags_index = {}
            self.words_index = {}
            for i in range(len(tags)):
                self.tags_index[self.tags[i]] = encoded_tags[i]
            for i in range(len(vocab)):
                self.words_index[self.vocab[i]] = encoded_words[i]

        self.data = []
        for sentence in sentences:
            sente = []
            label = []
            for token in sentence:
                word = token['form']
                pos = token['upostag']
                if pos not in self.tags_index:
                    continue
                sente.append(self.words_index.get(
                    word, self.words_index[UNKNOWN_TOKEN]))
                label.append(self.tags_index[pos])
            self.data.append((sente, label))

    def make_sequences(self, p, s):
        self.X = []
        self.Y = []
        for i in range(len(self.data)):
            sentence = self.data[i][0]
            labels = self.data[i][1]
            for j in range(len(sentence)):
                x = []
                y = list(np.zeros(len(self.tags)))
                y[labels[j]] = 1
                for k in range(j-p, j+s+1):
                    if (k < 0):
                        x.append(self.words_index[START_TOKEN])
                    elif (k >= len(sentence)):
                        x.append(self.words_index[END_TOKEN])
                    else:
                        x.append(sentence[k])
                self.X.append(x)
                self.Y.append(y)
        self.X = torch.tensor(self.X)
        self.Y = torch.tensor(self.Y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        sequence = self.X[index]
        label = self.Y[index]
        return sequence, label


class POSDatatset_LSTM:
    def __init__(self, file_path, training_args=None):
        self.training_args = training_args
        if training_args != None:
            self.vocab = training_args['vocab']
            self.words_index = training_args['words_index']
            self.tags_index = training_args['tags_index']
            self.tags_one_hot = training_args['tags_one_hot']
        self.read_conllu(file_path)

    def read_conllu(self, file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            data = f.read()
            sentences = parse(data)

        if self.training_args == None:

            freq = {}
            tags = set()
            vocab = set()
            for sentence in sentences:
                for token in sentence:
                    vocab.add(token['form'])
                    tags.add(token['upostag'])
                    freq[token['form']] = freq.get(token['form'], 0)+1

            freq_cutoff = 3
            removed_words = set()
            for word in vocab:
                if (freq[word] < freq_cutoff):
                    removed_words.add(word)

            for word in removed_words:
                vocab.remove(word)

            tags.add(PAD_TOKEN)

            vocab.add(UNKNOWN_TOKEN)
            vocab.add(PAD_TOKEN)

            label_encoder = LabelEncoder()
            self.tags = list(tags)
            self.vocab = list(vocab)
            encoded_tags = label_encoder.fit_transform(self.tags)
            encoded_words = label_encoder.fit_transform(self.vocab)

            self.tags_index = {}
            self.words_index = {}
            self.tags_one_hot = {}
            for i in range(len(tags)):
                self.tags_index[self.tags[i]] = encoded_tags[i]
                label = np.zeros(len(tags))
                label[encoded_tags[i]] = 1
                self.tags_one_hot[self.tags[i]] = list(label)
            for i in range(len(vocab)):
                self.words_index[self.vocab[i]] = encoded_words[i]

        max_len = 0
        for sentence in sentences:
            max_len = max(max_len, len(sentence))

        self.data = []
        for sentence in sentences:
            sente = []
            label = []
            for token in sentence:
                word = token['form']
                pos = token['upostag']
                if pos not in self.tags_one_hot:
                    sente.append(self.words_index[PAD_TOKEN])
                    label.append(self.tags_one_hot[PAD_TOKEN])
                    continue
                sente.append(self.words_index.get(
                    word, self.words_index[UNKNOWN_TOKEN]))
                label.append(self.tags_one_hot[pos])
            for i in range(max_len-len(sentence)):
                sente.append(self.words_index[PAD_TOKEN])
                label.append(self.tags_one_hot[PAD_TOKEN])
            self.data.append((torch.tensor(sente), torch.tensor(label)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sequence = self.data[index][0]
        label = self.data[index][1]
        return sequence, label
