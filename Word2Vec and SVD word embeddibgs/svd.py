import re
import pickle
import torch
import torch.nn as nn
import pandas as pd
from collections import defaultdict
from nltk.tokenize import word_tokenize
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix
import warnings
warnings.filterwarnings("ignore")

PAD_TOKEN = '<PAD>'
UNKNOWN_TOKEN = '<UNKNOWN>'
START_TOKEN = '<START>'
END_TOKEN = '<END>'


class SVDWordEmbeddings(nn.Module):
    def __init__(self, data, frequency_threshold=3, window_size=5, vector_size=300):
        self.data = data
        self.frequency_threshold = frequency_threshold
        self.window_size = window_size
        self.vector_size = vector_size
        self.vocab = set()
        self.vocab.add(PAD_TOKEN)
        self.vocab.add(UNKNOWN_TOKEN)
        self.sentences = []
        self.frequency = defaultdict(int)
        self.co_occurrence_matrix = defaultdict(int)
        self.word_vectors = None

    def preprocess_text(self, type='train'):
        for text in self.data:
            text = re.sub(r'[^\w\s\n]', ' ', str(text).lower())
            words = word_tokenize(text)
            words = [START_TOKEN] + words + [END_TOKEN]
            self.sentences.append(words)
            for word in words:
                self.frequency[word] += 1

        if (type == 'train'):
            for i in range(len(self.sentences)):
                for j in range(len(self.sentences[i])):
                    if self.frequency[self.sentences[i][j]] < self.frequency_threshold:
                        self.sentences[i][j] = UNKNOWN_TOKEN

        for sentence in self.sentences:
            for word in sentence:
                self.vocab.add(word)

    def build_co_occurrence_matrix(self):
        for tokens in self.sentences:
            for i in range(len(tokens)):
                for j in range(max(0, i - self.window_size), min(len(tokens), i + self.window_size + 1)):
                    if i != j:
                        self.co_occurrence_matrix[(tokens[i], tokens[j])] += 1

    def apply_svd(self):
        words = list(self.vocab)
        words.sort()
        self.word_index = {word: i for i, word in enumerate(words)}
        rows, cols, data = [], [], []
        for (word1, word2), count in self.co_occurrence_matrix.items():
            rows.append(self.word_index[word1])
            cols.append(self.word_index[word2])
            data.append(count)
        co_occurrence_matrix = csr_matrix(
            (data, (rows, cols)), shape=(len(words), len(words)))
        svd = TruncatedSVD(n_components=self.vector_size)
        self.word_vectors = svd.fit_transform(co_occurrence_matrix)

    def fit(self):
        self.preprocess_text()
        self.build_co_occurrence_matrix()
        self.apply_svd()


if __name__ == "__main__":
    train_data = pd.read_csv('./train.csv')
    embeddings_model_train = SVDWordEmbeddings(train_data['Description'])
    embeddings_model_train.fit()
    word_index_svd = embeddings_model_train.word_index
    word_vectors_svd = embeddings_model_train.word_vectors
    with open('./word_index_svd.pkl', 'wb') as file:
        pickle.dump(word_index_svd, file)
    torch.save(word_vectors_svd, './svd-word-vectors.pth')
