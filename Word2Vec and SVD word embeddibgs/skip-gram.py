import numpy as np
import pandas as pd
import pickle
import torch
import torch.nn as nn
from collections import defaultdict
import random
import re
from nltk.tokenize import word_tokenize
import warnings
warnings.filterwarnings("ignore")

PAD_TOKEN = '<PAD>'
UNKNOWN_TOKEN = '<UNKNOWN>'
START_TOKEN = '<START>'
END_TOKEN = '<END>'

def preprocess_text(data,type='train'):
    sentences=[]
    frequency=dict()
    for text in data:
        text = re.sub(r'[^\w\s\n]', ' ', str(text).lower())
        words = word_tokenize(text)
        words = [START_TOKEN] + words + [END_TOKEN]
        sentences.append(words)
        for word in words:
            frequency[word]=frequency.get(word,0)+1
    
    if type=='train':
        frequency_threshold=3
        for i in range(len(sentences)):
            for j in range(len(sentences[i])):
                if frequency[sentences[i][j]]<frequency_threshold:
                    sentences[i][j]=UNKNOWN_TOKEN
    return sentences

class SkipGramNegativeSampling(nn.Module):
    def __init__(self, corpus, vector_size=300, window_size=5, negative_samples=5, learning_rate=0.025):
        self.corpus = corpus
        self.vector_size = vector_size
        self.window_size = window_size
        self.negative_samples = negative_samples
        self.learning_rate = learning_rate
        self.word2id = {}
        self.id2word = {}
        self.vocab_size = 0
        self.word_count = defaultdict(int)
        self.word_pairs = []
        self.vocab=set()
        self.initialize()

    def initialize(self):
        for sentence in self.corpus:
            for word in sentence:
                self.word_count[word] += 1
                self.vocab.add(word)
        self.vocab.add(PAD_TOKEN)
        self.word_count[PAD_TOKEN]=1
        words = list(self.vocab)
        words.sort()
        for i, word in enumerate(words):
            self.word2id[word] = i
            self.id2word[i] = word
        self.vocab_size = len(self.vocab)

    def similarity(self, vec1, vec2):
        dot_product = np.dot(vec1, vec2)
        return dot_product
    
    def sigmoid(self,z):
        return 1 / (1 + np.exp(-z))

    def generate_word_pairs(self):
        word_pairs=set()
        for sentence in self.corpus:
            for i, target_word in enumerate(sentence):
                target_word_id = self.word2id[target_word]
                for j in range(i - self.window_size, i + self.window_size + 1):
                    if j != i and j >= 0 and j < len(sentence):
                        context_word = sentence[j]
                        context_word_id = self.word2id[context_word]
                        word_pairs.add((target_word_id, context_word_id))
        self.word_pairs=list(word_pairs)
        random.shuffle(self.word_pairs)

    def train(self, epochs=5):
        self.generate_word_pairs()
        self.W = np.random.uniform(-0.5/self.vector_size, 0.5/self.vector_size, (self.vocab_size, self.vector_size))
        self.C = np.random.uniform(-0.5/self.vector_size, 0.5/self.vector_size, (self.vocab_size, self.vector_size))
        for epoch in range(epochs):
            loss = 0
            for target_word_id, context_word_id in self.word_pairs:
                loss += self.train_pair(target_word_id, context_word_id)
            print(f"Epoch {epoch + 1}: Loss = {loss / len(self.word_pairs)}")

    def train_pair(self, target_word_id, context_word_id):
        target_vector = self.W[target_word_id] # w
        context_vector = self.C[context_word_id] # c_pos

        negative_samples = random.choices(range(self.vocab_size), k=self.negative_samples)

        similarity_score = self.similarity(target_vector, context_vector) # w.c_pos
        sigmoid_score = self.sigmoid(similarity_score) # sigmoid(w.c_pos)
        gradients = (sigmoid_score - 1) * context_vector 
        self.C[context_word_id]-=self.learning_rate *(sigmoid_score - 1) * target_vector # C^(t+1)_pos=C^(t)_pos-lr*(sigmoid(c_pos.w)-1)*w
        loss=-np.log(sigmoid_score+1e-10) # log(sigmoid(c_pos.w))

        for sample_word_id in negative_samples :
            sample_vector = self.C[sample_word_id] # c_neg
            similarity_score = self.similarity(target_vector, sample_vector) # w.c_neg
            sigmoid_score = self.sigmoid(similarity_score) # sigmoid(w.c_neg)
            gradients += sigmoid_score * sample_vector
            self.C[sample_word_id]-=self.learning_rate*sigmoid_score*target_vector # C^(t+1)_neg=C^(t)_neg-lr*sigmoid(c_neg.w)*w
            loss+=-np.log(1-sigmoid_score+1e-10) # log(1-sigmoid(c_neg.w))
        
        self.W[target_word_id] -= self.learning_rate * gradients
        return loss
    
if __name__ == "__main__":
    train_data=pd.read_csv('./train.csv')
    sentences_train = preprocess_text(train_data['Description'])
    sg_model = SkipGramNegativeSampling(sentences_train)
    sg_model.train(epochs=10)
    word_vectors_skipgram=sg_model.W
    word_index_sg=sg_model.word2id
    with open('./word_index_skipgram.pkl', 'wb') as file:
        pickle.dump(word_index_sg, file)
    torch.save(word_vectors_skipgram, './skip-gram-word-vectors.pth')

