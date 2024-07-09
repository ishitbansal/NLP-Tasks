import torch
import pickle
import re
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
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

train_data = pd.read_csv('./train.csv')
test_data = pd.read_csv('./test.csv')

with open('./word_index_skipgram.pkl', 'rb') as file:
    word_index_sg = pickle.load(file)

word_vectors_sg = torch.load('./skip-gram-word-vectors.pth')

sentences_train= preprocess_text(train_data['Description'])
sentences_test = preprocess_text(test_data['Description'],'test')

sentence_lengths = [len(sentence) for sentence in sentences_train]
sorted_lengths = sorted(sentence_lengths)
index_95th_percentile = int(np.percentile(range(len(sorted_lengths)), 95))
length_95th_percentile = sorted_lengths[index_95th_percentile]
length_sentence = length_95th_percentile

X_train = []
for sentence in sentences_train:
    sentence_embedding = [word_vectors_sg[word_index_sg[word]] for word in sentence]
    if len(sentence) < length_sentence:
        padding_needed = length_sentence - len(sentence)
        sentence_embedding.extend(padding_needed*[word_vectors_sg[word_index_sg[PAD_TOKEN]]])
    if sentence_embedding:
        X_train.append((sentence_embedding[:length_sentence]))
y_train = pd.get_dummies(train_data['Class Index'], prefix='value', dtype=int).values

X_test = []
for sentence in sentences_test:
    sentence_embedding = [word_vectors_sg[word_index_sg.get(word, word_index_sg[UNKNOWN_TOKEN])] for word in sentence]
    if len(sentence) < length_sentence:
        padding_needed = length_sentence - len(sentence)
        sentence_embedding.extend(padding_needed*[word_vectors_sg[word_index_sg[PAD_TOKEN]]])
    if sentence_embedding:
        X_test.append((sentence_embedding[:length_sentence]))
y_test = pd.get_dummies(test_data['Class Index'], prefix='value', dtype=int).values

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

class RNNClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNNClassifier, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        output, _ = self.rnn(x)
        output = self.fc(output[:, -1, :]) 
        return output
    
input_size = 300  
hidden_size = 128
output_size = 4
model = RNNClassifier(input_size, hidden_size, output_size)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

epochs = 10
for epoch in range(epochs):
    model.train()
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

torch.save(model, './skip-gram-classification-model.pth')

model.eval()
with torch.no_grad():
    outputs = model(X_test_tensor)
    _, predicted = torch.max(outputs, 1)
    accuracy = accuracy_score(torch.argmax(y_test_tensor, dim=1).numpy(), predicted.numpy())
    print("Accuracy:", accuracy)