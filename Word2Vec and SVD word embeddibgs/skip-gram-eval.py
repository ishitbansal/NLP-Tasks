import pickle
import re
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from nltk.tokenize import word_tokenize
from torch.utils.data import DataLoader, TensorDataset
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


class RNNClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNNClassifier, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        output, _ = self.rnn(x)
        output = self.fc(output[:, -1, :])
        return output

train_data = pd.read_csv('./train.csv')
test_data = pd.read_csv('./test.csv')

sentences_train= preprocess_text(train_data['Description'])
sentences_test = preprocess_text(test_data['Description'],'test')

with open('./word_index_skipgram.pkl', 'rb') as file:
    word_index_sg = pickle.load(file)

word_vectors_sg = torch.load('./skip-gram-word-vectors.pth')

sentence_lengths = [len(sentence) for sentence in sentences_train]
sorted_lengths = sorted(sentence_lengths)
index_95th_percentile = int(np.percentile(range(len(sorted_lengths)), 95))
length_95th_percentile = sorted_lengths[index_95th_percentile]
length_sentence = length_95th_percentile

X_train = []
for sentence in sentences_train:
    sentence_embedding = [word_vectors_sg[word_index_sg.get(word, word_index_sg[UNKNOWN_TOKEN])] for word in sentence]
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

model = torch.load('./skip-gram-classification-model.pth')

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

model.eval()
with torch.no_grad():
    y_true = []
    y_pred = []
    correct_samples=0
    total_samples=0
    for batch_X, batch_y in train_loader:
        outputs = model(batch_X)
        _, predicted = torch.max(outputs, 1)
        y_true.extend(torch.argmax(batch_y, dim=1).numpy())
        y_pred.extend(predicted.numpy())
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    cm = confusion_matrix(y_true, y_pred)
    print("Train Set:")
    print("Accuracy:",accuracy)
    print("Precision:", precision)
    print("Recall:",recall)
    print("F1 Score:",f1)
    print("Confusion Matrix:",cm)
    print()

model.eval()
with torch.no_grad():
    outputs = model(X_test_tensor)
    _, predicted = torch.max(outputs, 1)
    y_true=torch.argmax(y_test_tensor, dim=1).numpy()
    y_pred=predicted.numpy()
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    cm = confusion_matrix(y_true, y_pred)
    print("Test Set:")
    print("Accuracy:",accuracy)
    print("Precision:", precision)
    print("Recall:",recall)
    print("F1 Score:",f1)
    print("Confusion Matrix:",cm)
