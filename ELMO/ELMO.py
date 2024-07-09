import pandas as pd
import numpy as np
import re
from nltk.tokenize import word_tokenize
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

UNKNOWN_TOKEN='UNK'
PAD_TOKEN='PAD'
START_TOKEN = '<START>'
END_TOKEN = '<END>'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_data=pd.read_csv('/kaggle/input/assignment-3/train.csv')
test_data=pd.read_csv('/kaggle/input/assignment-3/test.csv')

def preprocess_text(data,type='train'):
    sentences=[]
    vocab=set()
    vocab.add(PAD_TOKEN)
    vocab.add(UNKNOWN_TOKEN)
    total=0

    frequency=dict()
    for text in data:
        text = re.sub(r'[^\w\s\n]', ' ', str(text).lower())
        words = word_tokenize(text)
        words = [START_TOKEN] + words + [END_TOKEN]
        sentences.append(words)
        for word in words:
            frequency[word]=frequency.get(word,0)+1
            total+=1
    
    if type=='train':
        frequency_threshold=3
        for i in range(len(sentences)):
            for j in range(len(sentences[i])):
                if frequency[sentences[i][j]]<frequency_threshold:
                    sentences[i][j]=UNKNOWN_TOKEN

    for sentence in sentences:
        for word in sentence:
            vocab.add(word)
    vocab=list(vocab)
    vocab = sorted(vocab)
    return sentences,vocab

sentences_train,vocab = preprocess_text(train_data['Description'])
sentences_test,_ = preprocess_text(test_data['Description'],'test')

word2id = {}
id2word = {}
sorted_vocab = sorted(vocab)
for i, word in enumerate(sorted_vocab):
    word2id[word] = i
    id2word[i] = word


sentence_lengths = [len(sentence) for sentence in sentences_train]
sorted_lengths = sorted(sentence_lengths)
index_95th_percentile = int(np.percentile(range(len(sorted_lengths)), 95))
length_95th_percentile = sorted_lengths[index_95th_percentile]
length_sentence=length_95th_percentile

for i in range(len(sentences_train)):
    sentence=sentences_train[i]
    sentences_train[i]=[word2id.get(word,word2id[UNKNOWN_TOKEN]) for word in sentence]
for i in range(len(sentences_test)):
    sentence=sentences_test[i]
    sentences_test[i]=[word2id.get(word,word2id[UNKNOWN_TOKEN]) for word in sentence]

glove_file_path = '/kaggle/input/gloveembeddings/glove.6B.100d.txt'
glove_dict={}
with open(glove_file_path, 'r', encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        vector = [float(val) for val in values[1:]]
        glove_dict[word] = vector

embedding_matrix = torch.zeros((len(vocab), 100))
for i, word in enumerate(vocab):
    embedding_matrix[i] = torch.tensor(glove_dict.get(word,np.random.uniform(-1, 1, size=100)))

class ELMo(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, embedding_matrix):
        super(ELMo, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.embedding1 = nn.Embedding.from_pretrained(embedding_matrix)
        self.embedding2 = nn.Embedding.from_pretrained(embedding_matrix)
        self.lstm_forward1 = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.lstm_forward2 = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.lstm_backward1 = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.lstm_backward2 = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.linear_mode1 = nn.Linear(200, vocab_size)
        self.linear_mode2 = nn.Linear(200, vocab_size)

    def forward(self, input_data, mode):
        if mode == 1:
            forward_embed = self.embedding1(input_data)
            forward_lstm1, _ = self.lstm_forward1(forward_embed) 
            forward_lstm2, _ = self.lstm_forward2(forward_lstm1) 
            lstm_concat = torch.cat((forward_lstm1, forward_lstm2), dim=-1)
            output = self.linear_mode1(lstm_concat)
            return output
        
        elif mode == 2:
            backward_embed = self.embedding2(input_data)
            backward_lstm1, _ = self.lstm_backward1(backward_embed) 
            backward_lstm2, _ = self.lstm_backward2(backward_lstm1) 
            lstm_concat = torch.cat((backward_lstm1, backward_lstm2), dim=-1)
            output = self.linear_mode2(lstm_concat)
            return output
        
vocab_size = len(vocab)
embedding_dim = 100
hidden_dim = 100
batch_size=32

elmo = ELMo(vocab_size, embedding_dim, hidden_dim, embedding_matrix)
elmo.to(device)

X_train = []
for sentence in sentences_train:
    if len(sentence) < length_sentence:
        padding_needed = length_sentence - len(sentence)
        sentence.extend(padding_needed*[word2id[PAD_TOKEN]])
    X_train.append(torch.tensor(sentence[:length_sentence]))
y_train = pd.get_dummies(train_data['Class Index'], prefix='value', dtype=int).values

X_test = []
for sentence in sentences_test:
    if len(sentence) < length_sentence:
        padding_needed = length_sentence - len(sentence)
        sentence.extend(padding_needed*[word2id[PAD_TOKEN]])
    X_test.append(torch.tensor(sentence[:length_sentence]))
y_test = pd.get_dummies(test_data['Class Index'], prefix='value', dtype=int).values


X_train_tensor = torch.stack(X_train)
X_test_tensor = torch.stack(X_test)
y_train_tensor = torch.tensor(y_train,dtype=torch.float32)
y_test_tensor = torch.tensor(y_test,dtype=torch.float32)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

num_epochs = 10
criterion = nn.CrossEntropyLoss(ignore_index=word2id[PAD_TOKEN])
optimizer = optim.Adam(elmo.parameters(), lr=0.001)  

for epoch in range(num_epochs):
    elmo.train()
    total_loss = 0.0
    total_tokens = 0
    for inputs in tqdm(train_loader):
        inputs = inputs[0]
        inputs = inputs.to(device)
        optimizer.zero_grad()
        input_seq = inputs[:, :length_sentence-1]
        target_seq = inputs[:, 1:]
        outputs = elmo(input_seq, mode=1)
        loss = criterion(outputs.permute(0, 2, 1), target_seq)
        total_loss += loss.item()
        total_tokens += target_seq.numel()
        loss.backward()
        optimizer.step()
    avg_loss = total_loss / total_tokens
    print(f"Epoch {epoch+1}/{num_epochs} (Mode 1) - Train Loss: {avg_loss:.4f}")

for epoch in range(num_epochs):
    elmo.train()
    total_loss = 0.0
    total_tokens = 0
    for inputs in tqdm(train_loader):
        inputs = inputs[0]
        inputs = inputs.to(device)
        optimizer.zero_grad()
        inputs = torch.flip(inputs, dims=[1])
        input_seq = inputs[:, :length_sentence-1]
        target_seq = inputs[:, 1:]
        outputs = elmo(input_seq, mode=1)
        loss = criterion(outputs.permute(0, 2, 1), target_seq)
        total_loss += loss.item()
        total_tokens += target_seq.numel()
        loss.backward()
        optimizer.step()
    avg_loss = total_loss / total_tokens
    print(f"Epoch {epoch+1}/{num_epochs} (Mode 2) - Train Loss: {avg_loss:.4f}")

elmo.eval()
total_loss = 0.0
total_tokens = 0 
with torch.no_grad():
    for inputs in tqdm(test_loader):
        inputs = inputs[0]
        inputs = inputs.to(device)
        input_seq = inputs[:, :length_sentence-1]
        target_seq = inputs[:, 1:]
        outputs = elmo(input_seq, mode=1)
        loss = criterion(outputs.permute(0, 2, 1), target_seq) 
        total_loss += loss.item()
        total_tokens += target_seq.numel()
    avg_loss = total_loss / total_tokens
print(f"Test Loss (Mode 1): {avg_loss:.4f}")

elmo.eval()
total_loss = 0.0
total_tokens = 0  
with torch.no_grad():
    for inputs in tqdm(test_loader):
        inputs = inputs[0]
        inputs = inputs.to(device)
        inputs = torch.flip(inputs, dims=[1])
        input_seq = inputs[:, :length_sentence-1]
        target_seq = inputs[:, 1:]
        outputs = elmo(input_seq, mode=2)
        loss = criterion(outputs.permute(0, 2, 1), target_seq) 
        total_loss += loss.item()
        total_tokens += target_seq.numel()
    avg_loss = total_loss / total_tokens
print(f"Test Loss (Mode 2): {avg_loss:.4f}")

model_path='/kaggle/working/bilstm.pth'
torch.save(elmo.state_dict(), model_path)