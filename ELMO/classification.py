import pandas as pd
import numpy as np
import re
from nltk.tokenize import word_tokenize
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
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

model_path = "/kaggle/input/bilstm/bilstm.pth"
elmo = ELMo(vocab_size, embedding_dim, hidden_dim, embedding_matrix)
state_dict = torch.load(model_path)

elmo.load_state_dict(state_dict)

class ELMo_Embeddings(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, embedding_matrix):
        super(ELMo_Embeddings, self).__init__()
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

    def forward(self, input_data):
        forward_embed = self.embedding1(input_data)
        forward_lstm1, _ = self.lstm_forward1(forward_embed) 
        forward_lstm2, _ = self.lstm_forward2(forward_lstm1)

        input_data = torch.flip(input_data, dims=[1])
        backward_embed = self.embedding2(input_data)
        backward_lstm1, _ = self.lstm_backward1(backward_embed)
        backward_lstm2, _ = self.lstm_backward2(backward_lstm1)
        backward_lstm1 = torch.flip(backward_lstm1, dims=[1])
        backward_lstm2 = torch.flip(backward_lstm2, dims=[1])

        e1 = torch.cat((forward_embed, forward_embed), dim=-1)
        e2 = torch.cat((forward_lstm1, backward_lstm1), dim=-1)
        e3 = torch.cat((forward_lstm2, backward_lstm2), dim=-1)

        return e1,e2,e3
    

elmo_embed=ELMo_Embeddings(vocab_size, embedding_dim, hidden_dim, embedding_matrix)
elmo_embed.load_state_dict(elmo.state_dict())
elmo_embed.to(device)

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

class LSTMModel_Trainable(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel_Trainable, self).__init__()
        self.weights=nn.Parameter(torch.tensor([0.33,0.33,0.33]))
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, e1, e2, e3):
        weights_softmax = torch.nn.functional.softmax(self.weights, dim=0)
        x = e1 * weights_softmax[0] + e2 * weights_softmax[1] + e3 * weights_softmax[2]
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out
    
input_size = 200  
hidden_size = 128
output_size = 4
model = LSTMModel_Trainable(input_size, hidden_size, output_size).to(device)

criterion = nn.CrossEntropyLoss(ignore_index=word2id[PAD_TOKEN])
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 10
for epoch in range(epochs):
    model.train()
    total_loss = 0.0
    total_samples = 0
    
    for batch_X, batch_y in tqdm(train_loader):
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        e1, e2, e3 = elmo_embed(batch_X)
        e1, e2, e3 = e1.to(device), e2.to(device), e3.to(device)
        optimizer.zero_grad()
        outputs = model(e1, e2, e3)
        _, target_indices = batch_y.max(dim=1)
        loss = criterion(outputs, target_indices)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch_X.size(0)
        total_samples += batch_X.size(0)
    epoch_loss = total_loss / total_samples
    print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss}")

model_path='/kaggle/working/classifier.pth'
torch.save(model.state_dict(), model_path)

model.eval()
with torch.no_grad():
    y_true = []
    y_pred = []
    for batch_X, batch_y in tqdm(train_loader):
        batch_X = batch_X.to(device)
        e1, e2, e3 = elmo_embed(batch_X)
        e1, e2, e3 = e1.to(device), e2.to(device), e3.to(device)
        outputs = model(e1, e2, e3)
        _, predicted = torch.max(outputs, 1)
        y_true.extend(torch.argmax(batch_y, dim=1).cpu().numpy())  
        y_pred.extend(predicted.cpu().numpy())
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    cm = confusion_matrix(y_true, y_pred)
    print("Train Set:")
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print("Confusion Matrix:", cm)
    print()
    
model.eval()
with torch.no_grad():
    y_true = []
    y_pred = []
    for batch_X, batch_y in tqdm(test_loader):
        batch_X = batch_X.to(device)
        e1, e2, e3 = elmo_embed(batch_X)
        e1, e2, e3 = e1.to(device), e2.to(device), e3.to(device)
        outputs = model(e1, e2, e3)
        _, predicted = torch.max(outputs, 1)
        y_true.extend(torch.argmax(batch_y, dim=1).cpu().numpy())  
        y_pred.extend(predicted.cpu().numpy())
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    cm = confusion_matrix(y_true, y_pred)
    print("Test Set:")
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print("Confusion Matrix:", cm)
    print()