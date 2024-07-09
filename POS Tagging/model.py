import torch.nn as nn


class FFNN_Tagger(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim, p, s, hidden_layers=2, activation=nn.ReLU()):
        super(FFNN_Tagger, self).__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.activation = activation
        self.ffnn = nn.Sequential()
        self.ffnn.add_module('linear1', module=nn.Linear(
            embedding_dim*(p+s+1), hidden_dim))
        self.ffnn.add_module('act1', module=activation)
        for i in range(hidden_layers-2):
            self.ffnn.add_module(
                f'linear{i+2}', module=nn.Linear(hidden_dim, hidden_dim))
            self.ffnn.add_module(f'act{i+2}', module=activation)
        self.ffnn.add_module(
            f'linear{hidden_layers}', module=nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        out = self.embedding(x)
        out = out.view(out.size(0), -1)
        out = self.ffnn(out)
        return out


class LSTM_Tagger(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, stacks, output_dim, bidirectional=False):
        super(LSTM_Tagger, self).__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim,
                            stacks, bidirectional=bidirectional)
        self.hidden_to_tag = nn.Linear(hidden_dim, output_dim)
        if bidirectional:
            self.hidden_to_tag = nn.Linear(2*hidden_dim, output_dim)

    def forward(self, x):
        out = self.embedding(x)
        out, (_, _) = self.lstm(out)
        out = self.hidden_to_tag(out)
        return out
