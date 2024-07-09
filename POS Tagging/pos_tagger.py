import sys
import torch
from dataset import POSDatatset_FFNN, POSDatatset_LSTM
from model import FFNN_Tagger, LSTM_Tagger
from train import POS_Tagger

if __name__ == "__main__":
    type = sys.argv[1]
    if type == '-f':
        best_tagger = torch.load('ffnn_model.pth')
    elif type == '-r':
        best_tagger = torch.load('lstm_model.pth')
    sentence = input('> ')
    best_tagger.predict(sentence)
