import sys
import torch
from dataset import POSDatatset_FFNN, POSDatatset_LSTM
from model import FFNN_Tagger, LSTM_Tagger
from train import POS_Tagger

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
        test_dataset = POSDatatset_FFNN(test_filepath, p, s, training_args)

        tagger = torch.load('ffnn_model.pth')
        accuracy, recall, f1_micro, f1_macro, confusion_mat = tagger.evaluate(
            dev_dataset)
        print("Accuracy:", accuracy)
        print("Recall:", recall)
        print("F1 micro:", f1_micro)
        print("F1 macro:", f1_macro)
        print("Confusion matrix:", confusion_mat)

    elif type == '-r':

        train_dataset = POSDatatset_LSTM(train_filepath)
        training_args = {'vocab': train_dataset.vocab, 'tags': train_dataset.tags, 'words_index': train_dataset.words_index,
                         'tags_index': train_dataset.tags_index, 'tags_one_hot': train_dataset.tags_one_hot}
        dev_dataset = POSDatatset_LSTM(dev_filepath, training_args)
        test_dataset = POSDatatset_LSTM(test_filepath, training_args)

        tagger = torch.load('lstm_model.pth')
        accuracy, recall, f1_micro, f1_macro, confusion_mat = tagger.evaluate(
            dev_dataset)
        print("Accuracy:", accuracy)
        print("Recall:", recall)
        print("F1 micro:", f1_micro)
        print("F1 macro:", f1_macro)
        print("Confusion matrix:", confusion_mat)
