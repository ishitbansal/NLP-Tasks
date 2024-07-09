# Introduction to NLP

# POS Tagging

#### Submitted by :- Ishit Bansal

Developed POS taggers using Feedforward Neural Networks (FFNN) and Long Short-Term Memory (LSTM) networks, showcasing advanced proficiency in neural network architectures for natural language processing tasks.

## Directory Structure

```
.
├── dataset.py
├── model.py
├── train.py
├── eval.py
├── pos_tagger.py
├── helper.ipynb
├── Report.pdf
├── README.md

```

## Dataset

Download dataset by clicking [here](https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-5287/ud-treebanks-v2.13.tgz) 

Unzip the folder and place it in the same directory as this file.

## Pretrained Models

To download FFNN model, click [here](https://iiitaphyd-my.sharepoint.com/:u:/g/personal/ishit_bansal_students_iiit_ac_in/Ea-hWY3dVmBPhvW-oyP2cPgB6riah3xrtC2kw6cOgFRvlw?e=8Mogd4)

To download LSTM model, click [here](https://iiitaphyd-my.sharepoint.com/:u:/g/personal/ishit_bansal_students_iiit_ac_in/ERamNGMUFGZAtIXyIg3ZbMQBeJnP7Xz8UAzzGldTQu-cQA?e=pPj38L)

Place these pretrained models in the same directory as this file.

## Train

### FFNN

```
Hyperparameters used:

embedding_dim = 256
hidden_dim = 128
hidden_layers = 4
activation = nn.ReLU()
epochs = 10
learning_rate = 0.001
batch_size = 32
```
To modify hyperparameters, edit `train.py` accordingly.

To train FFNN model, run:

```bash
python train.py -f
```


### LSTM

```
Hyperparameters used:

embedding_dim = 64
hidden_dim = 128
stacks = 2
bidirectional = False
epochs = 10
learning_rate = 0.001
batch_size = 32
```
To modify hyperparameters, edit `train.py` accordingly.

To train LSTM model, run:

```bash
python train.py -r
```

This would save the trained models in `ffnn_model.pth` and `lstm_model.pth` files respectively and output the evaluation metrics on dev set.

## Evaluate

### FFNN 

To evaluate the saved model on dev set in `ffnn_model.pth`, run:

```bash
python eval.py -f
```

### LSTM

To evaluate the saved model on dev set in `lstm_model.pth`, run:

```bash
python eval.py -r
```


## Predict sentence

### FFNN

To use the FFNN model, run :

```bash
python postagger.py -f
```

This would load the saved model in `ffnn_model.pth` and then prompt us to enter a sentence.

After entering the sentence, predicted POS tags would be displayed.


### LSTM

To use the LSTM model, run :

```bash
python postagger.py -r
```

This would load the saved model in `lstm_model.pth` and then prompt us to enter a sentence.

After entering the sentence, predicted POS tags would be displayed.

## Analysis

Analysis in `Report.pdf`.

`helper.ipynb` consists code for graphs.


