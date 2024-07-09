# Introduction to NLP

# ELMO

#### Submitted by :- Ishit Bansal

Implemented deep contextualized word representations using ELMo with BiLSTM layers as described in the paper "Deep Contextualized Word Representations" by Peters et al and evaluated these embeddings on a downstream classification task.

## Directory Structure

```
.
├── ELMO.py
├── classification.py
├── eval.py
├── Report.pdf
├── README.md
├── helper.ipynb

```

## Dataset

Download dataset from [here](https://iiitaphyd-my.sharepoint.com/:u:/g/personal/advaith_malladi_research_iiit_ac_in/EWjgIboHC19Ppq6Of9klUo4BlKgAqynxC0TRBURzQ0lEzA?e=tWZqY5)

Train dataset - `train.csv`

Test dataset - `test.csv`

Place these files in the same directory as this file.

Also download 100 dimension Glove embeddings which are used as initial embeddings.


## Pretrained Models

All pretrained word vectors and models are saved [here](https://iiitaphyd-my.sharepoint.com/:u:/g/personal/ishit_bansal_students_iiit_ac_in/EQkdRRxjXBxEkGlIpoKBuTcBM8_HK8xEXqjen_b1ObOHeQ?e=ArJbje)

Download this zip folder and place all the pretrained vectors and models in it in the same directory as this file.

<li>bilstm.pth</li>
<li>classifier1.pth - lstm model trainable λs</li>
<li>classifier2.pth - lstm model frozen λs</li>
<li>classifier3.pth - lstm model learnable function</li>
<br>
Rename any to classifier.pth to evaluate it.

## Train ELMO bilstm model

To generate pretrained ELMO bilstm model, run: 

```bash
python ELMO.py
```

This would save the ELMO bilstm model in `bilstm.pth`


## Train Model

To train, the LSTM trainable classification model using the saved ELMO bilstm model `bilstm.pth`, run 

```bash
python classification.py
```

This would save the model in `classifier.pth`


## Evaluate

To evaluate the saved classifier model `classifier.pth` and ELMO bilstm model `bilstm.pth` on test set, run:

```bash
python eval.py
```

This would output the performance metrics like accuracy, precision, recall, f1 score and confusion matrix on train and test set.


## Analysis

Analysis in `Report.pdf`.

`helper.ipynb` consists code for hyperparameter tuning.


