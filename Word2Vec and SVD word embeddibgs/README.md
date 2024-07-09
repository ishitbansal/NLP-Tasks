# Introduction to NLP

# Word2Vec and SVD Word Embeddings

#### Submitted by :- Ishit Bansal

Developed word embeddings using Singular Value Decomposition (SVD) and Skip-Gram with Negative Sampling, and evaluated their performance by using them for the downstream classification task.

## Directory Structure

```
.
├── train.csv
├── test.csv
├── svd.py
├── svd-classification.py
├── svd-eval.py
├── skip-gram.py
├── skip-gram-classification.py
├── skip-gram-eval.py
├── svd_helper.ipynb
├── skip-gram_helper.ipynb
├── Report.pdf
├── README.md

```

## Dataset

Download dataset from [here](https://iiitaphyd-my.sharepoint.com/:u:/g/personal/advaith_malladi_research_iiit_ac_in/EWjgIboHC19Ppq6Of9klUo4BlKgAqynxC0TRBURzQ0lEzA?e=tWZqY5)

Train dataset - `train.csv`

Test dataset - `test.csv`

Place these files in the same directory as this file.

## Pretrained Models

All pretrained word vectors and models are saved [here](https://iiitaphyd-my.sharepoint.com/:u:/g/personal/ishit_bansal_students_iiit_ac_in/EanDlC45CKlCvUSrsqyTIMMBqZ66dUZqFpFrV1HRKRysfg?e=6IOoky)

Download this zip folder and place all the pretrained vectors and models in it in the same directory as this file.

## Generate Word Vectors

### SVD

To generate word vectors, run: 

```bash
python svd.py
```

This would save the word vectors in `svd-word-vectors.pth`


### SkipGram with Negative Sampling

To generate word vectors, run: 

```bash
python skip-gram.py
```

This would save the word vectors in `skip-gram-word-vectors.pth`


## Train Model

### SVD

To train, the LSTM classification model using the saved word vectors, run 

```bash
python svd-classification.py
```

This would save the model in `svd-word-classification-model.pth`

### SkipGram with Negative Smapling

To train, the LSTM classification model using the saved word vectors, run 

```bash
python skip-gram-classification.py
```

This would save the model in `skip-gram-word-classification-model.pth`


## Evaluate

### SVD

To evaluate the saved model and word vectors on test set, run:

```bash
python svd-eval.py
```

This would output the performance metrics like accuracy, precision, recall, f1 score and confusion matrix on train and test set.

### SkipGram with Negative Sampling

To evaluate the saved model and word vectors on test set, run:

```bash
python skip-gram-eval.py
```

This would output the performance metrics like accuracy, precision, recall, f1 score and confusion matrix on train and test set.


## Analysis

Analysis in `Report.pdf`.

`svd_helper.ipynb` and `skip-gram_helper.ipynb` consists code for hyperparameter tuning.


