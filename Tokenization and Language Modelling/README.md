# Introduction to NLP

## Tokenisation and Language Modelling

#### Submitted by :- Ishit Bansal 

Developed a tokenizer and implemented language modeling with smoothing techniques such as Good-Turing and Interpolation to enhance language processing accuracy and performance.

## Directory Structure

```
.
├── tokenizer.py
├── language_model.py
├── generator.py
├── Report.pdf
├── README.md
├── PrideAndPrejudice.txt
├── Ulysses.txt

```

## Instructions to Run

> To run tokenizer.py : ``` python3 tokenizer.py ``` and then enter the text in the prompt. Output will be the tokenized text.

> To run language_model.py : ``` python3 language_model.py <lm_type> <corpus_path> ```,    ``` <lm_type> ``` can be ``` n ``` for no smoothing, ``` g ``` for good turing smoothing and ``` i ``` for linear interpolation, then enter the sentence in the prompt. Output will be the probability of the sentence.

> To run generator.py : ``` python3 generator.py <lm_type> <corpus_path> <k> ``` and then enter the input sentence in the prompt. Output will be top k most probable next word.

> Regular expressions of url and mailid are referred from internet.

> For language models, punctuations are removed from each sentence and each sentence is converted to lower case for getting better results.

> For unknown words, probability of 1e-5 is assigned.