import sys
from tokenizer import Tokenizer
from language_model import N_Gram_model
import re

if __name__ == "__main__":
    lm_type=sys.argv[1]
    corpus_path=sys.argv[2]
    k=int(sys.argv[3])
    n=3
    model=N_Gram_model(n,lm_type)
    model.read_file(corpus_path)
    model.train()
    sentence=input('input sentence: ')
    sentence=re.sub(r'[^a-zA-Z\s<>]', '', sentence)
    sentence=sentence.lower()
    model.generate(sentence,k)

    # sentence=input('input sentence: ')
    # sentence=re.sub(r'[^a-zA-Z\s<>]', '', sentence)
    # sentence=sentence.lower()
    # for n in range (2,8):
    #     model=N_Gram_model(n,lm_type)
    #     model.read_file(corpus_path)
    #     model.train()
    #     print('n =',n)
    #     model.generate(sentence,k)