import sys
import re
import math
import random
import numpy as np
from scipy.stats import linregress
from tokenizer import Tokenizer

class N_Gram_model:
    def __init__(self,N,smoothing="n"):
        self.N=N
        self.smoothing=smoothing
        self.vocabulary=set()
        self.tokenizer=Tokenizer()

    def read_file(self,file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            self.corpus = file.read()

    def preprocess(self):
        text_with_placeholders = self.tokenizer.replace_tokens(self.corpus)
        sentences = self.tokenizer.tokenize_sentences(text_with_placeholders)
        for i in range(len(sentences)):
            sentences[i]=re.sub(r'[^a-zA-Z\s<>]', '', sentences[i])
            sentences[i]=re.sub(r'\n', ' ', sentences[i])
            sentences[i]=sentences[i].lower()
        random_numbers = random.sample(range(0,len(sentences)), 1000)
        self.test_data=[sentences[i] for i in random_numbers]
        self.train_data=[sentences[i] for i in range(len(sentences)) if i not in random_numbers]

    def add_vocabulary(self):
        sentences = self.train_data
        for sentence in sentences:
            words = self.tokenizer.tokenize_words(sentence)
            words = (['<BOS>'] * 2) + words + ['</EOS>']
            for word in words:
                self.vocabulary.add(word)

    def get_frequency_sentence(self,words,n):
        frequency = {}
        for i in range(len(words) - n + 1):
            n_gram = tuple(words[i:i+n])
            if n_gram in frequency:
                frequency[n_gram] += 1
            else:
                frequency[n_gram] = 1
        return frequency
    
    def calculate_frequencies(self):
        sentences = self.train_data
        frequency_n={}
        frequency_n_1={}
        for sentence in sentences:
            words = self.tokenizer.tokenize_words(sentence)
            words = (['<BOS>'] * (self.N-1)) + words + ['</EOS>']
            frequency1=self.get_frequency_sentence(words,self.N)
            frequency2=self.get_frequency_sentence(words,self.N-1)
            for key, value in frequency1.items():
                frequency_n[key] = frequency_n.get(key, 0) + value
            for key, value in frequency2.items():
                frequency_n_1[key] = frequency_n_1.get(key, 0) + value
        self.frequency_n=frequency_n
        self.frequency_n_1=frequency_n_1
    
    def get_log_probabilities_no_smoothing(self,sentence):
        words=self.tokenizer.tokenize_words(sentence)
        words = (['<BOS>'] * (self.N-1)) + words + ['</EOS>']
        log_probabilities=[]
        for i in range(len(words) - self.N + 1):
            n_gram = tuple(words[i:i+self.N])
            n_1_gram = tuple(words[i:i+self.N-1])
            if n_1_gram not in self.frequency_n_1 or n_gram not in self.frequency_n:
                log_probabilities.append(math.log(1e-5))
                continue
            log_probabilities.append(math.log(self.frequency_n[n_gram]/self.frequency_n_1[n_1_gram]))
        return log_probabilities
    
    def perplexity_sentence(self,sentence):
        if(self.smoothing=='n'):
            log_probabilities=self.get_log_probabilities_no_smoothing(sentence)
        if(self.smoothing=='g'):
            log_probabilities=self.get_log_probabilities_good_turing(sentence)
        if(self.smoothing=='i'):
            log_probabilities=self.get_log_probabilities_interpolation(sentence)
        perplexity=math.exp(-1*sum(log_probabilities)/len(log_probabilities))
        return perplexity
    
    def perplexity(self,data):
        sentences=data
        total_perplexity=0
        perplexity_scores={}
        for sentence in sentences:
            perplexity=self.perplexity_sentence(sentence)
            perplexity_scores[sentence]=perplexity
            total_perplexity+=perplexity
        average_perplexity=total_perplexity/len(sentences)
        return average_perplexity,perplexity_scores
    
    def train_no_smoothing(self):
        self.preprocess()
        self.add_vocabulary()
        self.calculate_frequencies()

    def get_n_r(self):
        n_r=dict()
        for value in self.frequency_n.values():
            if value in n_r:
                n_r[value]+=1
            else:
                n_r[value]=1
        return n_r
    
    def func(self):
        sentences = self.train_data
        map = {}
        for sentence in sentences:
            words = self.tokenizer.tokenize_words(sentence)
            words = (['<BOS>'] * 2) + words + ['</EOS>']
            for i in range(len(words) - 2):
                two_gram=tuple([words[i],words[i+1]])
                if two_gram in map:
                    map[two_gram].append(words[i+2])
                else:
                    map[two_gram] = [words[i+2]]
        for k in map.keys():
            map[k]=set(map[k])
        self.two_gram_map=map

    def get_z_r(self):
        n_r=self.get_n_r()
        r_values=list(set(self.frequency_n.values()))
        r_values.sort()
        z_r=dict()
        for i in range(0,len(r_values)):
            r=r_values[i]
            if(i==len(r_values)-1):
                q=r_values[i-1]
                z_r[r]=n_r[r]/(r-q)
            elif i==0:
                q=0
                t=r_values[i+1]
                z_r[r]=n_r[r]/(0.5*(t-q))
            else:
                q=r_values[i-1]
                t=r_values[i+1]
                z_r[r]=n_r[r]/(0.5*(t-q))
        return z_r
    
    def smoothing_function(self,r):
        if r==0:
            return 1
        return np.exp(self.a+self.b*(np.log(r)))
    
    def get_log_probabilities_good_turing(self,sentence):
        words=self.tokenizer.tokenize_words(sentence)
        words = (['<BOS>'] * (self.N-1)) + words + ['</EOS>']
        log_probabilities=[]
        for i in range(len(words) - self.N + 1):
            n_gram = tuple(words[i:i+self.N])
            if(n_gram not in self.frequency_n):
                r=self.smoothing_function(1)
            else:
                r=self.frequency_n[n_gram]
                r=(r+1)*self.smoothing_function(r+1)/self.smoothing_function(r)
            denominator=0
            two_gram=tuple([words[i],words[i+1]])
            if two_gram in self.two_gram_map:
                for word in self.two_gram_map[two_gram]:
                    three_gram=tuple([words[i],words[i+1],word])
                    r_=self.frequency_n[three_gram]
                    r_=(r_+1)*self.smoothing_function(r_+1)/self.smoothing_function(r_)
                    denominator+=r_
                denominator+=(len(self.vocabulary)-len(self.two_gram_map[two_gram]))*self.smoothing_function(1)
            else:
                denominator=len(self.vocabulary)*self.smoothing_function(1)
            log_probabilities.append(math.log(r/denominator))
        return log_probabilities
    
    def train_good_turing(self):
        self.preprocess()
        self.add_vocabulary()
        self.calculate_frequencies()
        self.func()
        z_r=self.get_z_r()
        r_values = np.array(list(z_r.keys()))
        zr_values = np.array(list(z_r.values()))
        slope, intercept, r_value, p_value, std_err = linregress(np.log(r_values), np.log(zr_values))
        self.a = intercept
        self.b = slope

    def calculate_frequencies_interpolation(self):
        sentences = self.train_data
        frequency_3={}
        frequency_2={}
        frequency_1={}
        for sentence in sentences:
            words = self.tokenizer.tokenize_words(sentence)
            words = (['<BOS>'] * 2) + words + ['</EOS>']
            frequency3=self.get_frequency_sentence(words,3)
            frequency2=self.get_frequency_sentence(words,2)
            frequency1=self.get_frequency_sentence(words,1)
            for key, value in frequency3.items():
                frequency_3[key] = frequency_3.get(key, 0) + value
            for key, value in frequency2.items():
                frequency_2[key] = frequency_2.get(key, 0) + value
            for key, value in frequency1.items():
                frequency_1[key] = frequency_1.get(key, 0) + value
        self.frequency_1=frequency_1
        self.frequency_2=frequency_2
        self.frequency_3=frequency_3

    
    def get_log_probabilities_interpolation(self,sentence):
        words=self.tokenizer.tokenize_words(sentence)
        words = (['<BOS>'] * 2) + words + ['</EOS>']
        log_probabilities=[]
        for i in range(len(words) - 2):
            p1=self.frequency_1.get(tuple([words[i+2]]),0)/len(self.frequency_1)
            p2=self.frequency_2.get(tuple([words[i+1],words[i+2]]),0)/self.frequency_1.get(tuple([words[i+1]]),1e5)
            p3=self.frequency_3.get(tuple([words[i],words[i+1],words[i+2]]),0)/self.frequency_2.get(tuple([words[i],words[i+1]]),1e5)
            probability=self.lambda1*p1+self.lambda2*p2+self.lambda3*p3
            if(probability==0.0):
                log_probabilities.append(math.log(1e-6))
            else:
                log_probabilities.append(math.log(self.lambda1*p1+self.lambda2*p2+self.lambda3*p3))
        return log_probabilities
    
    def train_interpolation(self):
        self.preprocess()
        self.add_vocabulary()
        self.calculate_frequencies_interpolation()
        lambda1=0
        lambda2=0
        lambda3=0
        for trigram in self.frequency_3.keys():
            if self.frequency_2[tuple([trigram[0],trigram[1]])]-1==0:
                case1=0
            else:
                case1=(self.frequency_3[trigram]-1)/(self.frequency_2[tuple([trigram[0],trigram[1]])]-1)
            if self.frequency_1[tuple([trigram[1]])]-1==0:
                case2=0
            else:
                case2=(self.frequency_2[tuple([trigram[1],trigram[2]])]-1)/(self.frequency_1[tuple([trigram[1]])]-1)
            if len(self.frequency_1)-1==0:
                case3=0
            else:
                case3=(self.frequency_1[tuple([trigram[2]])]-1)/(len(self.frequency_1)-1)
            
            if case1>case2 and case1>case3:
                lambda3+=self.frequency_3[trigram]
            elif case2>case3 and case2>case1:
                lambda2+=self.frequency_3[trigram]
            else:
                lambda1+=self.frequency_3[trigram]
        k=lambda1+lambda2+lambda3
        lambda1/=k
        lambda2/=k
        lambda3/=k
        self.lambda1=lambda1
        self.lambda2=lambda2
        self.lambda3=lambda3
    
    def generate_no_smoothing(self,sentence,k):
        words=self.tokenizer.tokenize_words(sentence)
        words=(self.N-1)*['<BOS>']+words
        n_1_gram=tuple(words[(-self.N+1):])
        outputs={}
        for word in self.vocabulary:
            n_gram=n_1_gram+(word,)
            outputs[word]=self.frequency_n.get(n_gram,0)
        outputs=dict(sorted(outputs.items(),key=lambda item:item[1],reverse=True))
        s=sum(list(outputs.values()))
        if(s==0):
            print('Context does not exist')
            return
        outputs=dict(list(outputs.items())[:k])
        print('output:')
        for key,value in outputs.items():
            print(key,value/s)

    def generate_interpolation(self,sentence,k):
        words=self.tokenizer.tokenize_words(sentence)
        words=(self.N-1)*['<BOS>']+words
        n_1_gram=tuple(words[(-self.N+1):])
        outputs={}
        for word in self.vocabulary:
            if(word=='<BOS>' or word=='</EOS>'):
                continue
            n_gram=n_1_gram+(word,)
            p1=self.frequency_1.get(tuple([n_gram[2]]),0)/len(self.frequency_1)
            p2=self.frequency_2.get(tuple([n_gram[1],n_gram[2]]),0)/self.frequency_1.get(tuple([n_gram[1]]),1e5)
            p3=self.frequency_3.get(n_gram,0)/self.frequency_2.get(n_1_gram,1e5)
            probability=self.lambda1*p1+self.lambda2*p2+self.lambda3*p3
            outputs[word]=probability
        outputs=dict(sorted(outputs.items(),key=lambda item:item[1],reverse=True))
        outputs=dict(list(outputs.items())[:k])
        print('output:')
        for key,value in outputs.items():
            print(key,value)

    def generate(self,sentence,k):
        if self.smoothing=='n':
            self.generate_no_smoothing(sentence,k)
        if self.smoothing=='i':
            self.generate_interpolation(sentence,k)
        if self.smoothing=='g':
            pass

    def save(self,output_file_path):
        with open(output_file_path, "w") as output_file:
            output_file.write(f"avg_perplexity\t{self.average_perplexity}\n")
            for sentence, perplexity in self.perplexity_scores.items():
                output_file.write(f"{sentence}\t{perplexity}\n")

    def train(self):
        if(self.smoothing=='n'):
            self.train_no_smoothing()
        if(self.smoothing=='g'):
            self.train_good_turing()
        if(self.smoothing=='i'):
            self.train_interpolation()
    
    def evaluate(self,data='test'):
        if(data=='train'):
            data=self.train_data
        if(data=='test'):
            data=self.test_data

        self.average_perplexity,self.perplexity_scores=self.perplexity(data)
        print(self.average_perplexity)

    def get_probability_sentence(self,sentence):
        if(self.smoothing=='n'):
            log_probabilities=self.get_log_probabilities_no_smoothing(sentence)
        if(self.smoothing=='g'):
            log_probabilities=self.get_log_probabilities_good_turing(sentence)
        if(self.smoothing=='i'):
            log_probabilities=self.get_log_probabilities_interpolation(sentence)
        probability=math.exp(sum(log_probabilities))
        return probability
    

if __name__ == "__main__":
    
    lm_type=sys.argv[1]
    corpus_path=sys.argv[2]
    model=N_Gram_model(3,lm_type)
    model.read_file(corpus_path)
    model.train()
    sentence=input('input sentence: ')
    sentence=re.sub(r'[^a-zA-Z\s<>]', '', sentence)
    sentence=sentence.lower()
    # print(sentence)
    print(model.get_probability_sentence(sentence))

    # lm_type='g'
    # corpus_path='./PrideAndPrejudice.txt'
    # model=N_Gram_model(3,lm_type)
    # model.read_file(corpus_path)
    # model.train()
    # model.evaluate('train')
    # model.save('2021101083_LM1_train-perplexity.txt')

    # lm_type='g'
    # corpus_path='./PrideAndPrejudice.txt'
    # model=N_Gram_model(3,lm_type)
    # model.read_file(corpus_path)
    # model.train()
    # model.evaluate('test')
    # model.save('2021101083_LM1_test-perplexity.txt')

    # lm_type='i'
    # corpus_path='./PrideAndPrejudice.txt'
    # model=N_Gram_model(3,lm_type)
    # model.read_file(corpus_path)
    # model.train()
    # model.evaluate('train')
    # model.save('2021101083_LM2_train-perplexity.txt')

    # lm_type='i'
    # corpus_path='./PrideAndPrejudice.txt'
    # model=N_Gram_model(3,lm_type)
    # model.read_file(corpus_path)
    # model.train()
    # model.evaluate('test')
    # model.save('2021101083_LM2_test-perplexity.txt')

    # lm_type='g'
    # corpus_path='./Ulysses.txt'
    # model=N_Gram_model(3,lm_type)
    # model.read_file(corpus_path)
    # model.train()
    # model.evaluate('train')
    # model.save('2021101083_LM3_train-perplexity.txt')

    # lm_type='g'
    # corpus_path='./Ulysses.txt'
    # model=N_Gram_model(3,lm_type)
    # model.read_file(corpus_path)
    # model.train()
    # model.evaluate('test')
    # model.save('2021101083_LM3_test-perplexity.txt')

    # lm_type='i'
    # corpus_path='./Ulysses.txt'
    # model=N_Gram_model(3,lm_type)
    # model.read_file(corpus_path)
    # model.train()
    # model.evaluate('train')
    # model.save('2021101083_LM4_train-perplexity.txt')

    # lm_type='i'
    # corpus_path='./Ulysses.txt'
    # model=N_Gram_model(3,lm_type)
    # model.read_file(corpus_path)
    # model.train()
    # model.evaluate('test')
    # model.save('2021101083_LM4_test-perplexity.txt')