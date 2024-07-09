import re

class Tokenizer:
    def __init__(self):
        self.sentence_tokenizer = re.compile(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s')
        self.word_tokenizer = re.compile(r'(?:<HASHTAG>|<NUM>|<MENTION>|<URL>|<MAILID>)|\b\w+\b|[^\w\s]')
        self.number_tokenizer = re.compile(r'\b\d+(\.\d+)?\b')
        self.mail_id_tokenizer = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,4}\b')
        self.punctuation_tokenizer = re.compile(r'[^\w\s<>]')
        self.url_tokenizer = re.compile(r'(https:\/\/www\.|http:\/\/www\.|https:\/\/|http:\/\/)?[a-zA-Z]{2,}(\.[a-zA-Z]{2,})(\.[a-zA-Z]{2,})?\/[a-zA-Z0-9]{2,}|((https:\/\/www\.|http:\/\/www\.|https:\/\/|http:\/\/)?[a-zA-Z]{2,}(\.[a-zA-Z]{2,})(\.[a-zA-Z]{2,})?)|(https:\/\/www\.|http:\/\/www\.|https:\/\/|http:\/\/)?[a-zA-Z0-9]{2,}\.[a-zA-Z0-9]{2,}\.[a-zA-Z0-9]{2,}(\.[a-zA-Z0-9]{2,})?')
        self.hashtag_tokenizer = re.compile(r'#\w+')
        self.mention_tokenizer = re.compile(r'@\w+')

    def tokenize_sentences(self, text):
        return re.split(self.sentence_tokenizer, text)

    def tokenize_words(self, sentence):
        return re.findall(self.word_tokenizer, sentence)
    
    def replace_tokens(self, text):
        text = re.sub(self.mail_id_tokenizer, '<MAILID>', text)
        text = re.sub(self.hashtag_tokenizer, '<HASHTAG>', text)
        text = re.sub(self.number_tokenizer, '<NUM>', text)
        text = re.sub(self.mention_tokenizer, '<MENTION>', text)
        text = re.sub(self.url_tokenizer, '<URL>', text)
        return text

if __name__ == "__main__":
    text = input("your text: ")
    tokenizer = Tokenizer()
    text_with_placeholders = tokenizer.replace_tokens(text)
    sentences = tokenizer.tokenize_sentences(text_with_placeholders)
    tokenized_text=[]
    for sentence in sentences:
        words = tokenizer.tokenize_words(sentence)
        tokenized_text.append(words)
    print("tokenized text:",tokenized_text)