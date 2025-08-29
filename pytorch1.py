import datasets
import matplotlib.pyplot as plt 
import numpy as np

import nltk



train_data, test_data = datasets.load_dataset("imdb", split=["train", "test"])

train_text = [sample['text'] for sample in train_data]
train_data
train_text

seq_len=[len(text.split()) for text in train_text]


plt.hist(seq_len)
plt.show()

np.mean(seq_len) # trucuation size

class Tokenizer:
    def __init__(self, stop_words, puncts):
        self.stop_words=stop_words
        self.puncts=puncts
        self.vocab = {}
        
    def tokenize(self, text):
        tokens = [token for token in text.lower().split() if not ((token in self.stop_words) or (token in self.puncts))]
        return tokens
        
    def fit(self, train_text):
        for text in train_text:
            tokens=self.tokenize(text)
            for token in tokens:
                if token in self.vocab:
                    self.vocab[token]+=1
                else:
                    self.vocab[token]=1
        self.w2i = {k:idx for idx, (k,v) in enumerate((self.vocab))}
        self.i2w = {v:k for (k,v) in  self.w2i .items()}
        
 #Import stop words       
from nltk.corpus import stopwords

nltk.download('stopwords')

#Import punctuation

import string



stop_words = stopwords.words("english")
puncts = [punt for punt in string.punctuation]

tokenizer = Tokenizer(stop_words, puncts)
tokenizer.fit(train_text)






        
        
        
