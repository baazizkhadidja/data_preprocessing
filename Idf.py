import datasets
#import matplotlib.pyplot as plt 
import numpy as np
import nltk
 #Import stop words       
from nltk.corpus import stopwords
nltk.download('stopwords')
import string




train_data, test_data = datasets.load_dataset("imdb", split=["train", "test"])

train_text = [sample['text'] for sample in train_data]


class Tokenizer:
    def __init__(self, stop_words, puncts, truncation_size=256):
        self.stop_words=stop_words
        self.puncts=puncts
        self.idf = {}
        self.truncation_size = truncation_size
        
    def format_string(self, text):
        tokens = [token for token in text.lower().split() if not ((token in self.stop_words) or (token in self.puncts))]
        return tokens
    
    def tokenize(self, text, truncation=False):
        tokens = self.format_string(text)
        tmp = []
        for token in tokens:
            if token in self.w2i :
                tmp.append(self.w2i[token])
            else:
                tmp.append(self.w2i['<unk>'])
                
        if truncation:
            tmp = tmp[: self.truncation_size] # couper le tweet trop long
            output = np.ones(self.truncation_size)*self.w2i['<pad>'] # padding : remplir le tweet court par 'pad' pour que tous les tweets auront la meme taille 
            output[:len(tmp)] = tmp
            return list(output)
        else:
            return tmp
    
    def detokenize(self, idxs):
        words = [self.i2w[idx] for idx in idxs ]
        return "".join(word+' ' for word in words )
        
    
    def fit(self, train_text):
        for text in train_text:
            tokens=set(self.format_string(text))
            for token in tokens:
                if token in self.idf:
                    self.idf[token]+=1
                else:
                    self.idf[token]=1
                    
        self.idf['<pad>']=1    # padding : remplir le tweet par 'pad' pour que tous les tweets auront la meme taille       
        self.idf['<unk>']=1
        self.w2i = {k:idx for idx,k in enumerate(self.idf)}
        self.i2w = {idx:k for k, idx in  self.w2i.items()}
                    
        
        
stop_words = stopwords.words("english")
puncts = [punt for punt in string.punctuation]

tokenizer = Tokenizer(stop_words, puncts)

tokenizer.fit(train_text)
tokenizer.idf









                    
                    