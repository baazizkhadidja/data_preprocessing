import datasets
import matplotlib.pyplot as plt 
import numpy as np

import nltk
import torch
from torch.utils.data import Dataset, DataLoader


train_data, test_data = datasets.load_dataset("imdb", split=["train", "test"])

train_text = [sample['text'] for sample in train_data]
train_data
train_text

seq_len=[len(text.split()) for text in train_text]

plt.hist(seq_len)
plt.show()

np.mean(seq_len) # trucuation size

class Tokenizer:
    def __init__(self, stop_words, puncts, truncation_size=256):
        self.stop_words=stop_words
        self.puncts=puncts
        self.vocab = {}
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
            tokens=self.format_string(text)
            for token in tokens:
                if token in self.vocab:
                    self.vocab[token]+=1
                else:
                    self.vocab[token]=1
                    
        self.vocab['<pad>']=1    # padding : remplir le tweet par 'pad' pour que tous les tweets auront la meme taille       
        self.vocab['<unk>']=1
        self.w2i = {k:idx for idx,k in enumerate(self.vocab)}
        self.i2w = {idx:k for k, idx in  self.w2i.items()}
        
 #Import stop words       
from nltk.corpus import stopwords

nltk.download('stopwords')


import string



stop_words = stopwords.words("english")
puncts = [punt for punt in string.punctuation]

tokenizer = Tokenizer(stop_words, puncts)
tokenizer.fit(train_text)

#len(tokenizer.w2i)

#len(tokenizer.tokenize(train_text[0]))



tokenizer.detokenize(tokenizer.tokenize(train_text[2], truncation=True))


#tokenizer.tokenize(train_text[0], truncation=True)


class CustomDataset(Dataset):
    def __init__(self, train_data, tokenizer, return_type='BoG'):
        """
        Initialize the dataset with data and targets.
        Args:
            data: The input data (e.g., features).
            targets: The corresponding labels or targets.
        """
        self.train_data = train_data
        self.tokenizer = tokenizer
        self. return_type=return_type # BoG, Tf-Idf Ids
    def __len__(self):
        """
        Return the total number of samples.
        """
        return len(self.train_data)

    def __getitem__(self, idx):
        """
        Retrieve a sample and its target at the given index.
        """
        text,label= self.train_data[idx]['text'], self.train_data[idx]['label']
        if self.return_type=='Ids'  :       
    
            idxs= np.array(self.tokenizer.tokenize(text, truncation=True)).astype('int32')
            return idxs, label
        elif self.return_type=='BoG':
            vec=np.zeros(len(self.tokenizer.w2i))
            idxs= self.tokenizer.tokenize(text)
            for idx in idxs:
                vec[idx]+=1
            return vec, label    
        else:
            tf=np.zeros(len(self.tokenizer.w2i))
            idxs= self.tokenizer.tokenize(text)
            for idx in idxs:
                tf[idx]+=1
            tf/=len(idxs) 

            for i in range(len(tf)):
                tf[i]*=self.tokenizer.idf[i]
            return tf, label


train_dataset = CustomDataset(train_data, tokenizer,return_type='Tf-Idf' )
test_dataset = CustomDataset(test_data, tokenizer)


    # Example usage
data = torch.tensor(([1,2], [3,4], [5,6]))
targets = torch.tensor([0,1,0])
dataset = CustomDataset(data, targets)

 #print("Number of samples: ",len(dataset))
# print("First sample: ", dataset[0])
  
  
idx = 99

text, label = train_data[idx]['text'] , train_data[idx]['label'] 
    
    
np.array(tokenizer.tokenize(text, truncation=True)).astype('int32')
        
dataset = CustomDataset(train_data, tokenizer)  

dataset[0]   

#Define a DataLoader

def custom_collate(batch):
    inputs, targets = zip(*batch)
    # Example: Convert to tensors or handle variable-length sequences
    return torch.tensor(inputs), torch.tensor(targets)

dataloader = DataLoader(dataset, batch_size=2, collate_fn=custom_collate)
        
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)

test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False)
batch=next(iter(train_dataloader))