import datasets
import matplotlib.pyplot as plt 
import numpy as np
import string
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')


import torch
from torch.utils.data import Dataset, DataLoader

train_data, test_data = datasets.load_dataset("imdb", split=["train", "test"])

train_text=[sample['text'] for sample in train_data]
len(train_text)
seq_lens=[len(text.split()) for text in train_text]
np.mean(seq_lens) # trucation size 
np.argmax(seq_lens)
train_text[np.argmax(seq_lens)]

plt.hist(seq_lens)
plt.show() 

train_data

class Tokenizer:
  def __init__(self, stop_words, puncts, truncation_size=256 ):
      
    self.stop_words=stop_words
    self.puncts=puncts
    self.df = {}
    self.truncation_size=truncation_size
  def format_string(self, text):
      tokens=[ token for token  in text.lower().split() if not ((token in self.stop_words) or  (token in self.puncts))   ]
      return tokens  
      
  def tokenize(self, text, truncation=False):
      tokens=self.format_string(text)
      tmp=[]
      for token in tokens:
          if token in self.w2i :
              tmp.append(self.w2i[token])
          else:
              tmp.append(self.w2i['<unk>'])
              
      if truncation:
          tmp=tmp[: self.truncation_size]
          output= np.ones(self.truncation_size)*self.w2i['<pad>']
          output[:len(tmp)]=tmp
          return list(output)
      else:
          return tmp
  def detokenize(self, idxs):
      words=[self.i2w[idx] for idx in idxs]  
      
      return ''.join(word+' ' for word in words )
  def fit(self, train_text):
    for text in train_text:
        tokens=set(self.format_string(text))
        for token in tokens:
            if token in self.df:
                self.df[token]+=1
            else: 
                self.df[token]=1
    self.df['<unk>']=1
    self.df['<pad>']=1

    
    
    self.w2i = { k:idx for idx, (k,v) in enumerate(self.df.items())}
    self.i2w = { v:k for (k,v) in  self.w2i .items() }

    self.idf=np.zeros(len(self.df))
    for (k,v) in self.w2i.items():
        self.idf[v]=np.log((1+len(train_text))/(1+self.df[k]))  
        
        
stop_words= stopwords.words('english')
puncts=  [punt for punt in string.punctuation]


tokenizer=Tokenizer(stop_words , puncts)
tokenizer.fit(train_text)


tokenizer.detokenize(tokenizer.tokenize(train_text[4], truncation=True))


class CustomDataset(Dataset):
    def __init__(self, data, targets):
        """
        Initialize the dataset with data and targets.
        Args:
            data: The input data (e.g., features).
            targets: The corresponding labels or targets.
        """
        self.data = data
        self.targets = targets

    def __len__(self):
        """
        Return the total number of samples.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Retrieve a sample and its target at the given index.
        """
        return self.data[idx], self.targets[idx]

# Example usage
data = torch.tensor([[1, 2], [3, 4], [5, 6]])
targets = torch.tensor([0, 1, 0])
dataset = CustomDataset(data, targets)

print("Number of samples:", len(dataset))
print("First sample:", dataset[0])

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

train_dataset[1]

train_dataset[1][0].shape

len(tokenizer.w2i)

#Define a DataLoader


def custom_collate(batch):
    inputs, targets = zip(*batch)
    # Example: Convert to tensors or handle variable-length sequences
    return torch.tensor(inputs), torch.tensor(targets)

dataloader = DataLoader(dataset, batch_size=2, collate_fn=custom_collate)

train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)

test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False)

batch=next(iter(train_dataloader))
len(batch)
batch[0].shape





















