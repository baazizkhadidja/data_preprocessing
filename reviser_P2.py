
x = -3

abs(-3)

round(3,66)

liste = [True, True, False]

all(liste)

any(liste)

x = 10 
type(x)
x =str(x)
x
type(x)

a = '20'
y = int(a)
type(y)

tuple1 = tuple([1,2,3])
l = list(tuple1)

type(l)

bin(15)
hex(1)

#x = input('entrer le nom   :  ')
#x

diccc = {
    "chien":"dog",
    "chat":"cat",
    "lapin":"rabbit",
    "oiseaux": "bird"
    }

import numpy as np

di = {
      "W1": np.random.randn(2,4),
      "b1": np.zeros((2,1)),
      "W2" : np.random.randn(2,2),
      "b2" : np.zeros((2,2))
      }

for i in range(1,3):
    print("couche", i)
    print(di[f"W{i}"])
    
    



f = open('fichier.txt', 'w')

f.write('bonjour')
f.close()

f = open('fichier.txt', 'r')
print(f.read())
f.close()

with open('fichier.txt', 'r') as f:
    print(f.read())
    
    
with open("fichier1.txt", "w") as f:
    for i in range(10):
        f.write(f"{i} est un chiffre \n")
        
        
with open("test.txt", "w") as f:
    for k, v in diccc.items(): 
        
        f.write(f"{k} s'appelle {v} en anglais \n")
    
    
import os
import glob

os.getcwd()
        
files_names = glob.glob("*.txt")

for file in files_names:
    with open(file, 'r') as f:
       print( f.read())

















    