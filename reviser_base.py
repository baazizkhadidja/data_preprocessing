
import pandas as pd
import numpy as np


dic = {
       "caracteristiques":["nom", "prenom", "taille", "poid", "age"],
       "per1":["baaz", "doody", 168, 74, 38],
       "per2":["baaz", "jooj", 184, 85, 35]
       
       }

df = pd.DataFrame(dic)
#df


carac = ["nom", "pren", "taille", "poid", "age"]

data = {
    "per1": ["baaz", "doody", 168, 74, 38],
    "per2": ["baaz", "jooj", 184, 85, 35],
    "per3": ["xman", "toto", 175, 70, 40]  # <-- facile d’ajouter une nouvelle personne
}

# Construction du DataFrame
df2 = pd.DataFrame(data, index = carac).T

#print(df2)

indexi = ["sara", "farah", "amir", "boudi", "doody"]

dic_sal = {
   "sal_jan" :[1700, 1500, 1260, 2500, 7000],
    "sal_fev" : [2700, 3600, 2654, 1700, 5000],
    "sal_mars" : [2599, 3456, 2345, 1234, 4321],
    "sal_avril" : [7654, 9876, 5432, 1234, 987]
    }

tablo = pd.DataFrame(dic_sal, index=indexi)

garants = {
    'jan':[3456, 6543, 9876],
    'fev':[9876, 9875, 987],
    'mars':[1234, 5432, 6574]
    }
indexo = ['Per1', 'Per2', 'Per3']

#df3 = pd.DataFrame(garants, index=indexo)


inventaire = {
    "bananes":5000,
    "pommes":2094,
    "poires":41324,
    "cerises":32165
    }

inventaire.values()
inventaire.keys()

len(inventaire)

inventaire["abricot"]=5432
inventaire


prenom = ["hicham", "khadidja", "hadjer", "abba"]
dixio={}
dixio.fromkeys(prenom, "BAAZIZ")


inventaire.pop('pommes')
inventaire

for k,v in enumerate(inventaire):
    print(k, "vaut ", v )


for k, v in inventaire.items():
    print(k,v)













import pandas as pd


dic = {
       "caracteristiques":["nom", "prenom", "taille", "poid", "age"],
       "per1":["baaz", "doody", 168, 74, 38],
       "per2":["baaz", "jooj", 184, 85, 35]
       
       }

df = pd.DataFrame(dic)
#df


carac = ["nom", "pren", "taille", "poid", "age"]

data = {
    "per1": ["baaz", "doody", 168, 74, 38],
    "per2": ["baaz", "jooj", 184, 85, 35],
    "per3": ["xman", "toto", 175, 70, 40]  # <-- facile d’ajouter une nouvelle personne
}

# Construction du DataFrame
df2 = pd.DataFrame(data, index = carac).T

#print(df2)

indexi = ["sara", "farah", "amir", "boudi", "doody"]

dic_sal = {
   "sal_jan" :[1700, 1500, 1260, 2500, 7000],
    "sal_fev" : [2700, 3600, 2654, 1700, 5000],
    "sal_mars" : [2599, 3456, 2345, 1234, 4321],
    "sal_avril" : [7654, 9876, 5432, 1234, 987]
    }

tablo = pd.DataFrame(dic_sal, index=indexi)