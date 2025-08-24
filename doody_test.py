
#Import libraries
import pandas as pd
import numpy as np
import matplotlib as mtp




#Import dataset
dataset = pd.read_csv('Data.csv')

dataset

X = dataset.iloc[: , :-1].values
X
y = dataset.iloc[: , 3]
y

#Taking care of missing data
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy="mean")

imputer = imputer.fit(X[ : ,1:3])

X[ : , 1:3] = imputer.transform((X[ : , 1:3]))
X[ : , 1:3]

#Encoding categorical data: pour transformer les pays en chiffre
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer


ct = ColumnTransformer(transformers=[("encoder",OneHotEncoder(),[0])],remainder = "passthrough")


X = ct.fit_transform(X)

labelEncoder_y = LabelEncoder()
y = labelEncoder_y.fit_transform(y)
