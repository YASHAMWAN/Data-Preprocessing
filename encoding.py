#data preprocessing

#import libraries
import numpy as np
import matplotlib as plt
import pandas as pd

# get current working directory
import os
print(os.getcwd())

#set current working directory'
os.chdir('C:/Users/ThinkPad/Desktop/Machine Learning A-Z/Part 1 - Data Preprocessing/Data_Preprocessing')
#import dataset
dataset = pd.read_csv('Data.csv')
print(dataset)

X = dataset.iloc[ :, :-1].values
y = dataset.iloc[:, 3].values
print(X)
#taking care of missing data using Univariate feature imputation
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[: ,1:3])

print(X)

#encoding the variables
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder() #create an object
X[:, 0] = labelencoder_X.fit_transform(X[:,0])

onehotencoder = OneHotEncoder(categorical_features = [1], categories = auto)
X[:, 0] = onehotencoder.fit_transform(X).toarray()
