
import numpy as np 
import pandas as pd 
from sklearn import preprocessing

import random 
from pprint import pprint

from dt_func import * 

import time 


### LOAD & PREP DATA ###

# iris.csv
data = pd.read_csv("iris.csv")
data = data.drop("Id", axis=1)
data = data.rename(columns={"species": "label"})

# unsw botnet
# data = pd.read_csv("unsw_10_train.csv", delimiter=',')
# data = pd.read_csv("unsw_10_best.csv", delimiter=';', low_memory=False)
# data = data.drop(['pkSeqID', 'attack', 'subcategory'], axis=1)
# data = data.rename(columns={'category': 'label'})

# encoding -> for UNSW botnet dataset (not needed for Iris.csv)
# print("encoding...")
# enc = preprocessing.LabelEncoder()
# list_of_cols_to_encode = ['proto', 'saddr', 'sport', 'daddr', 'dport']
# data[list_of_cols_to_encode] = data[list_of_cols_to_encode].apply(enc.fit_transform)

# train-test split
random.seed(0)
train, test = train_test_split(data, 0.2) # 20% will be used for testing

### MAIN ###

tree = decision_tree(train)
pprint(tree)

example = test.iloc[2]
classi = classify(example, tree)

classacc = calc_acc(test, tree)
print('accuracy:', classacc) 

print()
print("You have reached the end and the input() function is keeping the window open - press Enter...")
input()