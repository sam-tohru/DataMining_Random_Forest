
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn import preprocessing
# %matplotlib inline

from tqdm import tqdm
import random
from pprint import pprint

#from sklearn.model_selection import train_test_split
#from sklearn import metrics

### LOAD & PREP DATA ###
# think about splitting to smaller set for original testing

data = pd.read_csv("unsw_10_test.csv", delimiter=',')
data = data.drop(['pkSeqID', 'category', 'subcategory'], axis=1)
data = data.rename(columns={'attack': 'label'})

'''
df = pd.read_csv("iris.csv")
df = df.drop("Id", axis=1)
df = df.rename(columns={"species": "label"})
'''
print(data.head())
print('------------------------------------------------------------')
#print(data.info())

# Encoding -> may leave out, requires preprossing import

print("encoding...")
enc = preprocessing.LabelEncoder()
# data = data.apply(enc.fit_transform)
list_of_cols_to_encode = ['proto', 'saddr', 'sport', 'daddr', 'dport']
data[list_of_cols_to_encode] = data[list_of_cols_to_encode].apply(enc.fit_transform)

print(data.head())
print(data.dtypes)

print('Splitting data for Training-Testing...')
ID = data.index.tolist()
test_size = round(0.30 * len(data)) # 30% of random data for testing
test_ID = random.sample(population=ID, k=test_size)

test = data.loc[test_ID]
train = data.drop(test_ID)

# Seeing test/train info -> un-comment to see for debuggin
'''
print(test.head())
print(test.info())
print('------------------------------------------------------------')
print(train.head())
print(train.info())
'''

### HELPER FUNCTIONS ###
# turns into numpy 2D array (goes much faster this way)
tnV = train.values
#print(tnV[:5])

# checks if data is pure -> just 1 label 
def check_purity(df):
	label_col = df[:, -1]
	unique = np.unique(label_col)
	if len(unique) == 1:
		return True
	else:
		return False

# debug >= 0 -> False | > 0 -> True
#print(check_purity(data[data.label >= 0].values))
#print(len(data[data.label == 0].values))
#print(len(data[data.label == 1].values))

# Classify 
def classify_data(df):
	# 1 -> is an attack, 0 -> not an attack
	label_col = df[:, -1]
	unique, counts_unique = np.unique(label_col, return_counts=True)
	index = counts_unique.argmax()
	classification = unique[index]
	return classification

#print(classify_data(data[data.label >= 1].values))


# Potential splits -> returns list of splits

def get_potential_splits(df):

	potential_splits = {}
	_, n_col = df.shape
	
	for i in range(n_col - 1):
		potential_splits[i] = []
		val = df[:, i]
		unique = np.unique(val)

		for j in range(len(unique)):
			if j != 0:
				curr = unique[j]
				prev = unique[j - 1]
				potential_split = (curr + prev) / 2
				potential_splits[i].append(potential_split)

	return potential_splits


# split data -> 
# df : data
# col : column to split on
# val : where on col to split
def split_data(df, col, val):
	
	col_values = df[:, col]
	data_below = df[col_values <= val]
	data_above = df[col_values > val]

	return data_below, data_above


# Entropy functions - Lowest overall entropy
# calculate entropy
def calc_entropy(df):

	label_col = df[:, -1]
	_, count = np.unique(label_col, return_counts=True)

	prob = count / count.sum()
	entropy = sum(prob * -np.log2(prob))

	return entropy

def calc_overall_ent(data_above, data_below):

	n_datapoints = len(data_below) + len(data_above)
	p_data_below = len(data_below) / n_datapoints
	p_data_above = len(data_above) / n_datapoints

	overall_ent = (p_data_below * calc_entropy(data_below) + p_data_above * calc_entropy(data_above))
    
	return overall_ent 

# Determines best split locations for splitting stage
# data : numpy array of data (tnV)
# potential_splits : value from get_potential_splits()
def best_split(data, potential_splits):
    
	overall_entropy = 9999
	for i in tqdm(potential_splits):
		for j in tqdm(potential_splits[i]):
			data_below, data_above = split_data(data, col=i, val=j)
			curr_overall = calc_overall_ent(data_below, data_above)

			if curr_overall <= overall_entropy:
				overall_entropy = curr_overall
				best_split_col = i
				best_split_val = j

	return best_split_col, best_split_val

potential_splits = get_potential_splits(tnV)
a = best_split(tnV, potential_splits)
print(a)