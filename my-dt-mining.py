
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
#data = data.drop(['pkSeqID', 'attack', 'subcategory'], axis=1)
#data = data.rename(columns={'category': 'label'})


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
random.seed(0)
ID = data.index.tolist()
test_size = round(0.30 * len(data)) # 30% of random data for testing
test_ID = random.sample(population=ID, k=test_size)

test = data.loc[test_ID]
train = data.drop(test_ID)



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


### DECISION TREE ALGRORITHM ###
# if label == 1 it is an attack , if label == 0 it is not an attack
# may look at type of attack to give more sub-trees
def decision_tree(df, count=0, min_samples=2, max_depth=5):

	# df = non-numpy | data = numpy
	if count == 0: 
		global COL_NAMES # outputs feature names instead of column number
		COL_NAMES = df.columns
		data = df.values
	else:
		data = df 

	# base case
	# check purity 
	if (check_purity(data)) or (len(data) < min_samples) or (count == max_depth):
		classification = classify_data(data)
		return classification

	# recursive - if data not pure
	else:
		count += 1
		potential_splits = get_potential_splits(data)
		split_col, split_val = best_split(data, potential_splits)
		data_below, data_above = split_data(data, split_col, split_val)

		# sub_tree
		features = COL_NAMES[split_col]
		question = "{} <= {}".format(features, split_val)
		sub_tree = {question: []}

		# find answer to question
		yes_ans = decision_tree(data_below, count, min_samples, max_depth)
		no_ans = decision_tree(data_above, count, min_samples, max_depth)

		if yes_ans == no_ans:
			sub_tree = yes_ans
		else:
			sub_tree[question].append(yes_ans)
			sub_tree[question].append(no_ans)
		return sub_tree


tree = decision_tree(train)
pprint(tree)

### CLASSIFICATION & ACCURACY ###

def classify(example, tree):

	question = list(tree.keys())[0]
	feature, comparison, value = question.split()

	# ask question
	# float(value) as feature is string
	if example[feature] <= float(value):
		ans = tree[question][0]
	else:
		ans = tree[question][1]

	# base case
	if not isinstance(ans, dict):
		return ans
	else: 
		# recursion 
		return classify(example, ans)

example = test.iloc[2]
classi = classify(example, tree)
print(classi) 

print()
print('degbug stuff below')
print()

print(test.iloc[0])
print('1-')
print(list(tree.keys())[0])
print('2-')
print()

# df = non-numpy data
def calc_acc(df, tree):
	print('calculating accuracy...')

	df['classification'] = df.apply(classify, axis=1, args=(tree,))
	df['classification_correct'] = df.classification == df.label
	accuracy = df.classification_correct.mean()

	return accuracy


classacc = calc_acc(test, tree)
print('accuracy:', classacc) 