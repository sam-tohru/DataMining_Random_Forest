
import pandas as pd
import numpy as np
from tqdm import tqdm

import random


def train_test_split(df, test_size):
    
    if isinstance(test_size, float):
        test_size = round(test_size * len(df))

    i = df.index.tolist()
    test_ID = random.sample(population=i, k=test_size)

    test = df.loc[test_ID]
    train = df.drop(test_ID)
    
    return train, test

# checks if data is pure -> just 1 label 
def check_purity(df):
	label_col = df[:, -1]
	unique = np.unique(label_col)
	if len(unique) == 1:
		return True
	else:
		return False


# Classify 
def classify_data(df):
	# 1 -> is an attack, 0 -> not an attack
	label_col = df[:, -1]
	unique, counts_unique = np.unique(label_col, return_counts=True)
	index = counts_unique.argmax()
	classification = unique[index]
	return classification


# Potential splits -> returns list of splits
# random variable used in random forest
def get_potential_splits(df, random_n):

	potential_splits = {}
	_, n_col = df.shape
	col_ID = list(range(n_col - 1))

	if random_n and random_n <= len(col_ID):
		col_ID = random.sample(population=col_ID, k=random_n)

	for i in col_ID:
		potential_splits[i] = []
		val = df[:, i]
		unique = np.unique(val)

		potential_splits[i] = unique

		# for j in range(len(unique)):
		# 	if j != 0:
		# 		curr = unique[j]
		# 		prev = unique[j - 1]
		# 		potential_split = (curr + prev) / 2
		# 		potential_splits[i].append(potential_split)

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

# random variable used for random forest
# may look at type of attack to give more sub-trees
def decision_tree(df, count=0, min_samples=2, max_depth=5, random=None):

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
		potential_splits = get_potential_splits(data, random)
		split_col, split_val = best_split(data, potential_splits)
		data_below, data_above = split_data(data, split_col, split_val)

		# check for empty 
		if (len(data_below) == 0) or (len(data_above) == 0):
			classification = classify_data(data)
			return classification			

		# sub_tree
		features = COL_NAMES[split_col]
		question = "{} <= {}".format(features, split_val)
		sub_tree = {question: []}

		# find answer to question
		yes_ans = decision_tree(data_below, count, min_samples, max_depth, random)
		no_ans = decision_tree(data_above, count, min_samples, max_depth, random)

		if yes_ans == no_ans:
			sub_tree = yes_ans
		else:
			sub_tree[question].append(yes_ans)
			sub_tree[question].append(no_ans)
		return sub_tree


### CLASSIFICATION & ACCURACY ###

def classify(example, tree):

	question = list(tree.keys())[0]
	# print(question)
	feature, comparison, value = question.split()

	# ask question
	# float(value) as feature is string
	if example[feature] <= float(value):
	# if str(example[feature]) == value:
		ans = tree[question][0]
	else:
		ans = tree[question][1]


	# base case
	if not isinstance(ans, dict):
		return ans
	else: 
		# recursion 
		return classify(example, ans)


# df = non-numpy data
def calc_acc(df, tree):
	print('calculating accuracy...')

	df['classification'] = df.apply(classify, axis=1, args=(tree,))
	df['classification_correct'] = df.classification == df.label
	accuracy = df.classification_correct.mean()

	return accuracy

