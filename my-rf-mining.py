# random forest extension for decision tree

import numpy as np 
import pandas as pd 

import random 
from pprint import pprint

from dt_func import * 

### READ & PREP DATA ###

# wine csv 
# data = pd.read_csv('wine.csv')
# data['label'] = data.quality
# data = data.drop('quality', axis=1)

# column_names = []
# for column in data.columns:
#     name = column.replace(" ", "_")
#     column_names.append(name)
# data.columns = column_names

# print(data.head())

# wine_quality = data.label.value_counts(normalize=True)
# wine_quality = wine_quality.sort_index()

# def transform_label(value):
#     if value <= 5:
#         return "bad"
#     else:
#         return "good"

# data["label"] = data.label.apply(transform_label)

data = pd.read_csv("iris.csv")
data = data.drop("Id", axis=1)
data = data.rename(columns={"species": "label"})

# -------------------------------------------------------------- # 

# train-test split
random.seed(0)
train, test = train_test_split(data, 0.2)

### RANDOM FOREST ALGORITHM ###

def bootstrap(train, n):

	boot_ID = np.random.randint(low=0, high=len(train), size=n)
	data_bs = train.iloc[boot_ID]

	return data_bs


def random_forest(train, trees, bs, rand, depth):

	forest = [] # list of trees 
	for i in range(trees):
		data_bs = bootstrap(train, bs)
		tree = decision_tree(data_bs, max_depth=depth, random=rand)
		forest.append(tree)

	return forest

def tree_predictions(test, tree):
	predictions = test.apply(classify, axis=1, args=(tree,)) 
	return predictions

def predictions(test, tree):

	all_pred = {}

	for i in range(len(tree)):
		col_name = "tree_{}".format(i)
		predictions = test.apply(classify, axis=1, args=(tree[i],)) 
		all_pred[col_name] = predictions

	all_pred = pd.DataFrame(all_pred)

	#print(all_pred)

	return predictions

def accur(predictions, labels):
	correct = predictions == labels
	acc = correct.mean()
	return acc

forest = random_forest(train, trees=4, bs=800, rand=4, depth=4)
pred = predictions(test, forest)
acc = accur(pred, test.label)

print('accuracy:', acc)

