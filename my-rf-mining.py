# random forest extension for decision tree

import numpy as np 
import pandas as pd 
from sklearn import preprocessing

import random 
from pprint import pprint

from dt_func import * 
from multiprocessing.dummy import Pool as ThreadPool 
import multiprocessing

### VARIABLES ### 

rand = 4 # Random elemant for random forest
bs = 800 # Bootstrap num 

max_depth = 4 # Depth of the Decision Trees
min_samples = 4 # Minimum samples for Decision Trees
n_trees = 4 # number of trees in the forest

# Multithreading variables
save_cores = 4 # amount of cores to save (i.e. 0 will use all cores, 4 will use all but 4 cores)
n_cores = multiprocessing.cpu_count() - save_cores 
multithread_bool = True # to multithread or not


### READ & PREP DATA ###

# iris.csv
# data = pd.read_csv("iris.csv")
# data = data.drop("Id", axis=1)
# data = data.rename(columns={"species": "label"})

# unsw botnet
# data = pd.read_csv("unsw_10_train.csv", delimiter=',')
data = pd.read_csv("unsw_10_best.csv", delimiter=';', low_memory=False)
data = data.drop(['pkSeqID', 'attack', 'subcategory'], axis=1)
data = data.rename(columns={'category': 'label'})

# encoding -> for UNSW botnet dataset (not needed for Iris.csv)
print("encoding...")
enc = preprocessing.LabelEncoder()
list_of_cols_to_encode = ['proto', 'saddr', 'sport', 'daddr', 'dport']
data[list_of_cols_to_encode] = data[list_of_cols_to_encode].apply(enc.fit_transform)


# -------------------------------------------------------------- # 

# train-test split
random.seed(0)
train, test = train_test_split(data, 0.2)


### RANDOM FOREST ALGORITHM ###

def bootstrap(train, n):

	boot_ID = np.random.randint(low=0, high=len(train), size=n)
	data_bs = train.iloc[boot_ID]

	return data_bs


def random_forest(train, trees, bs=bs, rand=rand, min_samples=min_samples, max_depth=max_depth):

	forest = [] # list of trees 
	for i in range(trees):
		data_bs = bootstrap(train, bs)
		tree = decision_tree(data_bs, min_samples=min_samples, max_depth=max_depth, random=rand)
		forest.append(tree)

	return forest

def multi_rf(tree_iter, train=train, bs=bs, rand=rand, min_samples=min_samples, max_depth=max_depth):
	# print('multi')

	data_bs = bootstrap(train, bs)
	tree = decision_tree(data_bs, min_samples=min_samples, max_depth=max_depth, random=rand)

	return tree


def predictions(test, tree):
	print('predicting...')
	all_pred = {}

	for i in range(len(tree)):
		print(i)
		col_name = "tree_{}".format(i)
		predictions = test.apply(classify, axis=1, args=(tree[i],)) 
		all_pred[col_name] = predictions

	all_pred = pd.DataFrame(all_pred)
	#print(all_pred)
	return predictions


def accur(pred, labels):
	correct = pred == labels
	acc = correct.mean()
	return acc


### MAIN ### 

if multithread_bool == True: # Run multithread
	print('Multihread Random Forest Running...')
	pool = ThreadPool(n_cores)
	forest = pool.map(multi_rf, range(n_trees))
	pool.close()
	pool.join()
	print('done...')

else: # Run single thread
	print('Single-Thread Random Forest Running...')
	forest = random_forest(train, n_trees)


pred = predictions(test, forest)
acc = accur(pred, test.label)

print('accuracy:', acc)


