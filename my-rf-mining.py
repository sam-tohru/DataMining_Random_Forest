# random forest extension for decision tree

import numpy as np 
import pandas as pd 
from sklearn import preprocessing

import random 
from pprint import pprint

from dt_func import * 
from multiprocessing.dummy import Pool as ThreadPool 
import multiprocessing

### GLOBALS ###
global globRAND 
global globBS
global globDEPTH
global globTRAIN

globRAND = 4
globBS = 800
globDEPTH = 4

### READ & PREP DATA ###
# iris.csv
# data = pd.read_csv("iris.csv")
# data = data.drop("Id", axis=1)
# data = data.rename(columns={"species": "label"})

# unsw botnet (test -> smaller) dataset
data = pd.read_csv("unsw_10_train.csv", delimiter=',')
#data = data.drop(['pkSeqID', 'category', 'subcategory'], axis=1)
#data = data.rename(columns={'attack': 'label'})
data = data.drop(['pkSeqID', 'attack', 'subcategory'], axis=1)
data = data.rename(columns={'category': 'label'})

# encoding -> for unsw dataset 
print("encoding...")
enc = preprocessing.LabelEncoder()
# data = data.apply(enc.fit_transform)
list_of_cols_to_encode = ['proto', 'saddr', 'sport', 'daddr', 'dport']
data[list_of_cols_to_encode] = data[list_of_cols_to_encode].apply(enc.fit_transform)


# -------------------------------------------------------------- # 

# train-test split
random.seed(0)
train, test = train_test_split(data, 0.2)
globTRAIN = train.copy()

print('len tran:', len(globTRAIN))
print('len test:', len(test))
# print(globTRAIN.head())
# print(globTRAIN.info())

### RANDOM FOREST ALGORITHM ###

def bootstrap(train, n):

	boot_ID = np.random.randint(low=0, high=len(train), size=n)
	data_bs = train.iloc[boot_ID]

	return data_bs


def random_forest(train, trees, bs, rand, depth):
	print('running single-core random forest')
	forest = [] # list of trees 
	for i in range(trees):
		data_bs = bootstrap(train, bs)
		tree = decision_tree(data_bs, max_depth=depth, random=rand)
		forest.append(tree)

	return forest

def multi_rf(tree_iter, train=globTRAIN, bs=globBS, rand=globRAND, depth=globDEPTH):
	# print('multi')

	data_bs = bootstrap(train, bs)
	tree = decision_tree(data_bs, max_depth=depth, random=rand)

	return tree


def predictions(test, tree):
	print('pred')
	all_pred = {}

	for i in range(len(tree)):
		col_name = "tree_{}".format(i)
		predictions = test.apply(classify, axis=1, args=(tree[i],)) 
		all_pred[col_name] = predictions

	all_pred = pd.DataFrame(all_pred)
	print('end_pred')
	#print(all_pred)
	return predictions

# def tree_predictions(test, tree):
# 	predictions = test.apply(classify, axis=1, args=(tree,)) 
# 	return predictions

def accur(pred, labels):
	correct = pred == labels
	acc = correct.mean()
	return acc


# multithreading - experimental rn
# forest = []
n_trees = 4
n_cores = multiprocessing.cpu_count() - 4 # use all but 4 cores

pool = ThreadPool(4)
forest = pool.map(multi_rf, range(n_trees))
pool.close()
pool.join()
print('done')

# print('check')
# print(forest)

# forest = random_forest(train, trees=4, bs=800, rand=4, depth=4)
# print(forest)
# print(len(forest))

pred = predictions(test, forest)
acc = accur(pred, test.label)

print('accuracy:', acc)


