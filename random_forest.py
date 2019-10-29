# random forest extension for decision tree

import numpy as np 
import pandas as pd 
from sklearn import preprocessing

import random 
from pprint import pprint

from dt_func import * 
from multiprocessing.dummy import Pool as ThreadPool 
import multiprocessing
# from multiprocessing import Process, Queue

import time 

### VARIABLES ### 

rand = 4 # Random elemant for random forest
bs = 800 # Bootstrap num 

max_depth = 4 # Depth of the Decision Trees
min_samples = 4 # Minimum samples for Decision Trees
n_trees = 4 # number of trees in the forest

# Multithreading variables
multithread_bool = False # MultiProcess for Random Forest Stage
MultiPred = False # Multiprocess Prediction stage 

mp_forest = multiprocessing.Queue()
mp_pred = multiprocessing.Queue()


### PROMPT TO CHANGE VARIABLES ###

print('Default Values:')
print_met(rand, bs, max_depth, min_samples, n_trees, multithread_bool, MultiPred)
print()

change_default = input("Do you want to change default (y/n): ")
if change_default in ['y', 'Y', 'yes', 'Yes', 'YES']:
	print('Leave blank for default...')
	rand = input('Random (Default = 4): ') or 4
	bs = input('bs (Default = 800): ') or 800
	max_depth = input('max_depth (Default = 4): ') or 4
	min_samples = input('min_samples (Default = 4): ') or 4
	n_trees = input('n_trees (Default = 4): ') or 4
	print('---')

change_multi = input('Do you want to use Multiple Processes (y/n): ')
if change_multi in ['y', 'Y', 'yes', 'Yes', 'YES']:
	# print('Warning Multiprocessing may work weirdly on Windows...')
	mRF = input('MultiProcess Random Forest Stage (y/n): ')
	if mRF in ['y', 'Y', 'yes', 'Yes', 'YES']:
		print('MultiProcess Random Forest On..')
		multithread_bool = True

	mP = input('MultiProcess Prediction Stage (y/n): ')
	if mP in ['y', 'Y', 'yes', 'Yes', 'YES']:
		print('MultiProcess Prediction On..')
		MultiPred = True


### READ & PREP DATA ###

# iris.csv
# data = pd.read_csv("iris.csv")
# data = data.drop("Id", axis=1)
# data = data.rename(columns={"species": "label"})

# unsw botnet - 
# test is smaller set for flo upload, 
# best is what i tested mainly on but is too large, visit: https://www.unsw.adfa.edu.au/unsw-canberra-cyber/cybersecurity/ADFA-NB15-Datasets/bot_iot.php
data = pd.read_csv("unsw_10_test.csv", delimiter=',')
# data = pd.read_csv("unsw_10_best.csv", delimiter=';', low_memory=False)
data = data.drop(['pkSeqID', 'attack', 'subcategory'], axis=1)
data = data.rename(columns={'category': 'label'})

# encoding -> for UNSW botnet dataset (not needed for Iris.csv)
print("encoding...")
enc = preprocessing.LabelEncoder()
list_of_cols_to_encode = ['proto', 'saddr', 'sport', 'daddr', 'dport']
data[list_of_cols_to_encode] = data[list_of_cols_to_encode].apply(enc.fit_transform)

# train-test split
random.seed(0)
train, test = train_test_split(data, 0.2) # 20% will be used for testing



### RANDOM FOREST ALGORITHM ###

# Bootstrap Function
def bootstrap(train, n):

	boot_ID = np.random.randint(low=0, high=len(train), size=n)
	data_bs = train.iloc[boot_ID]

	return data_bs

# Random Forest - Single & MultiProcess Functions
def random_forest(train, trees, bs=bs, rand=rand, min_samples=min_samples, max_depth=max_depth):

	forest = [] # list of trees 
	for i in range(trees):
		data_bs = bootstrap(train, bs)
		tree = decision_tree(data_bs, min_samples=min_samples, max_depth=max_depth, random=rand)
		forest.append(tree)

	return forest

def multi_rf(tree_iter, train=train, bs=bs, rand=rand, min_samples=min_samples, max_depth=max_depth):

	data_bs = bootstrap(train, bs)
	tree = decision_tree(data_bs, min_samples=min_samples, max_depth=max_depth, random=rand)
	mp_forest.put(tree)


# Predicting Functions - single & multiprocess
def predictions(test, tree):
	print('predicting...')
	all_pred = {}
	#ran = range(len(tree))

	for i in range(len(tree)):
		print(i)
		col_name = "tree_{}".format(i)
		predictions = test.apply(classify, axis=1, args=(tree[i],)) 
		all_pred[col_name] = predictions

	print('compiling list of predictions - may take a bit...')
	all_pred = pd.DataFrame(all_pred)
	#print(all_pred)

	predictions = all_pred.mode(axis=1)[0]
	return predictions

def multi_pred(tree, test=test):
	print('Multi Predicting...')

	predictions = test.apply(classify, axis=1, args=(tree,)) 

	mp_pred.put(predictions)
	# print(predictions)

	print('n2')

	# return predictions

# Accuracy Function
def accur(pred, labels):
	correct = pred == labels
	acc = correct.mean()
	return acc



### MAIN ### 

## Running Random Forest ##

if multithread_bool == True: # Run multithread
	# MultiProcess (GIL unlocked)
	print('MultiProcess Random Forest Running...')
	t0 = time.time()
	process = [multiprocessing.Process(target=multi_rf, args=(i,)) for i in range(n_trees)]

	for p in process:
		p.start()

	print('joining')
	forest = [mp_forest.get() for i in range(n_trees)]
	for p in process:
		p.join()

	t1 = time.time()
	# forest = [mp_forest.get() for i in range(n_trees)]

else: # Run single thread
	print('Single-Thread Random Forest Running...')
	t0 = time.time()
	forest = random_forest(train, n_trees)
	t1 = time.time()


## Predicting ##

if MultiPred == True: # Run multithread
	# MultiProcess (GIL unlocked)
	print('MultiProcess Predicting Running...')
	all_pred = {}
	tp0 = time.time()
	pprocess = [multiprocessing.Process(target=multi_pred, args=(forest[i],)) for i in range(len(forest))]

	for pp in pprocess:
		pp.start()

	for i in range(len(forest)):
		col_name = "tree_{}".format(i)
		all_pred[col_name] = mp_pred.get()

	# print(all_pred)
	for pp in pprocess:
		pp.join()

	print('compiling list of predictions - may take a bit...')
	all_pred = pd.DataFrame(all_pred)
	pred = all_pred.mode(axis=1)[0] # This stage is what slows it down

	tp1 = time.time()

else: # Run single thread
	print('Single-Thread Random Forest Running...')
	tp0 = time.time()
	pred = predictions(test, forest)
	tp1 = time.time()


## Accuracy & Metrics ##

pprint(forest)

time = t1 - t0
pred_time = tp1 - tp0
print()
print('total rf time:', time)
print('total predicting time:', pred_time)
print()
# print('pred length:', len(pred))
# print('pred is type:', type(pred))
# print('labels:', pred.keys())

acc = accur(pred, test.label)
print('accuracy:', acc)

print()
print("You have reached the end and the input() function is keeping the window open - press Enter...")
input()