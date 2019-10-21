# random forest extension for decision tree

import numpy as np 
import pandas as pd 

import random 
from pprint import pprint

from dt_func import * 

### READ & PREP DATA ###

# wine csv 
data = pd.read_csv('wine.csv')
data['label'] = data.quality
data = data.drop('quality', axis=1)

column_names = []
for column in data.columns:
    name = column.replace(" ", "_")
    column_names.append(name)
data.columns = column_names

print(data.head())

wine_quality = data.label.value_counts(normalize=True)
wine_quality = wine_quality.sort_index()

def transform_label(value):
    if value <= 5:
        return "bad"
    else:
        return "good"

data["label"] = data.label.apply(transform_label)

# -------------------------------------------------------------- # 

# train-test split
random.seed(0)
train, test = train_test_split(data, 0.2)

### RANDOM FOREST ALGORITHM ###

def bootstrap(train, n):

	boot_ID = np.random.randint(low=0, high=len(train), size=n)
	data_bs = train.iloc[boot_ID]

	return data_bs

