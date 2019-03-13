#%% Imports
import numpy as np
import pandas as pd
from tsfresh import extract_relevant_features
from tsfresh.feature_extraction import MinimalFCParameters


#%% Load data
data=np.load('../data/dataset.npy')
labels = ['test_num', 'location', 'x', 'y', 'z', 't', 'weight']
dataset = pd.DataFrame({ labels[0]:data[:,0].astype(int) , labels[1]:data[:,1] , labels[2]:data[:,2], \
labels[3]:data[:,3] , labels[4]:data[:,4] , labels[5]:data[:,5] , labels[6]:data[:,6]})
y = np.load('../data/y.npy')
y = pd.Series(y.astype(int))

#%% Access the data method
def getTest(dataset,num):
    return dataset[dataset['test_num']==num]

#%% Extract relevant features

features =  extract_relevant_features(dataset, y, column_id='test_num', column_sort='t', default_fc_parameters=MinimalFCParameters())
features.to_pickle('../data/features.pkl')