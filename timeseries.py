#%% 
import numpy as np
import pandas as pd
from tsfresh import extract_relevant_features
from tsfresh.examples.robot_execution_failures import download_robot_execution_failures, \
    load_robot_execution_failures
download_robot_execution_failures()
timeseriese, ye = load_robot_execution_failures()

#%% Load data
data=np.load('../data/dataset.npy')
labels = ['test_num', 'location', 'x', 'y', 'z', 't', 'weight']
dataset = pd.DataFrame({ labels[0]:data[:,0] , labels[1]:data[:,1] , labels[2]:data[:,2], \
labels[3]:data[:,3] , labels[4]:data[:,4] , labels[5]:data[:,5] , labels[6]:data[:,6]})

#%% Access the data method
def getTest(dataset,num):
    return dataset[dataset['test_num']==num]

#%% Extract class vector


y = pd.Series(y.astype(int))
#%% Extract relevant features

features =  extract_relevant_features(dataset, y, column_id='test_num', column_sort='t')