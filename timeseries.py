#%% 
import numpy as np
import pandas as pd
import tsfresh

#%% Load data
data=np.load('../data/dataset.npy')
labels = ['test_num', 'location', 'x', 'y', 'z', 't', 'weight']
dataset = pd.DataFrame({ labels[0]:data[:,0] , labels[1]:data[:,1] , labels[2]:data[:,2], \
labels[3]:data[:,3] , labels[4]:data[:,4] , labels[5]:data[:,5] , labels[6]:data[:,6]})

#%%

