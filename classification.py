#%% Imports
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import ShuffleSplit
from tsfresh.transformers import RelevantFeatureAugmenter
import pandas as pd
import numpy as np

#%% Load Data
X = pd.read_pickle('../data/features.pkl')
y = np.load('../data/y.npy')
y = pd.Series(y.astype(int))
N = len(X)


#%% Train-Test split with shuffle
rs = ShuffleSplit(n_splits=5, test_size=.25, random_state=8)
for tr,tst in rs.split(X):
    train_x = np.array(X)[tr]
    train_y = np.array(y)[tr]
    test_x = np.array(X)[tst]
    test_y = np.array(y)[tst]
#%% Random Forest Classification
forest = RandomForestClassifier(n_estimators=100)
forest.fit(train_x,train_y)
print('Training score: {0}'.format(forest.score(train_x,train_y)))
print('Test score: {0}'.format(forest.score(test_x,test_y)))

#%% KNN
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(train_x,train_y)
print('Training score: {0}'.format(knn.score(train_x,train_y)))
print('Test score: {0}'.format(knn.score(test_x,test_y)))

#%% MLP
mlp = MLPClassifier()
mlp.fit(train_x,train_y)
print('Training score: {0}'.format(mlp.score(train_x,train_y)))
print('Test score: {0}'.format(mlp.score(test_x,test_y)))
#%%
