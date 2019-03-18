#%% Imports
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#%% Load Data
X = pd.read_pickle('../data/features.pkl')
y = np.load('../data/y.npy')
y = pd.Series(y.astype(int))
N = len(X)

rs = ShuffleSplit(n_splits=5, test_size=.25, random_state=8)
for tr,tst in rs.split(X):
    train_x = np.array(X)[tr]
    train_y = np.array(y)[tr]
    test_x = np.array(X)[tst]
    test_y = np.array(y)[tst]
#%% Eval function
def evalModel(model,train_x,train_y,test_x,test_y):
    y_hat = model.predict(test_x)
    print('Training score: {0}'.format(model.score(train_x,train_y)))
    print('Test score: {0}'.format(model.score(test_x,test_y)))
    plt.matshow(confusion_matrix(test_y,y_hat))
#%% Random Forest Classification
forest = RandomForestClassifier(n_estimators=100)
forest.fit(train_x,train_y)
#%% KNN
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(train_x,train_y)
evalModel(knn,train_x,train_y,test_x,test_y)

#%% MLP
mlp = MLPClassifier(hidden_layer_sizes=(38),max_iter=int(1e4),random_state=8)
mlp.fit(train_x,train_y)
evalModel(mlp,train_x,train_y,test_x,test_y)
