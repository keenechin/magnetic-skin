#%% imports
import numpy as np
import pandas as pd
import os
import sys
if sys.version_info[0] < 3: 
    import StringIO
else:
    from io import StringIO

#%% Get filenemaes
def getFilenames(dirname = "../data"):
    filenames = []
    for file in os.listdir(dirname):
        if file.endswith(".txt"):
            filenames.append(os.path.join(dirname,file))
    return filenames

filenames = getFilenames()

#%% Line by line get dataset
for name in filenames:
    file = open(name,'r')
    data = file.readlines()
    for datum in data:
        datum = datum.replace("[","")
        datum = datum.replace("]","")
        datum = datum.split(",")
        print(datum)


#%% Get dataset
frames = []
for name in filenames:
    file = open(name,'r')
    datastring = file.read()
    datastring = datastring.replace("[","")
    datastring = datastring.replace("]","")
    print(datastring)
    frame = pd.read_csv(StringIO(datastring))
    print(frame.shape)
    frames.append(frame)
dataset = np.array(pd.concat(frames))


#%% Get dataset
frames = []
for name in filenames:
    frame = pd.read_csv(name,sep=",",engine='python')
    array = np.array(frame)
    frames.append(pd.DataFrame(array))


dataset = np.array(pd.concat(frames))


#%%
