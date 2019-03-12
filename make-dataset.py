#%% imports
import numpy as np
import pandas as pd
import os
import sys
if sys.version_info[0] < 3: 
    import StringIO
else:
    from io import StringIO

#%% Get filenames
def getFilenames(dirname = "../data"):
    filenames = []
    for file in os.listdir(dirname):
        if file.endswith(".txt"):
            filenames.append(os.path.join(dirname,file))
    return filenames

filenames = getFilenames()

#%% Line by line get dataset
frames = []
labels = ['test_num', 'location', 'x', 'y', 'z', 't', 'weight']

for name in filenames:
    file = open(name,'r')
    data = file.readlines()
    test = []
    lastnum = 0
    tests = []
    for datum in data:
        datum = datum.replace("[","")
        datum = datum.replace("]","")
        datum = datum.replace("\n","")
        datum = datum.split(", ")
        testnum = datum[0]

        if testnum != lastnum:
            tests.append(test)
            test = []
            lastnum = testnum
        test.append(tuple(datum))
    tests.append(test)
    tests = np.array(tests[3:])
    for test in tests:
        frame = pd.DataFrame.from_records(test,columns=labels)
        frames.append(frame)

dataset = pd.concat(frames)
dataset.to_pickle('../data/all_data.pkl')


if False:
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
