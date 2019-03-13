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
test_num_offset = 0
for name in filenames:
    file = open(name,'r')
    data = file.readlines()
    data = data[3:]
    test = []
    lastnum = 0
    tests = []
    for datum in data:
        datum = datum.replace("[","")
        datum = datum.replace("]","")
        datum = datum.replace("\n","")
        datum = datum.split(", ")
        testnum = datum[0]
        datum[0] = int(datum[0]) + test_num_offset

        if testnum != lastnum:
            tests.append(test)
            test = []
            lastnum = testnum
        test.append(tuple(datum))
    tests.append(test)
    for test in tests:
        frame = pd.DataFrame.from_records(test,columns=labels)
        frames.append(frame)
    test_num_offset = int(testnum)

dataset = pd.concat(frames)
#%% save dataset
np.save('../data/dataset.npy',np.array(dataset))


