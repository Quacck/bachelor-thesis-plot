import numpy as np
import matplotlib.pyplot as plt
from numpy import random
from mpl_toolkits.mplot3d import Axes3D

#first we want to read in the data
dataFile = open('data_with_headers.csv', 'r')
header = dataFile.readline

timeweights = [0, 0.25, 0.5, 0.75, 1]
spacialweights = [1, 0.75, 0.5, 0.25, 0]
directionalweights = [0, 0.25, 0.5, 0.75, 1]

runSizesToIndex = {1 : 0, 5:1, 10:2, 20:3, 50:4, 100:5}

correlatedRunCounts = []
ratioOfWronglyCorrelated = []



# this array should hold all values in the following schema:
# correlatedRunCounts[runSize][time][spacial][ciretion].
# the values are in a string seperated by ,

for sizes in runSizesToIndex:
    correlatedRunCounts.append(np.full([len(timeweights), len(spacialweights), len(directionalweights)], '', dtype=object))
    ratioOfWronglyCorrelated.append(np.full([len(timeweights), len(spacialweights), len(directionalweights)], '', dtype=object))


for line in dataFile:
    #remove the trailing newline
    line = line.rstrip()
    values = line.split(',')
    time = float(values[0])
    distance = float(values[1])
    direction = float(values[2])
    runSize = float(values[3])
    ratioWrong = float(values[6]) / float(values[4])
    correlatedRunCounts[runSizesToIndex[runSize]][int(time*4)][int(distance*4)][int(direction*4)] += str(values[5] + ',')
    ratioOfWronglyCorrelated[runSizesToIndex[runSize]][int(time*4)][int(distance*4)][int(direction*4)] += str(ratioWrong) + ','


print(correlatedRunCounts)

print(ratioOfWronglyCorrelated)



x = np.linspace(0, 2, 100)




data = np.array([
[0,0.25,0.5,0.75,0],
[0,0.1,0.2,0.3,0],
[0,0.4,0.5,0.6,0],
[0,0.7,0.8,0.9,0],
[0,0.25,0.5,0.75,0],
])

fig, ax = plt.subplots()
im = ax.imshow(data)

ax.set_xticks(np.arange(len(timeweights)))
ax.set_yticks(np.arange(len(spacialweights)))

ax.set_xticklabels(timeweights)
ax.set_yticklabels(spacialweights)

plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

for i in range(len(timeweights)):
    for j in range(len(spacialweights)):
        text = ax.text(j, i, data[i, j],
                       ha="center", va="center", color="w")

plt.title("Simple Plot")
plt.legend()
# plt.show()


# plt.imshow(data)
# plt.show()