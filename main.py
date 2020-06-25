import numpy as np
import matplotlib.pyplot as plt
from statistics import mean
from numpy import random
from mpl_toolkits.mplot3d import Axes3D

#first we want to read in the data
dataFile = open('weights.csv', 'r')

timeweights = [0, 0.25, 0.5, 0.75, 1]
spacialweights = [1, 0.75, 0.5, 0.25, 0]
directionalweights = [0, 0.25, 0.5, 0.75, 1]

runSizesToIndex = {1 : 0, 5:1, 10:2, 20:3, 50:4, 100:5}

correlatedRunCounts = []
correlatedRunCountsAveraged = []
ratioOfWronglyCorrelated = []
ratioOfWronglyCorrelatedAveraged = []


# this array should hold all values in the following schema:
# correlatedRunCounts[runSize][time][spacial][direction].
# the values are in a string seperated by ,

for sizes in runSizesToIndex:
    correlatedRunCounts.append(np.full([len(timeweights), len(spacialweights), len(directionalweights)], '', dtype=object))
    correlatedRunCountsAveraged.append(np.zeros([len(timeweights), len(spacialweights), len(directionalweights)]))
    ratioOfWronglyCorrelated.append(np.full([len(timeweights), len(spacialweights), len(directionalweights)], '', dtype=object))
    ratioOfWronglyCorrelatedAveraged.append(np.zeros([len(timeweights), len(spacialweights), len(directionalweights)]))


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


# print(correlatedRunCounts)

# print(ratioOfWronglyCorrelated)

# calculate the averages 
for size in range(len(runSizesToIndex)):
    for time in range(len(timeweights)):
        for spacial in range(len(spacialweights)):
            for direction in range(len(directionalweights)):
                values = correlatedRunCounts[size][time][spacial][direction].split(',')
                values.pop()
                values = np.float_(values)
                correlatedRunCountsAveraged[size][time][spacial][direction] = mean(values)
                values = ratioOfWronglyCorrelated[size][time][spacial][direction].split(',')
                values.pop()
                values = np.float_(values)
                ratioOfWronglyCorrelatedAveraged[size][time][spacial][direction] = mean(values)

# print(correlatedRunCountsAveraged)

# print(ratioOfWronglyCorrelatedAveraged)

# create a matrix that show the count of train runs accross timeweights and distanceweights
for key in runSizesToIndex:
    sizeIndex = runSizesToIndex[key]

    heatmap = np.zeros([5,5])
    heaptMapDistanceWeight = 1
    for time in range(len(timeweights)):
        for spacial in range(len(spacialweights)):
            heatmap[len(timeweights) - time - 1][spacial] = correlatedRunCountsAveraged[sizeIndex][time][spacial][heaptMapDistanceWeight] / key
            # heatmap[len(timeweights) - time - 1][spacial] = time/4

    fig, ax = plt.subplots()
    im = ax.imshow(heatmap)

    ax.set_xticks(np.arange(len(timeweights)))
    ax.set_yticks(np.arange(len(spacialweights)))

    ax.set_title('ratio of outcoming trainruns with ' + str(key) + ' trains as input')

    ax.set_xlabel('weight of distance')
    ax.set_ylabel('weight of time')


    ax.set_xticklabels(timeweights)
    ax.set_yticklabels(spacialweights)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
            rotation_mode="anchor")

    for i in range(len(timeweights)):
        for j in range(len(spacialweights)):
            text = ax.text(j, i, round(heatmap[i, j],3),
                        ha="center", va="center", color="w")


# show correlation accuracy
for key in runSizesToIndex:
    sizeIndex = runSizesToIndex[key]

    heatmap = np.zeros([5,5])
    heaptMapDistanceWeight = 1
    for time in range(len(timeweights)):
        for spacial in range(len(spacialweights)):
            heatmap[len(timeweights) - time - 1][spacial] = ratioOfWronglyCorrelatedAveraged[sizeIndex][time][spacial][heaptMapDistanceWeight]
            # heatmap[len(timeweights) - time - 1][spacial] = time/4

    fig, ax = plt.subplots()
    im = ax.imshow(heatmap)

    ax.set_xticks(np.arange(len(timeweights)))
    ax.set_yticks(np.arange(len(spacialweights)))

    ax.set_title('ratio of wrong correlations with ' + str(key) + ' trains as input')

    ax.set_xlabel('weight of distance')
    ax.set_ylabel('weight of time')


    ax.set_xticklabels(timeweights)
    ax.set_yticklabels(spacialweights)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
            rotation_mode="anchor")

    for i in range(len(timeweights)):
        for j in range(len(spacialweights)):
            text = ax.text(j, i, round(heatmap[i, j],3),
                        ha="center", va="center", color="w")

# generate a graph for time and distance being weighted by 1 with different 

plt.legend()
plt.show()