import numpy as np
import matplotlib.pyplot as plt
from statistics import mean, median, stdev
from numpy import random
from mpl_toolkits.mplot3d import Axes3D
from enum import IntEnum
import copy

class DataMapping(IntEnum):
    TIME = 0
    DISTANCE = 1
    DIRECTION = 2
    TRAINCOUNT = 3
    EVENTCOUNT = 4
    TRAINCOUNT_CORRELATED = 5
    EVENTCOUNT_WRONG = 6

weights = {
    0 : 0,
    0.25 : 1,
    0.5 : 2,
    0.75 : 3,
    1 : 0,
}

trainCounts = {
    1 : 0,
    5 : 1,
    10 : 2,
    20 : 3,
    50 : 4,
    100 : 5,
}    

#first we want to read in the data
dataFile = open('weights.csv', 'r')

dataContainer = []

#this is our base object for reading in the data
#dataContainer[runSize][time][spacial][direction] will hold a list of values
for run in trainCounts:
    dataContainer.append([])
    for time in weights:
        dataContainer[-1].append([])
        for distance in weights:
            dataContainer[-1][-1].append([])
            for direction in weights:
                dataContainer[-1][-1][-1].append([])

#these are the specific objects that will hold the data which will
#then e.g. be used for aggregated and averages

correlatedRunCounts = copy.deepcopy(dataContainer)
correlatedRunCountsAveraged = copy.deepcopy(dataContainer)
ratioOfWronglyCorrelated = copy.deepcopy(dataContainer)
ratioOfWronglyCorrelatedAveraged = copy.deepcopy(dataContainer)


# this array should hold all values in the following schema:
# correlatedRunCounts[runSize][time][spacial][direction].
# the values are in a string seperated by ,

for line in dataFile:
    #remove the trailing newline
    line = line.rstrip()
    values = line.split(',')
    time = float(values[DataMapping.TIME.value])
    distance = float(values[DataMapping.DISTANCE.value])
    direction = float(values[DataMapping.DIRECTION.value])
    trainCount = float(values[DataMapping.TRAINCOUNT.value])
    trainCountCorrelated = float(values[DataMapping.TRAINCOUNT_CORRELATED.value])
    ratioWrong = float(values[DataMapping.EVENTCOUNT_WRONG.value]) / float(values[DataMapping.EVENTCOUNT.value])
    correlatedRunCounts[trainCounts[trainCount]][int(time*4)][int(distance*4)][int(direction*4)].append(trainCount / trainCountCorrelated)
    ratioOfWronglyCorrelated[trainCounts[trainCount]][int(time*4)][int(distance*4)][int(direction*4)].append(ratioWrong)


# print(correlatedRunCounts)

# print(ratioOfWronglyCorrelated)

# calculate the averages 
for trainCount, trainCountData in enumerate(correlatedRunCounts):
    for timeWeight, timeWeightData in enumerate(trainCountData):
        for distanceWeight, distanceWeightData in enumerate(timeWeightData):
            for directionalWeight, data  in enumerate(distanceWeightData):
                correlatedRunCountsAveraged[trainCount][timeWeight][distanceWeight][directionalWeight] = mean(data)

for trainCount, trainCountData in enumerate(ratioOfWronglyCorrelated):
    for timeWeight, timeWeightData in enumerate(trainCountData):
        for distanceWeight, distanceWeightData in enumerate(timeWeightData):
            for directionalWeight, data  in enumerate(distanceWeightData):
                ratioOfWronglyCorrelatedAveraged[trainCount][timeWeight][distanceWeight][directionalWeight] = mean(data)

fig, (ax2, ax1) = plt.subplots(1, 2)
fig.tight_layout()
# fig.suptitle('Korrelationsgenauigkeit bei zwei verschiedenen Zuganzahlen')
plt.subplots_adjust(wspace=0.3)


#first diagram
heatmap = np.zeros([5,5])
heaptMapDirectionalWeight = 2
for time in range(len(weights)):
    for spacial in range(len(weights)):
        # the (0,0) is in the bottem left corner, so we have to switch around the y axis,
        # also convert to correct correlation by using 1 - value
        heatmap[len(weights) - time - 1][spacial] = 1 - ratioOfWronglyCorrelatedAveraged[trainCounts[50]][time][spacial][heaptMapDirectionalWeight]
        # heatmap[len(timeweights) - time - 1][spacial] = time/4
im1 = ax1.imshow(heatmap, cmap='brg', vmin=0.3, vmax=0.7)

ax1.set_xticks(np.arange(len(weights)))
ax1.set_yticks(np.arange(len(weights)))

ax1.set_title('50 Züge')

ax1.set_xlabel('Zeitgewichtung')
ax1.set_ylabel('Ortsgewichtung')


ax1.set_xticklabels(weights)
ax1.set_yticklabels(weights)

plt.setp(ax1.get_xticklabels(), rotation=45, ha="right",
        rotation_mode="anchor")

for i in range(len(weights)):
    for j in range(len(weights)):
        text = ax1.text(j, i, round(heatmap[i, j],3),
                    ha="center", va="center", color="black")

#second diagram
heatmap = np.zeros([5,5])
heaptMapDistanceWeight = 1
for time in range(len(weights)):
    for spacial in range(len(weights)):
        # the (0,0) is in the bottem left corner, so we have to switch around the y axis,
        # also convert to correct correlation by using 1 - value
        heatmap[len(weights) - time - 1][spacial] = 1 - ratioOfWronglyCorrelatedAveraged[trainCounts[10]][time][spacial][heaptMapDistanceWeight]
        # heatmap[len(timeweights) - time - 1][spacial] = time/4
im1 = ax2.imshow(heatmap, cmap='brg', vmin=0.3, vmax=0.7)

ax2.set_xticks(np.arange(len(weights)))
ax2.set_yticks(np.arange(len(weights)))

ax2.set_title('10 Züge')

ax2.set_xlabel('Zeitgewichtung')
ax2.set_ylabel('Ortsgewichtung')


ax2.set_xticklabels(weights)
ax2.set_yticklabels(weights)

plt.setp(ax2.get_xticklabels(), rotation=45, ha="right",
        rotation_mode="anchor")

for i in range(len(weights)):
    for j in range(len(weights)):
        text = ax2.text(j, i, round(heatmap[i, j],3),
                    ha="center", va="center", color="black")

# generate a graph for time and distance being static with different direction weights 
timeWeight = 0.5
distanceWeight = 0.75
runSize = 50

accuracies = []

for directionalWeight in weights:
    accuracies.append(ratioOfWronglyCorrelatedAveraged[trainCounts[50]][int(timeWeight * 4)][int(distanceWeight * 4)][int(directionalWeight*4)])

# plt.plot(accuracies, [0, 0.25, 0.5, 0.75, 1])

fig, directional = plt.subplots()
# fig.suptitle('Korrelationsgenauigkeit bei unterschiedlicher Richtungsgewichtung bei 50 Zügen \n Zeitgewichtung = 0.5, Ortsgewichtung = 0.75')

weightKeys = list(weights.keys())

im = directional.plot(weightKeys, accuracies)
directional.set_ylim([0.4, 0.6])
# directional.set_xticks(np.arange(len(directionalweights)))
# directional.set_xticklabels(directionalweights)

for j in range(len(weights)):
        text = directional.text(0, j, weightKeys[j],
                    ha="center", va="center", color="black")


directional.set_xlabel('Richtungsgewichtung')
directional.set_ylabel('Korrelationsgenauigkeit')

plt.legend()
plt.show()