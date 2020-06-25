import numpy as np
import matplotlib.pyplot as plt
from statistics import mean
from numpy import random
from mpl_toolkits.mplot3d import Axes3D
from enum import IntEnum

test = np.zeros([2,2])

# data is in form of timewindow,distanceWindow,testSize (which is always 50), trainEventCount, trainRunsCorrelated, countOfWronglyCorrelated
class dataIndexes(IntEnum):
    TIME_MIN = 0
    DISTANCE_KM = 1
    TESTSIZE = 2
    TRAIN_EVENT_COUNT = 3
    TRAIN_RUNS_CORRELATED = 4
    COUNT_WRONG_CORRELATIONS = 5

timeWindows = {60:0, 360:1, 720:2, 1440:3, 2880:4}
distanceWindows = {50:0, 100:1, 250:2, 500:3}

countRatios = []
accuracies = []

for timeWindow in timeWindows:
    countRatios.append([])
    accuracies.append([])
    for distanceWindow in distanceWindows:
        countRatios[timeWindows[timeWindow]].append([])
        accuracies[timeWindows[timeWindow]].append([])

dataFile = open('windowing.csv','r')
for line in dataFile:
        # values = np.array(line)
        values = line.split(',')
        a = int(values[dataIndexes.TIME_MIN.value])
        b = int(values[dataIndexes.DISTANCE_KM.value])
        timeIndex = timeWindows[a]
        distanceIndex = distanceWindows[b]

        countRatios[timeIndex][distanceIndex].append(
            float(values[dataIndexes.TRAIN_RUNS_CORRELATED]) / float(values[dataIndexes.TESTSIZE])
        )
        accuracies[timeIndex][distanceIndex].append(
            float(values[dataIndexes.COUNT_WRONG_CORRELATIONS]) / float(values[dataIndexes.TRAIN_EVENT_COUNT])
        )

countRatiosAveraged = np.zeros([len(timeWindows), len(distanceWindows)])
countRatiosAveragedDistanceToOne = np.zeros([len(timeWindows), len(distanceWindows)])
accuraciesAveraged = np.zeros([len(timeWindows), len(distanceWindows)])

for timeIndex, row in enumerate(countRatios):
    for distanceIndex, cell in enumerate(row):
        countRatiosAveraged[len(timeWindows) - timeIndex - 1][distanceIndex] =  mean(cell)
        countRatiosAveragedDistanceToOne[len(timeWindows) - timeIndex - 1][distanceIndex] =  abs(1- mean(cell))
    
for timeIndex, row in enumerate(accuracies):
    for distanceIndex, cell in enumerate(row):
        accuraciesAveraged[len(timeWindows) - timeIndex - 1][distanceIndex] =  1 - mean(cell)



# create a heatmap from the data
fig, (traincountPlot, accuracyPlot) = plt.subplots(2)
fig.tight_layout()
# print(countRatios)

traincountPlot.imshow(countRatiosAveragedDistanceToOne, cmap='RdYlGn_r', vmin=0, vmax=1)
accuracyPlot.imshow(accuraciesAveraged, cmap='RdYlGn', vmin=0, vmax=0.7)

# draw labels for tiles
for i in range(len(timeWindows)):
    for j in range(len(distanceWindows)):
        text = traincountPlot.text(j, i, round(countRatiosAveraged[i, j],3),
                    ha="center", va="center", color="black")
        text = accuracyPlot.text(j, i, round(accuraciesAveraged[i, j],3),
                    ha="center", va="center", color="black")

traincountPlot.set_ylabel('Zeitgewichtung')
traincountPlot.set_xlabel('Ortsgewichtung')

accuracyPlot.set_ylabel('Zeitgewichtung')
accuracyPlot.set_xlabel('Ortsgewichtung')

traincountPlot.set_title('Verhätnis der auskommenden Zuganzahl / einkommende Zuganzahl')
accuracyPlot.set_title('Genauigkeit der Korrelation')


traincountPlot.set_yticks(np.arange(len(timeWindows)))
traincountPlot.set_xticks(np.arange(len(distanceWindows)))

traincountPlot.set_yticklabels(reversed(timeWindows))
traincountPlot.set_xticklabels(reversed(distanceWindows))

accuracyPlot.set_yticks(np.arange(len(timeWindows)))
accuracyPlot.set_xticks(np.arange(len(distanceWindows)))

accuracyPlot.set_yticklabels(timeWindows)
accuracyPlot.set_xticklabels(distanceWindows)

plt.show()

# print(accuracies)