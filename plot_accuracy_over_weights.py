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

fig, (ax2, ax1) = plt.subplots(1, 2)
fig.tight_layout()
# fig.suptitle('Korrelationsgenauigkeit bei zwei verschiedenen Zuganzahlen')
plt.subplots_adjust(wspace=0.3)


#first diagram
heatmap = np.zeros([5,5])
heaptMapDistanceWeight = 1
for time in range(len(timeweights)):
    for spacial in range(len(spacialweights)):
        # the (0,0) is in the bottem left corner, so we have to switch around the y axis,
        # also convert to correct correlation by using 1 - value
        heatmap[len(timeweights) - time - 1][spacial] = 1 - ratioOfWronglyCorrelatedAveraged[runSizesToIndex[50]][time][spacial][heaptMapDistanceWeight]
        # heatmap[len(timeweights) - time - 1][spacial] = time/4
im1 = ax1.imshow(heatmap, cmap='brg', vmin=0.3, vmax=0.7)

ax1.set_xticks(np.arange(len(timeweights)))
ax1.set_yticks(np.arange(len(spacialweights)))

ax1.set_title('50 Züge')

ax1.set_xlabel('Zeitgewichtung')
ax1.set_ylabel('Ortsgewichtung')


ax1.set_xticklabels(timeweights)
ax1.set_yticklabels(spacialweights)

plt.setp(ax1.get_xticklabels(), rotation=45, ha="right",
        rotation_mode="anchor")

for i in range(len(timeweights)):
    for j in range(len(spacialweights)):
        text = ax1.text(j, i, round(heatmap[i, j],3),
                    ha="center", va="center", color="black")

#second diagram
heatmap = np.zeros([5,5])
heaptMapDistanceWeight = 1
for time in range(len(timeweights)):
    for spacial in range(len(spacialweights)):
        # the (0,0) is in the bottem left corner, so we have to switch around the y axis,
        # also convert to correct correlation by using 1 - value
        heatmap[len(timeweights) - time - 1][spacial] = 1 - ratioOfWronglyCorrelatedAveraged[runSizesToIndex[10]][time][spacial][heaptMapDistanceWeight]
        # heatmap[len(timeweights) - time - 1][spacial] = time/4
im1 = ax2.imshow(heatmap, cmap='brg', vmin=0.3, vmax=0.7)

ax2.set_xticks(np.arange(len(timeweights)))
ax2.set_yticks(np.arange(len(spacialweights)))

ax2.set_title('10 Züge')

ax2.set_xlabel('Zeitgewichtung')
ax2.set_ylabel('Ortsgewichtung')


ax2.set_xticklabels(timeweights)
ax2.set_yticklabels(spacialweights)

plt.setp(ax2.get_xticklabels(), rotation=45, ha="right",
        rotation_mode="anchor")

for i in range(len(timeweights)):
    for j in range(len(spacialweights)):
        text = ax2.text(j, i, round(heatmap[i, j],3),
                    ha="center", va="center", color="black")

# generate a graph for time and distance being static with different direction weights 
timeWeight = 0.5
distanceWeight = 0.75
runSize = 50

accuracies = []

for directionalWeight in directionalweights:
    accuracies.append(ratioOfWronglyCorrelatedAveraged[runSizesToIndex[50]][int(timeWeight * 4)][int(distanceWeight * 4)][int(directionalWeight*4)])

print(accuracies)
print(directionalweights)

# plt.plot(accuracies, [0, 0.25, 0.5, 0.75, 1])

fig, directional = plt.subplots()
# fig.suptitle('Korrelationsgenauigkeit bei unterschiedlicher Richtungsgewichtung bei 50 Zügen \n Zeitgewichtung = 0.5, Ortsgewichtung = 0.75')


im = directional.plot(directionalweights, accuracies)
directional.set_ylim([0.4, 0.6])
# directional.set_xticks(np.arange(len(directionalweights)))
# directional.set_xticklabels(directionalweights)

for j in range(len(directionalweights)):
        text = directional.text(0, j, directionalweights[j],
                    ha="center", va="center", color="black")


directional.set_xlabel('Richtungsgewichtung')
directional.set_ylabel('Korrelationsgenauigkeit')

plt.legend()
plt.show()