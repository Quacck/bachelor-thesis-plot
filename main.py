import numpy as np
import matplotlib.pyplot as plt
from numpy import random
from mpl_toolkits.mplot3d import Axes3D

x = np.linspace(0, 2, 100)

timeweights = [0, 0.25, 0.5, 0.75, 1]
spacialweights = [0, 0.25, 0.5, 0.75, 1]


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


plt.imshow(data)
plt.show()