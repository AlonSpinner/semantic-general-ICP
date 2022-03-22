import numpy as np
from sgicp.clusters import cluster2
from sgicp.utils import transforms2D, plotting2D
import copy
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt


np.random.seed(seed=1)

classLabels = ("table","MEP","chair","pillar","clutter")
xrange = (-2,2)
yrange = (-1,3)
sigmarange = (-0.5,0.5)

clusterA = cluster2()
clusterA.fillClusterRandomly(4,xrange, yrange, sigmarange, classLabels)

rot = transforms2D.R2(np.radians(0))
t = np.array([[1],
              [0]])
clusterB = copy.deepcopy(clusterA)
clusterB.transform(rot,t, transformCov = True)

ax, _ = plotting2D.prepAxes()
clusterA.plot(ax = ax, plotIndex = True,plotCov = True, markerSize = 30, markerColor = 'b', markerShape = 'o')
clusterB.plot(ax = ax, plotIndex = True,plotCov = True, markerSize = 30, markerColor = 'r', markerShape = 'o')
 
neigh = NearestNeighbors(n_neighbors = 1)
neigh.fit(clusterA.points.reshape(-1,2))
i = neigh.kneighbors(clusterB.points.reshape(-1,2), return_distance = False)

plt.show()
print(i)