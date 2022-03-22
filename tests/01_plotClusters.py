import numpy as np
from sgicp.clusters import cluster2
from sgicp.utils import transforms2D, plotting2D
import copy
import matplotlib.pyplot as plt

np.random.seed(seed=2)

classLabels = ("table","MEP","chair","pillar","clutter")
xrange = (-2,2)
yrange = (-1,3)
sigmarange = (-0.5,0.5)

clusterA = cluster2()
clusterA.fillClusterRandomly(20,xrange, yrange, sigmarange, classLabels)

rot = transforms2D.R2(np.radians(90))
t = np.array([[5],
              [0]])
clusterB = copy.deepcopy(clusterA)
clusterB.transform(rot,t, transformCov = True)

ax, _ = plotting2D.prepAxes()
clusterA.plot(ax = ax, plotIndex = True,plotCov = True, markerSize = 30, markerColor = 'b', markerShape = 'o')
clusterB.plot(ax = ax, plotIndex = True,plotCov = True, markerSize = 30, markerColor = 'r', markerShape = 'o')

plt.show()