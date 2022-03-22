import numpy as np
from sgicp import clusters
from sgicp.utils import transforms2D, plotting2D
import copy
from sgicp.gicp import gicp
import matplotlib.pyplot as plt

np.random.seed(seed=2)

classLabels = ("table","MEP","chair","pillar","clutter")
xrange = (-2,2)
yrange = (-1,3)
sigmarange = (-0.5,0.5)

cluster1 = clusters.cluster2()
cluster1.fillClusterRandomly(4,xrange, yrange, sigmarange, classLabels)

x_true = np.array([0.5,0,np.pi/8])
R_true, t_true = transforms2D.x_to_Rt(x_true)
cluster2 = copy.deepcopy(cluster1)
cluster2.transform(R_true,t_true, transformCov = True)

ax, _ = plotting2D.prepAxes()
cluster1.plot(ax = ax, plotIndex = True,plotCov = True, markerSize = 30, markerColor = 'b', markerShape = 'o')
cluster2.plot(ax = ax, plotIndex = True,plotCov = True, markerSize = 30, markerColor = 'r', markerShape = 'o')
ax.set_title('initial')

invx_est, fmin, itr, df, i =  gicp(cluster2.points, cluster2.covariances, #source
                                        cluster1.points,cluster1.covariances, #target
                                        n_iter_max = 50,
                                        x0 = np.zeros(3),
                                        n = 1)
                                        #x0 = utils.transforms2D.inverse_x(x_true))
x_est = transforms2D.inverse_x(invx_est)
invEst_R, invEst_t = transforms2D.x_to_Rt(invx_est)
cluster2.transform(invEst_R,invEst_t, transformCov = True)

ax, _ = plotting2D.prepAxes()
cluster1.plot(ax = ax, plotIndex = True,plotCov = True, markerSize = 30, markerColor = 'b', markerShape = 'o')
cluster2.plot(ax = ax, plotIndex = True,plotCov = True, markerSize = 30, markerColor = 'r', markerShape = 'o')
ax.set_title('after ICP')

plt.show()
print(x_est)
print(fmin)
print(itr)
print(df)