import numpy as np
from map import map
import copy
import gtsam
from gtsam.symbol_shorthand import L, X
from functools import partial
from typing import List, Optional

def ICP(clusterA: map,clusterB: map ,T0: np.ndarray((3,3)),threshold = 1e-5, dmax = 5):
    #Cluster: [ [location1,cov1], [location,cov2] ]
    T = T0
    bi = clusterB.getLocations()
    converged = False

    x = gtsam.symbol('x', 0)
    while not converged:
        R = T[:2,:2]; t = T[:2,[2]]
        
        TclusterB = copy.deepcopy(clusterB)
        TclusterB.transform(R,t)
        Tbi = TclusterB.getLocations()
      
        for point in cl

        for w in wi:
            if w:
                factor = gtsam.CustomFactor()
                factor = gtsam.CustomFactor(gps_model, [unknown[k]],
                                partial(error_gps, np.array([g[k]])))
                graph.addfactor(factor)

        
        
        initialvalues = gtsam.Values()
        initialvalues.insert(x,T)

        # Initialize optimizer
        params = gtsam.GaussNewtonParams()
        optimizer = gtsam.GaussNewtonOptimizer(graph,initialvalues, params)
        result = optimizer.optimize()
        T = result.x
        
        # if np.norm(T - Tcandidate) < threshold:
        #     converged = True

        # Test = Tcandidate

    return T

def solveArgmin(clusterS,clusterT,I,Tinit):
        '''
        cluster: list of dictionaries[ ("pnt": gtsam.Point2, "cov": gtsam.noiseModel), ... ]
        S - source
        T - target, this is the static point cloud.

        I - list of data assosications [ (index of point in source, index of point in target) ]

        Tinit - gtsam.Pose2 representing guess of Tsource2target
        '''

        graph = gtsam.NonlinearFactorGraph()
        initial_values = gtsam.Values()
        
        #insert X0
        X0 = gtsam.Pose2((0,0,0)) #assume target pose. makes no difference 
        X0cov = 0.001* np.eye(3)
        X0_prior_noise = gtsam.noiseModel.Gaussian.Covariance(X0cov)
        initial_values.insert(X(0), X0)
        graph.push_back(gtsam.PriorFactorPose2(X(0), X0, X0_prior_noise))

        #insert X1 (our transform) to initial values
        initial_values.insert(X(1), Tinit) 

        for da in I:
            i = da[1] #index of landmark

            #add factors for source cluster
            factor = gtsam.CustomFactor(clusterT[da[0]]["cov"], X(1), L(i), partial(errorFunction, clusterT[da[1]]["pnt"]))
            graph.push_back(factor)

            #add factors for target cluster
            factor = gtsam.PriorFactorPoint2(L(i), clusterT[da[1]]["pnt"],clusterT[da[1]]["cov"])
            graph.push_back(factor)


        params = gtsam.LevenbergMarquardtParams()
        optimizer = gtsam.LevenbergMarquardtOptimizer(graph,initial_values,params)
        result = optimizer.optimize()

        return result

def error_lm(measurement: np.ndarray, this: gtsam.CustomFactor,
             values: gtsam.Values,
             jacobians: Optional[List[np.ndarray]]) -> float:
    """Landmark Factor error function
    :param measurement: Landmark measurement, to be filled with `partial`
    :param this: gtsam.CustomFactor handle
    :param values: gtsam.Values
    :param jacobians: Optional list of Jacobians
    :return: the unwhitened error
    """

    #basically I want this delta factor: https://gtsam.org/doxygen/a05408.html
    #based on https://github.com/borglab/gtsam/blob/develop/gtsam_unstable/slam/RelativeElevationFactor.cpp

    X_key = this.keys()[0]
    L_key = this.keys()[1]

    Xest = values.atPose2(X_key)
    Lest = values.atPoint2(L_key)

    #error = (point - pose) - measurement
    error = Xest.transformTo(Lest) - measurement
    
    if jacobians is not None:
        jacobians[0] = I

    return error
