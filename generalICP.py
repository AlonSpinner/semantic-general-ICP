from scipy.optimize import fmin_cg
import numpy as np
from utils.transforms2D import dR2, x_to_Rt
from sklearn.neighbors import NearestNeighbors
import copy
 
def generalICP(sourcePoints, sourceCov, targetPoints, targetCov, 
                x0 = np.zeros(3),tol = 1e-6, n_iter_max = 100):

    '''
    find transform T such that target = T(source)

    sourcePoints, targetPoints: np.ndarray((m,2,1))
    sourceCov, targetCoV: np.ndarray((m,2,2))
    
    '''
    x = x0
    itr = 0
    fminPrev = np.inf
    converged = False
    while not converged:
        itr += 1
        
        #find data assosications
        R, t = x_to_Rt(x)
        TsourcePoints = R @ sourcePoints - t
        neigh = NearestNeighbors(n_neighbors = 1)
        neigh.fit(targetPoints.reshape(-1,2))
        i = neigh.kneighbors(TsourcePoints.reshape(-1,2), return_distance = False)

        f = lambda x: loss(x, TsourcePoints , targetPoints[i],
                     sourceCov, targetCov[i])
        fprime = lambda x: grad(x, TsourcePoints , targetPoints[i],
                     sourceCov, targetCov[i])
        out = fmin_cg(f = f, x0 = x, disp = False, full_output = True)
        x = out[0]; fmin = out[1]

        if itr > n_iter_max or abs(fmin - fminPrev) < tol:
            break

    return x  



def lossPair(x,a,b,aCov,bCov):
    '''
    x : (x,y,theta) representing transform
    a : source point 2x1
    b : target point 2x1
    aCov,bCov ~ 2x2 covariance matrices
    '''

    R, t = x_to_Rt(x)
    d = b - R @ a - t
    invCov = np.linalg.inv(bCov + R @ aCov @ R.T)
    loss = d.T @ invCov @ d
    return np.asscalar(loss)

def gradPair(x,a,b,aCov,bCov):
    '''
    x : (x,y,theta) representing transform
    a : source point 2x1
    b : target point 2x1
    aCov,bCov ~ 2x2 covariance matrices
    '''

    R, t = x_to_Rt(x)
    d = b - R @ a - t
    invCov = np.linalg.inv(bCov + R @ aCov @ R.T)

    #computations copied from "SEMANTIC ICPTHROUGHEM "Semantic Iterative Closest Point through Expectation-Maximization"
    grad_t = -2 * invCov @ d #2x1
    grad_R = -2 * invCov @ d @ (a.T + d.T @ invCov @ R @ aCov) #2x2
    
    grad_theta = np.sum(grad_R * dR2(x[2])) #assume R is made of 4 independent variables... to get this correctly we need lie algebra
    # https://proceedings.neurips.cc/paper/2009/file/82cec96096d4281b7c95cd7e74623496-Paper.pdf
    # https://github.com/EPFL-LGG/RotationOptimization/blob/master/doc/OptimizingRotations.pdf

    grad = np.array([grad_t[0,0],grad_t[1,0],grad_theta])
    return grad

def loss(x,a,b,aCov,bCov):
    '''
    x : (x,y,theta) representing transform
    a : source points mx2x1
    b : target point mxnx2x1
    aCov: source points covariance mx2x2
    bCov: source points covariance mxnx2x2
    '''

    loss = 0
    for i in range(len(a)):
        for j in range(len(b[i])):
            loss += lossPair(x,a[i],b[i][j],aCov[i],bCov[i][j])
    return loss

def grad(x,a,b,aCov,bCov):
    '''
    x : (x,y,theta) representing transform
    a : source points 2xm
    b : target point 2xm
    aCov: source points covariance mx2x2
    bCov: source points covariance mxnx2x2
    '''

    grad = 0
    for i in range(len(a)):
        for j in range(len(b[i])):
            grad += gradPair(x,a[i],b[i][j],aCov[i],bCov[i][j])
    return grad
