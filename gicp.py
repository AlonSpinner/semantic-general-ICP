from scipy.optimize import least_squares
import numpy as np
from utils.transforms2D import dR2, x_to_Rt
from sklearn.neighbors import NearestNeighbors
 
def generalICP(sourcePoints, sourceCov, targetPoints, targetCov, 
                x0 = np.zeros(3), n = 1,tol = 1e-6, n_iter_max = 50):

    '''
    find transform T such that target = T(source)

    sourcePoints, targetPoints: np.ndarray((m,2,1))
    sourceCov, targetCoV: np.ndarray((m,2,2))
    n: amount of nearest neighbors to compute argmin on
    '''
    x = x0
    itr = 0
    fminPrev = np.inf
    converged = False
    while not converged:

        #find data assosications
        i = mutualClosest(x,sourcePoints,targetPoints, n)
        
        #argmin
        fun = lambda x: loss(x, sourcePoints[i[:,0]] , targetPoints[i[:,1]],
                     sourceCov[i[:,0]], targetCov[i[:,1]])
        jac = lambda x: grad(x, sourcePoints[i[:,0]] , targetPoints[i[:,1]],
                     sourceCov[i[:,0]], targetCov[i[:,1]])
        out = least_squares(fun,x, jac = jac, loss = 'cauchy', f_scale = 1)
        x = out.x; fmin = out.cost

        #logistics
        df = abs(fmin - fminPrev)
        if itr == n_iter_max or df < tol:
            break

        fminPrev = fmin
        itr += 1

    return x, fmin, itr, df, i


def mutualClosest(x,a,b, n=1):
    '''
    inputs:
    a : source points, mx2x1
    b : target point, mx2x1

    outputs:
    i :  data assosication indcies [index in a, index in b], kx2
    '''

    R, t = x_to_Rt(x)
    m = R @ a + t

    neigh_b = NearestNeighbors(n_neighbors = n).fit(b.reshape(-1,2))
    m2b = neigh_b.kneighbors(m.reshape(-1,2), return_distance = False)
    
    neigh_m = NearestNeighbors(n_neighbors = n).fit(m.reshape(-1,2))
    b2m = neigh_m.kneighbors(b.reshape(-1,2), return_distance = False)

    i = []
    for im,m2b_im in enumerate(m2b): #go over points in m that point to b
        for ib in m2b_im: #go over index pointers to b 
            if im in b2m[ib]: #if point in b also points to im, take it
                i.append([im,ib])

    return np.array(i)

def lossPair(x,a,b,aCov,bCov):
    '''
    x : (x,y,theta) representing transform
    a : source point 2x1
    b : target point 2x1
    aCov,bCov ~ 2x2 covariance matrices
    '''

    R, t = x_to_Rt(x)
    d = b - (R @ a + t)
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
    d = b - (R @ a + t)
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
    b : target point mx2x1
    aCov: source points covariance mx2x2
    bCov: source points covariance mx2x2
    '''
    m = len(a)
    loss = np.zeros(m)
    for i in range(m):
            loss[i] = lossPair(x,a[i],b[i],aCov[i],bCov[i])
    return loss
def grad(x,a,b,aCov,bCov):
    '''
    x : (x,y,theta) representing transform
    a : source points 2xm
    b : target point 2xm
    aCov: source points covariance mx2x2
    bCov: source points covariance mx2x2
    '''
    m = len(a)

    grad = np.zeros((m,3))
    for i in range(m):
            grad[i,:] = gradPair(x,a[i],b[i],aCov[i],bCov[i])
    return grad
