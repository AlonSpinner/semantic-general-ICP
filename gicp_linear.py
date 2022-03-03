from scipy.optimize import fmin_cg
import numpy as np
from utils.transforms2D import dR2, x_to_Rt
from sklearn.neighbors import NearestNeighbors
 
def gicp_linear(sourcePoints, sourceCov, targetPoints, targetCov, 
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
        R, t = x_to_Rt(x)
        TsourcePoints = R @ sourcePoints + t
        neigh = NearestNeighbors(n_neighbors = n)
        neigh.fit(targetPoints.reshape(-1,2))
        i = neigh.kneighbors(TsourcePoints.reshape(-1,2), return_distance = False)

        #argmin
        invCov = computeInvCov(sourceCov, targetCov[i], R)
        fun = lambda x: loss(x, sourcePoints , targetPoints[i], invCov)
        jac = lambda x: grad(x, sourcePoints , targetPoints[i], invCov)
        out = fmin_cg(f = fun, fprime = jac, x0 = x, disp = False, full_output = True)
        x = out[0]; fmin = out[1]

        #logistics
        df = abs(fmin - fminPrev)
        if itr == n_iter_max or df < tol:
            break
        fminPrev = fmin
        itr += 1

    return x, fmin, itr, df

def computeInvCov(aCov, bCov, R):
    '''
    aCov: source points covariance mx2x2
    bCov: target points covariance mxnx2x2
    '''
    m = len(aCov)
    n = len(bCov[0])

    invCov = np.zeros_like(bCov)
    for i in range(m):
        for j in range(n):
            invCov[i][j] = np.linalg.inv(bCov[i][j] + R @ aCov[i] @ R.T)
    return invCov


def lossPair(R,t,a,b,invCov):
    '''
    a : source point 2x1
    b : target point 2x1
    invCov:  2x2 inverse covariance matrix
    '''
    d = b - (R @ a + t)
    loss = d.T @ invCov @ d
    return np.asscalar(loss)

def gradPair(x,a,b,invCov):
    '''
    x : (x,y,theta) representing transform
    a : source point 2x1
    b : target point 2x1
    invCov: 2x2 inverse covariance matrix
    '''

    R, t = x_to_Rt(x)
    d = b - (R @ a + t)

    #computations copied from "SEMANTIC ICPTHROUGHEM "Semantic Iterative Closest Point through Expectation-Maximization"
    grad_t = -2 * invCov @ d #2x1
    grad_R = -2 * invCov @ d @ a.T
    
    grad_theta = np.sum(grad_R * dR2(x[2])) #assume R is made of 4 independent variables... to get this correctly we need lie algebra
    # https://proceedings.neurips.cc/paper/2009/file/82cec96096d4281b7c95cd7e74623496-Paper.pdf
    # https://github.com/EPFL-LGG/RotationOptimization/blob/master/doc/OptimizingRotations.pdf

    grad = np.array([grad_t[0,0],grad_t[1,0],grad_theta])
    return grad
def loss(x,a,b,invCov):
    '''
    x : (x,y,theta) representing transform
    a : source points mx2x1
    b : target point mxnx2x1
    invCov:inverse covariance mxnx2x2
    '''
    m = len(a)
    n = len(b[0])
    R, t = x_to_Rt(x)

    loss = np.zeros((n*m))
    for i in range(m):
        for j in range(n):
            loss[i*n+j] = lossPair(R,t,a[i],b[i][j],invCov[i][j])
    return np.sum(loss)

def grad(x,a,b,invCov):
    '''
    x : (x,y,theta) representing transform
    a : source points 2xm
    b : target point 2xm
    invCov:inverse covariance mxnx2x2
    '''
    m = len(a)
    n = len(b[0])

    grad = np.zeros((n*m,3))
    for i in range(m):
        for j in range(n):
            grad[i*n+j,:] = gradPair(x,a[i],b[i][j],invCov[i][j])
    return np.sum(grad,axis=0)
