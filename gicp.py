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
        R, t = x_to_Rt(x)
        TsourcePoints = R @ sourcePoints + t
        neigh = NearestNeighbors(n_neighbors = n)
        neigh.fit(targetPoints.reshape(-1,2))
        i = neigh.kneighbors(TsourcePoints.reshape(-1,2), return_distance = False)

        #argmin
        fun = lambda x: loss(x, sourcePoints , targetPoints[i],
                     sourceCov, targetCov[i])
        jac = lambda x: grad(x, sourcePoints , targetPoints[i],
                     sourceCov, targetCov[i])
        out = least_squares(fun,x, jac = jac)
        x = out.x; fmin = out.cost

        #logistics
        df = abs(fmin - fminPrev)
        if itr == n_iter_max or df < tol:
            break

        fminPrev = fmin
        itr += 1

    return x, fmin, itr, df


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
    b : target point mxnx2x1
    aCov: source points covariance mx2x2
    bCov: source points covariance mxnx2x2
    '''
    m = len(a)
    n = len(b[0])

    loss = np.zeros((n*m))
    for i in range(m):
        for j in range(n):
            loss[i*n+j] = lossPair(x,a[i],b[i][j],aCov[i],bCov[i][j])
    return loss
def grad(x,a,b,aCov,bCov):
    '''
    x : (x,y,theta) representing transform
    a : source points 2xm
    b : target point 2xm
    aCov: source points covariance mx2x2
    bCov: source points covariance mxnx2x2
    '''
    m = len(a)
    n = len(b[0])

    grad = np.zeros((n*m,3))
    for i in range(m):
        for j in range(n):
            grad[i*n+j,:] = gradPair(x,a[i],b[i][j],aCov[i],bCov[i][j])
    return grad
