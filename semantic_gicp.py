from scipy.optimize import least_squares
import numpy as np
from utils.transforms2D import dR2, x_to_Rt
from sklearn.neighbors import NearestNeighbors
 
def semantic_gicp(sourcePoints, sourceCov, sourceLabels, 
                targetPoints, targetCov, targetLabels,
                x0 = np.zeros(3), n = 1,tol = 1e-6, n_iter_max = 50,
                solverloss = 'cauchy', solver_f_scale = 0.4):

    '''
    find transform T such that target = T(source)
    sourcePoints : np.ndarray((m_s,2,1))
    sourceCov    : np.ndarray((m_s,2,2))
    targetPoints : np.ndarray((m_t,2,1))
    targetCoV    : np.ndarray((m_t,2,2))
    n            : amount of nearest neighbors to compute argmin on
    '''
    x = x0
    itr = 0
    fminPrev = np.inf
    converged = False
    while not converged:

        #find data assosications
        i = DA_sourceClosestToTarget(x,sourcePoints,targetPoints, n)
        w = computeWeights(x, sourcePoints, sourceLabels,
                            targetPoints, targetLabels, i)
        
        #argmin
        fun = lambda x: loss(x, sourcePoints[i[:,0]] , targetPoints[i[:,1]],
                     sourceCov[i[:,0]], targetCov[i[:,1]],w)
        out = least_squares(fun,x, loss = solverloss, f_scale = solver_f_scale)
        x = out.x; fmin = out.cost

        #logistics
        df = abs(fmin - fminPrev)
        if itr == n_iter_max or df < tol:
            break

        fminPrev = fmin
        itr += 1

    return x, fmin, itr, df, i

def computeWeights(x,a,ca,b,cb,i):
    '''
    inputs:
    a  : source points, mx2x1
    ca : source points labels, list size m
    b  : target points, mx2x1
    cb : target points labels, list size m
    i  : data assosication indcies [index in a, index in b], kx2
    
    outputs:
    w  : weight for each assosication 
    '''
    R, t = x_to_Rt(x)
    m = R @ a + t

    w = np.zeros(len(i))
    for k,da in enumerate(i):
        residual = np.abs(b[da[1]] - m[da[0]])
        semantics = ca[da[0]] == cb[da[1]]
        w[k] =  1-0.5*semantics
    return w

def DA_perfect(a,b):
    return np.array([list(range(len(a))),list(range(len(b)))]).T

def DA_sourceClosestToTarget(x,a,b, n=1):
    '''
    inputs:
    a : source points, m_sx2x1
    b : target point, m_tx2x1

    outputs:
    i :  data assosication indcies [index in a, index in b], array of size kx2
    '''
    
    R, t = x_to_Rt(x)
    m = R @ a + t

    neigh_b = NearestNeighbors(n_neighbors = n).fit(b.reshape(-1,2))
    m2b = neigh_b.kneighbors(m.reshape(-1,2), return_distance = False)
    
    i = []
    for im,m2b_im in enumerate(m2b): #go over points in m that point to b
        for ib in m2b_im: #go over index pointers to b
                i.append([im,ib])
    return np.array(i)

def lossPair(x,a,b,aCov,bCov,w):
    '''
    x : (x,y,theta) representing transform
    a : source point 2x1
    b : target point 2x1
    aCov,bCov ~ 2x2 covariance matrices
    '''

    R, t = x_to_Rt(x)
    d = b - (R @ a + t)
    invCov = np.linalg.inv(bCov + R @ aCov @ R.T)
    loss = w * (d.T @ invCov @ d)
    return np.asscalar(loss)

def loss(x,a,b,aCov,bCov,w):
    '''
    x   : (x,y,theta) representing transform
    a   : source points m_sx2x1
    b   : target point m_tx2x1
    aCov: source points covariance m_sx2x2
    bCov: source points covariance m_tx2x2
    w   : wight for each lossPair. 1D array of size k
    '''
    m = len(a)
    loss = np.zeros(m)
    for i in range(m):
            loss[i] = lossPair(x,a[i],b[i],aCov[i],bCov[i],w[i])
    return loss