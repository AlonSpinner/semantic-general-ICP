import numpy as np

def kde(m0,S0,m1,S1):
    #based on https://arxiv.org/pdf/1811.04751v1.pdf
    iS0 = np.linalg.inv(S0)
    iS1 = np.linalg.inv(S1)
    i_iS0piS1 = np.linalg.inv(iS0+iS1)
    dS0 = np.linalg.det(S0)
    dS1 = np.linalg.det(S1)
    d_iS0piS1 = np.linalg.det(iS0+iS1)

    m = m0 - m1 #new m0 


    num = np.exp(-0.5 (m.T @ iS0 @ i_iS0piS1 @ iS1 @ m))
    denum = np.sqrt(dS0 * dS1 * d_iS0piS1) #ignore 1/(2*pi)^D

    return num/denum

def kl_mvn(m0, S0, m1, S1):
    """
    copied from https://stackoverflow.com/questions/44549369/kullback-leibler-divergence-from-gaussian-pm-pv-to-gaussian-qm-qv

    Kullback-Liebler divergence from Gaussian pm,pv to Gaussian qm,qv.
    Also computes KL divergence from a single Gaussian pm,pv to a set
    of Gaussians qm,qv.
    

    From wikipedia
    KL( (m0, S0) || (m1, S1))
         = .5 * ( tr(S1^{-1} S0) + log |S1|/|S0| + 
                  (m1 - m0)^T S1^{-1} (m1 - m0) - N )
    """
    # store inv diag covariance of S1 and diff between means
    N = m0.shape[0]
    iS1 = np.linalg.inv(S1)
    diff = m1 - m0

    # kl is made of three terms
    tr_term   = np.trace(iS1 @ S0)
    det_term  = np.log(np.linalg.det(S1)/np.linalg.det(S0)) #np.sum(np.log(S1)) - np.sum(np.log(S0))
    quad_term = diff.T @ np.linalg.inv(S1) @ diff #np.sum( (diff*diff) * iS1, axis=1)
    #print(tr_term,det_term,quad_term)
    return .5 * (tr_term + det_term + quad_term - N) 
