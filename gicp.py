import numpy as np
from map import map
import copy
import gtsam

def ICP(clusterA: map,clusterB: map ,T0: np.ndarray((3,3)),threshold = 1e-5, dmax = 5):
    T = T0
    bi = clusterB.getLocations()
    converged = False
    while not converged:
        R = T[:2,:2]; t = T[:2,[2]]
        
        TclusterB = copy.deepcopy(clusterB)
        TclusterB.transform(R,t)
        Tbi = TclusterB.getLocations()
        
        mi = clusterA.getLocations(TclusterB.findClosestLandmark(clusterA))
        wi = np.linalg.norm(Tbi-mi,axis = 1) < dmax

        
        for w in wi:
            if w:
                graph.addfactor(x,bi,mi)
        
        # if np.norm(T - Tcandidate) < threshold:
        #     converged = True

        # Test = Tcandidate

    return T

# def generalizedICP(A,B,R0,t0):
#     '''
#     based on 'Generalized-ICP' by Aleksandr V. Segal, Dirk Haehnel, Sebastian Thrun

#     A - {Mai, Cai} such that ai ~ N(Mai,Cai) 
#     B - {Mbi, Cbi} such that bi ~ N(Mbi,Cbi) 
#     T0 - inital guess

#     define distance metric based on transform from source (A) to target (B)
#         di(T) = bi-Tai 
#     we wish to find T*which holds:
#         Mbi = T* @ Mai
#         di(T*) ~ (0,Cbi  + T* @ Cai @ T*')
    
#     We use maximum liklihood:
#         T = argmax{prodcut_i { p(di(T)) }} = argmax { sum_i {log(p(di(T)))} } 
#         T = argmin{sum_i {di(T)' @ (Cbi + T @ cai T')^-1 @ di(T)} }

    
#     the function returns our approximite T*
#     '''

#     return
