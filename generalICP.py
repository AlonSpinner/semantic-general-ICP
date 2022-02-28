from scipy.optimize import fmin_cg
from clusters import cluster2
import numpy as np
from utils.transforms2D import R2 
 
def generalICP(source,target):
 
    
    while True:
        cpt = cpt+1
        R = rot_mat(x[3:])
        M = np.array([np.linalg.inv(cov_ref[i] + R @ cov_data[i] @ R.T) for i in range(n)])

        f = lambda x: loss(x,data.points[indexes_d],ref.points[indexes_r],M)
        df = lambda x: grad_loss(x,data.points[indexes_d],ref.points[indexes_r],M)

        out = fmin_cg(f = f, x0 = x, fprime = df, disp = False, full_output = True)

        x = out[0]
        f_min = out[1]
        if verbose:
            print("\t\t EM style iteration {} with loss {}".format(cpt,f_min))

        if last_min - f_min < tol:
            if verbose:
                print("\t\t\t Stopped EM because not enough improvement or not at all")
            break
        elif cpt >= n_iter_max:
            if verbose:
                print("\t\t\t Stopped EM because maximum number of iterations reached")
            break
        else:
            last_min = f_min


def loss(x,a,b,M):
    """
    loss for parameter x

    params:
        x : length 3 vector of transformation parameters
            (t_x,t_y,t_theta)
        a : data to align n*2, part of source
        b : ref point cloud n*2 a[i] is the nearest neibhor of Rb[i]+t
        M : central matrix for each data point n*3*3 (cf loss equation)

    returns:
        Value of the loss function
    """
    residual = b - R2(x[3])@ a -x[None,:2] # shape n*d
    tmp = np.sum(M * residual[:,None,:], axis = 2) # shape n*d
    return np.sum(residual * tmp)

def grad_loss(x,a,b,M):
    """
    Gradient of the loss loss for parameter x

    params:
        x : length 6 vector of transformation parameters
            (t_x,t_y,t_z, theta_x, theta_y, theta_z)
        a : data to align n*3
        b : ref point cloud n*3 a[i] is the nearest neibhor of Rb[i]+t
        M : central matrix for each data point n*3*3 (cf loss equation)

    returns:
        Value of the gradient of the loss function
    """
    t = x[:3]
    R = rot_mat(x[3:])
    g = np.zeros(6)
    residual = b - a @ R2(x[3]) -x[None,:2] # shape n*d
    tmp = np.sum(M * residual[:,None,:], axis = 2) # shape n*d

    g[:3] = - 2*np.sum(tmp, axis = 0)

    grad_R = - 2* (tmp.T @ a) # shape d*d
    grad_R_euler = grad_rot_mat(x[3:]) # shape 3*d*d
    g[3:] = np.sum(grad_R[None,:,:] * grad_R_euler, axis = (1,2)) # chain rule
    return g