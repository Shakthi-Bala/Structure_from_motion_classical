import os
import glob
import cv2
import numpy as np

from GetInliersRANSAC import sift_match, get_inliers_ransac, estimate_fundamental_matrix
from EssentialMatrixFromFundamentalMatrix import essential_matrix_from_fundamental
from ExtractCameraPose import extract_camera_poses
from LinearTriangulation import triangulate_3d_points
from scipy.optimize import least_squares

def make_3d_projection(K, C, R):
    C = C.reshape(3,1)
    I = np.eye(3, dtype = np.float64)
    P = K.dot(R).dot(np.hstack([I, -C]))

def project(P, X):
    X = np.asarray(X, dtype = np.float32).reshape(3,)
    Xh = np.array([X[0], X[1], X[2], 1.0], dtype = np.float32)
    z = X[2]
    v = X[1] / z
    u = X[0] / z

    return np.array([u, v], dtype = np.float32)

def residual_from_point(X, P1, P2, x1, x2):

    x1_hat = project(P1, X)
    x2_hat = project(P2, X)

    r = np.array([
        x1[0] - x1_hat[0],  
        x1[1] - x1_hat[1],
        x2[0] - x2_hat[0],
        x2[1] - x2_hat[1]
    ], dtype = np.float32)

    return r

def nonlinear_triangulation(K, C1, R1, C2, R2, x1, x2, X_init,
                            max_nfev=50, robust_loss="huber", f_scale=1.0): 
    x1 = np.asarray(x1, dtype=np.float64)
    x2 = np.asarray(x2, dtype=np.float64)
    X_init = np.asarray(X_init, dtype=np.float64)

    P1 = make_3d_projection(K, C1, R1)
    P2 = make_3d_projection(K, C2, R2)

    X_refined = np.zeros_like(X_init, dtype=np.float64)

    for i in range(X_init.shape[0]):
        Xi0 = X_init[i, :]
        obs1 = x1[i, :]
        obs2 = x2[i, :]

    def fun(Xvec):
        return residual_from_point(Xvec, P1, P2, obs1, obs2)

    res = least_squares(
        fun,
        Xi0,
        method="trf",
        loss=robust_loss,  
        f_scale=f_scale,
        max_nfev=max_nfev
    )
    X_refined[i, :] = res.x

    return X_refined    

def reprojection_rmse(K, C1, R1, C2, R2, x1, x2, X):

    x1 = np.asarray(x1, dtype=np.float64)
    x2 = np.asarray(x2, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)

    P1 = make_3d_projection(K, C1, R1)
    P2 = make_3d_projection(K, C2, R2)

    err = []
    for i in range(X.shape[0]):
        p1 = project(P1, X[i, :])
        p2 = project(P2, X[i, :])
        err.append(np.sum((x1[i, :] - p1) ** 2))
        err.append(np.sum((x2[i, :] - p2) ** 2))

    err = np.array(err, dtype=np.float64)
    return np.sqrt(np.mean(err))