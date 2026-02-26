import os
import glob
import cv2
import numpy as np

from GetInliersRANSAC import sift_match, get_inliers_ransac, estimate_fundamental_matrix
from EssentialMatrixFromFundamentalMatrix import essential_matrix_from_fundamental
from ExtractCameraPose import extract_camera_poses

def make_3d_projection(K, C, R):
    C = C.reshape(3,1)
    I = np.eye(3, dtype = np.float64)
    P = K.dot(R).dot(np.hstack([I, -C]))


def triangulate_3d_points(C1, R1, C2, R2, x1, x2):
    K = np.asarray(K, dtype=np.float64)
    C1 = np.asarray(C1, dtype=np.float64).reshape(3,)
    C2 = np.asarray(C2, dtype=np.float64).reshape(3,)
    R1 = np.asarray(R1, dtype=np.float64).reshape(3, 3)
    R2 = np.asarray(R2, dtype=np.float64).reshape(3, 3)
    x1 = np.asarray(x1, dtype=np.float64)
    x2 = np.asarray(x2, dtype=np.float64)

    P1 = make_3d_projection(K, C1, R1) 
    P2 = make_3d_projection(K, C2, R2) 

    X_out = np.zeros((x1.shape[0], 3), dtype = np.float64)

    for i in range(x1.shape[0]):
        u1, v1 = x1[i, 0], x1[i, 1]
        u2, v2 = x2[i, 0], x2[i, 1]

        A = np.zeros((4, 4), dtype=np.float64)
        A[0, :] = u1 * P1[2, :] - P1[0, :]
        A[1, :] = v1 * P1[2, :] - P1[1, :]
        A[2, :] = u2 * P2[2, :] - P2[0, :]
        A[3, :] = v2 * P2[2, :] - P2[1, :]

        _, _, Vt = np.linalg.svd(A)
        X_h = Vt[-1, :] 

        X_h = X_h / (X_h[3] + 1e-12)
        X_out[i, :] = X_h[:3]

    return X_out
    