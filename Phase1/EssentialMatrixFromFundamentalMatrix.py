import os
import glob
import cv2
import numpy as np

from GetInliersRANSAC import sift_match, get_inliers_ransac, estimate_fundamental_matrix


K = np.array([[531.122155322710, 0, 407.192550839899],
             [0, 531.541737503901, 313.308715048366],
             [0, 0, 1]], dtype = np.float32)

def essential_matrix_from_fundamental(F, K):
    F = np.asarray(F, dtype = np.float32)
    K = np.asarray(K, dtype = np.float32)

    E = K.T.dot(F).dot(K)

    U, S, Vt = np.linalg.svd(E)

    if np.linalg.det(U.dot(Vt)) < 0:
        Vt = -Vt

    E_corrected = U.dot(np.diag([1.0, 1.0, 0.0])).dot(Vt)

    return E_corrected
     