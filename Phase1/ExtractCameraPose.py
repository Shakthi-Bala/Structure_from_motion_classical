import os
import glob
import cv2
import numpy as np

from GetInliersRANSAC import sift_match, get_inliers_ransac, estimate_fundamental_matrix
from EssentialMatrixFromFundamentalMatrix import essential_matrix_from_fundamental

def extract_camera_poses(E):
    E = np.asarray(E, dtype = np.float32)
    #SVD for E
    U, S, Vt = np.linalg.svd(E)

    if np.linalg.det(U) < 0:
        U[:, -1] *= -1

    if np.linalg.det(Vt) < 0:
        Vt[-1, :] *= -1

    W = np.array([[0, 1, 0],
                  [-1, 0, 0],
                  [0, 0, 1]], dtype=np.float64)
    
    t = U[:, 2]

    R1 = U @ W @ Vt
    R2 = U @ W.T @ Vt

    # Ensure rotations are valid
    if np.linalg.det(R1) < 0:
        R1 = -R1
    if np.linalg.det(R2) < 0:
        R2 = -R2

    # Convert translation to camera center
    C1 = -R1.T @ t
    C2 = -R1.T @ (-t)
    C3 = -R2.T @ t
    C4 = -R2.T @ (-t)

    poses = [
        (C1, R1),
        (C2, R1),
        (C3, R2),
        (C4, R2),
    ]

    fixed_poses = []
    for Ci, Ri in poses:
        if np.linalg.det(Ri) < 0:
            Ri = -Ri
            Ci = -Ci
        fixed_poses.append((Ci, Ri))
    
    return fixed_poses
