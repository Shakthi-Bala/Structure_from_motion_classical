import os
import glob
import cv2
import numpy as np

from GetInliersRANSAC import sift_match, get_inliers_ransac, estimate_fundamental_matrix
from EssentialMatrixFromFundamentalMatrix import essential_matrix_from_fundamental
from ExtractCameraPose import extract_camera_poses

from LinearTriangulation import triangulate_3d_points

def chirality_count(C1, R1, C2, R2, X):
    C1 = C1.reshape(3,)
    C2 = C2.reshape(3,)
    r3_1 = R1[2, :].reshape(1, 3) 
    r3_2 = R2[2, :].reshape(1, 3)

    cond1 = (X - C1).dot(r3_1.T).reshape(-1) > 0
    cond2 = (X - C2).dot(r3_2.T).reshape(-1) > 0
    return int(np.sum(cond1 & cond2))

def disambiguate_cam_poses(K, poses, x1, x2):
    K = np.asarray(K, dtype=np.float64)

    C1 = np.zeros(3, dtype=np.float64)
    R1 = np.eye(3, dtype=np.float64)

    best_idx = -1
    best_count = -1
    best_C2, best_R2, best_X = None, None, None

    for i, (C2, R2) in enumerate(poses):
        C2 = np.asarray(C2, dtype = np.float32).reshape(3,)
        R2 = np.asarray(R2, dtype = np.float32).reshape(3,3)

        X = triangulate_3d_points(K, C1, R1, C2, R2, x1, x2)

        count = chirality_count(C1,R1,C2,R2,X)

        if count > best_count:
            best_count = count
            best_idx = i
            best_C2, best_R2, best_X = C2, R2, X
    
    return best_C2, best_R2, best_X, best_idx