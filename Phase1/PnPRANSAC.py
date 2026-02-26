import os
import glob
import cv2
import numpy as np

from GetInliersRANSAC import sift_match, get_inliers_ransac, estimate_fundamental_matrix
from EssentialMatrixFromFundamentalMatrix import essential_matrix_from_fundamental
from ExtractCameraPose import extract_camera_poses

from LinearTriangulation import triangulate_3d_points


def linearpnp(X_sample, x_sample, K):
    K = np.asarray(K, dtype=np.float64)
    X_world = np.asarray(X_sample, dtype=np.float64)
    x_image = np.asarray(x_sample, dtype=np.float64)

    
    N=X_world.shape[0]

    # Convert 2d image points to homogeneous coordinates
    x_h= np.hstack((x_image, np.ones((N, 1), dtype=np.float64)))

    # Normalize the 2D image points using the intrinsic matrix
    x_normalized = np.linalg.inv(K).dot(x_h.T).T

    # Construct the A matrix for DLT

    A = np.zeros((2 * N, 12), dtype=np.float64)
    for i in range(N):
        X, Y, Z = X_world[i]
        u, v, _ = x_normalized[i]

        
        # Two equations per point correspondence
        A[2 * i] = np.hstack((X, Y, Z, 1, 0, 0, 0, 0, -u*X, -u*Y, -u*Z, -u))
        A[2 * i + 1] = np.hstack((0, 0, 0, 0, X, Y, Z, 1, -v*X, -v*Y, -v*Z, -v))

    # Solve for the projection matrix P using SVD
    _, _, Vt = np.linalg.svd(A)
    P = Vt[-1].reshape(3, 4)

    # Extract R and C from P
    R = P[:, :3]
    t = P[:, 3]

    # Ensure R is a valid rotation matrix using SVD
    U, S_R, Vt = np.linalg.svd(R)
    R = U.dot(Vt)

    # Correct scale of t 
    scale = np.linalg.norm(R) / np.mean(S_R)
    t = t * scale

    # Valid R , Determinant should be 1 
    if np.linalg.det(R) < 0:
        R = -R
        t = -t
    
    # Compute camera center C 
    C = -R.T.dot(t)

    return P, R, C


def project_points(X_world, R, C, K):
    """
    Project 3D world points to 2D image points using the estimated camera pose.
    """
    X_world = np.asarray(X_world, dtype=np.float64)
    R = np.asarray(R, dtype=np.float64)
    C = np.asarray(C, dtype=np.float64).reshape(3,1)
    K = np.asarray(K, dtype=np.float64)

    # Translate to camera coordinate system
    X_cam = R.dot(X_world.T - C)

    # Project to image plane

    x_proj = K.dot(X_cam)

    # Divide by Z to get pixel coordinates
    u = x_proj[0, :] / (x_proj[2, :] + 1e-12)
    v = x_proj[1, :] / (x_proj[2, :] + 1e-12)

    return np.vstack((u, v)).T


def pnp_ransac(K, X_world, x_image, num_iterations=1000, threshold=5.0, confidence=0.99):
    best_inliers = []
    best_R = None
    best_C = None

    for _ in range(num_iterations):
        indices = np.random.choice(X_world.shape[0], 6, replace=False)
        X_sample = X_world[indices]
        x_sample = x_image[indices]

        
        _, R, C = linearpnp(X_sample, x_sample, K )

        x_projected = project_points(X_world, R, C, K)
        x_projected = x_projected.reshape(-1, 2)

        errors = np.linalg.norm(x_image - x_projected, axis=1)
        inliers = np.where(errors < threshold)[0]

        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_R = R
            best_C = C

    return best_R, best_C, best_inliers