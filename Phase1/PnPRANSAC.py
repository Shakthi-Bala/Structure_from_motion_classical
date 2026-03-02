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

    N = X_world.shape[0]

    # Convert 2D to homogeneous
    x_h = np.hstack((x_image, np.ones((N, 1))))

    # Normalize using intrinsics
    x_norm = (np.linalg.inv(K) @ x_h.T).T

    A = []

    for i in range(N):
        X, Y, Z = X_world[i]
        u, v, _ = x_norm[i]

        A.append([X, Y, Z, 1, 0, 0, 0, 0, -u*X, -u*Y, -u*Z, -u])
        A.append([0, 0, 0, 0, X, Y, Z, 1, -v*X, -v*Y, -v*Z, -v])

    A = np.asarray(A)

    # Solve Ap = 0
    _, _, Vt = np.linalg.svd(A)
    P = Vt[-1].reshape(3, 4)

    R = P[:, :3]
    t = P[:, 3]

    # Enforce R ∈ SO(3)
    U, _, Vt = np.linalg.svd(R)
    R = U @ Vt

    if np.linalg.det(R) < 0:
        R = -R
        t = -t

    # Camera center
    C = -R.T @ t

    return R, C


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

        # FIX: Remove the '_, ' because linearpnp only returns R and C
        R, C = linearpnp(X_sample, x_sample, K)

        x_projected = project_points(X_world, R, C, K)
        x_projected = x_projected.reshape(-1, 2)

        errors = np.linalg.norm(x_image - x_projected, axis=1)
        inliers = np.where(errors < threshold)[0]

        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_R = R
            best_C = C

    return best_R, best_C, best_inliers