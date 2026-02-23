#!/usr/bin/env python

import numpy as np 
import cv2
import random

def normalize_points(pts):
    pts = np.asarray(pts, dtype = np.float64)
    mean = np.mean(pts, axis = 0)
    pts_centered = pts - mean

    dists = np.sqrt(np.sum(pts_centered**2 , axis =1))
    mean_dist = np.mean(dists)

    s = np.sqrt(2.0) / mean_dist
    T = np.array([[s, 0, -s*mean[0]],
                  [0, s, -s*mean[1]],
                  [0, 0, 1]], dtype = np.float64)
    
    pts_h = np.hstack([pts, np.ones((pts.shape[0], 1), dtype = np.float64)])
    pts_n_h = (T.dot(pts_h.T)).T
    pts_n = pts_n_h[:,:2] / (pts_n_h[:, 2:3])
    return pts_n, T

def estimate_fundamental_matrix(pts1, pts2):
    pts1 = np.asarray(pts1, dtype = np.float64)
    pts2 = np.asarray(pts2, dtype = np.float64)
    if pts1.shape[0] < 8:
        raise ValueError("Need atleast 8 correspondences")

    p1n, T1 = normalize_points(pts1)
    p2n, T2 = normalize_points(pts2)

    x1, y1 = p1n[:, 0], p1n[:, 1]
    x2, y2 = p2n[:, 0], p2n[:, 1]

    A = np.column_stack([
        x1*x1, y1*x2, x2,
        x1*y2, y1*y2, y2,
        x1, y1, np.ones_like(x1)
    ])

    _, _ , Vt = np.linalg.svd(A)
    f = Vt[-1, :]
    F_hat = f.reshape(3,3)

    # Rank2 constraint
    U, S, Vt2 = np.linalg.svd(F_hat)
    S[-1] = 0.0
    F_rank2 = U.dot(np.diag(S)).dot(Vt2)

    # Denormalize 
    F = T2.T.dot(F_rank2).dot(T1)

    return F

def sampson_distance(F, pts1, pts2):
    pts1 = np.asarray(pts1, dtype=np.float64)
    pts2 = np.asarray(pts2, dtype=np.float64)

    x1 = np.hstack([pts1, np.ones((pts1.shape[0], 1))])
    x2 = np.hstack([pts2, np.ones((pts2.shape[0], 1))])

    Fx1 = (F.dot(x1.T)).T          # (N,3)
    Ftx2 = (F.T.dot(x2.T)).T       # (N,3)
    x2tFx1 = np.sum(x2 * Fx1, axis=1)  # (N,)

    denom = Fx1[:, 0]**2 + Fx1[:, 1]**2 + Ftx2[:, 0]**2 + Ftx2[:, 1]**2
    denom = denom + 1e-12

    d = (x2tFx1**2) / denom
    return d

def create_sift():
    if hasattr(cv2, "SIFT_create"):
        return cv2.SIFT_create()
    if hasattr(cv2, "xfeatures2d") and hasattr(cv2.xfeatures2d, "SIFT_create"):
        return cv2.xfeatures2d.SIFT_create()
    raise RuntimeError("SIFT is not available in your OpenCV build.")


def sift_match(img1_gray, img2_gray, ratio=0.75, max_matches=5000):
    sift = create_sift()
    kp1, des1 = sift.detectAndCompute(img1_gray, None)
    kp2, des2 = sift.detectAndCompute(img2_gray, None)

    if des1 is None or des2 is None or len(kp1) == 0 or len(kp2) == 0:
        return np.zeros((0, 2)), np.zeros((0, 2)), [], kp1, kp2

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    knn = bf.knnMatch(des1, des2, k=2)

    good = []
    for m, n in knn:
        if m.distance < ratio * n.distance:
            good.append(m)

    # Optional: keep strongest matches only
    good = sorted(good, key=lambda m: m.distance)
    if max_matches is not None and len(good) > max_matches:
        good = good[:max_matches]

    pts1 = np.array([kp1[m.queryIdx].pt for m in good], dtype=np.float64)
    pts2 = np.array([kp2[m.trainIdx].pt for m in good], dtype=np.float64)
    return pts1, pts2, good, kp1, kp2

def get_inliers_ransac(pts1, pts2, num_iters=2000, threshold=1e-3, seed=42):
    pts1 = np.asarray(pts1, dtype=np.float64)
    pts2 = np.asarray(pts2, dtype=np.float64)

    N = pts1.shape[0]
    if N < 8:
        raise ValueError("Need at least 8 matches to run RANSAC for F.")

    random.seed(seed)
    np.random.seed(seed)

    best_inliers = None
    best_F = None
    best_count = -1

    all_idx = np.arange(N)

    for _ in range(num_iters):
        sample_idx = np.random.choice(all_idx, size=8, replace=False)
        p1_s = pts1[sample_idx]
        p2_s = pts2[sample_idx]

        try:
            F = estimate_fundamental_matrix(p1_s, p2_s)
        except Exception:
            continue

        d = sampson_distance(F, pts1, pts2)
        inliers = d < threshold
        count = int(np.sum(inliers))

        if count > best_count:
            best_count = count
            best_inliers = inliers
            best_F = F

    if best_inliers is None:
        raise RuntimeError("RANSAC failed to find a valid fundamental matrix.")

    # Refit F on all inliers (important!)
    inlier_pts1 = pts1[best_inliers]
    inlier_pts2 = pts2[best_inliers]
    if inlier_pts1.shape[0] >= 8:
        best_F = estimate_fundamental_matrix(inlier_pts1, inlier_pts2)

    return best_F, best_inliers, inlier_pts1, inlier_pts2
