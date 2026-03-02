import numpy as np

def build_visibility_matrix(n_cameras, n_points, camera_point_indices):

    V = np.zeros((n_cameras, n_points), dtype=int)

    for cam_idx in range(n_cameras):
        visible_pts = camera_point_indices[cam_idx]
        V[cam_idx, visible_pts] = 1

    return V