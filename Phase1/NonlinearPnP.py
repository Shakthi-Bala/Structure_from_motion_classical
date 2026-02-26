import numpy as np 
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R

from PnPRANSAC import linearpnp,project_points

def reprojection_error(params, X_world, x_image, K):
    """
    Compute the reprojection error for the given camera parameters.
    """

    # Extract rotation and translation from parameters
    r_vec = params[:3]
    t_vec = params[3:]
    
    # Normalize quaternion
    q = r_vec / np.linalg.norm(r_vec)

    # Convert quaternion to rotation matrix
    R_mat = R.from_quat(q).as_matrix()

    # Project 3D points to 2D
    x_projected = project_points(X_world, R_mat, t_vec, K)

    # Compute reprojection error
    error = (x_projected - x_image).flatten()
    return error


def nonlinear_pnp(X_world, x_image, K, R_initial, C_initial ):
    """
    Refine the camera pose to minimize reprojection error.
    """

    X_world = np.asarray(X_world, dtype=np.float64)
    x_image = np.asarray(x_image, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    
    # convert rotation matrix to quaternion for optimization
    q_initial = R.from_matrix(R_initial).as_quat()
    params_initial = np.hstack((q_initial, C_initial))

    # Optimize using least squares
    result = least_squares(reprojection_error, params_initial, args=(X_world, x_image, K), method='lm')

    # Extract optimized parameters
    q_opt = result.x[:4]
    C_opt = result.x[4:]
    
    # Convert optimized quaternion back to rotation matrix
    q_opt = q_opt / np.linalg.norm(q_opt)  # Normalize quaternion
    R_opt = R.from_quat(q_opt).as_matrix()



    return q_opt, C_opt

