import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as Rot

from PnPRANSAC import project_points


def _unpack(params, n_points):
    q = params[:4]
    C2 = params[4:7]
    X = params[7:].reshape(n_points, 3)

    q = q / (np.linalg.norm(q) + 1e-12)
    R2 = Rot.from_quat(q).as_matrix()
    return R2, C2, X


def _ba_residual(params, K, C1, R1, x1, x2):
    n_points = x1.shape[0]
    R2, C2, X = _unpack(params, n_points)

    x1_proj = project_points(X, R1, C1, K)
    x2_proj = project_points(X, R2, C2, K)

    r1 = (x1_proj - x1).reshape(-1)
    r2 = (x2_proj - x2).reshape(-1)
    return np.hstack((r1, r2))


def bundle_adjust_two_view(K, C1, R1, C2_initial, R2_initial, x1, x2, X_initial):
    K = np.asarray(K, dtype=np.float64)
    C1 = np.asarray(C1, dtype=np.float64).reshape(3,)
    R1 = np.asarray(R1, dtype=np.float64).reshape(3, 3)
    C2_initial = np.asarray(C2_initial, dtype=np.float64).reshape(3,)
    R2_initial = np.asarray(R2_initial, dtype=np.float64).reshape(3, 3)
    x1 = np.asarray(x1, dtype=np.float64)
    x2 = np.asarray(x2, dtype=np.float64)
    X_initial = np.asarray(X_initial, dtype=np.float64)

    q0 = Rot.from_matrix(R2_initial).as_quat()
    p0 = np.hstack((q0, C2_initial, X_initial.reshape(-1)))

    res = least_squares(
        _ba_residual,
        p0,
        method="trf",
        args=(K, C1, R1, x1, x2),
    )

    R2_opt, C2_opt, X_opt = _unpack(res.x, X_initial.shape[0])
    return R2_opt, C2_opt, X_opt
