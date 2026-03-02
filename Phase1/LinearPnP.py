import numpy as np

def linear_pnp(K, X, x):
    K = np.asarray(K, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)

    N = X.shape[0]

    x_h = np.hstack([x, np.ones((N, 1))])
    x_norm = (np.linalg.inv(K) @ x_h.T).T  

    A = []

    for i in range(N):
        X_i = np.hstack([X[i], 1]) 
        u, v, w = x_norm[i]

        row1 = np.hstack([
            np.zeros(4),
            -w * X_i,
            v * X_i
        ])
        row2 = np.hstack([
            w * X_i,
            np.zeros(4),
            -u * X_i
        ])

        A.append(row1)
        A.append(row2)

    A = np.asarray(A)

    _, _, Vt = np.linalg.svd(A)
    P = Vt[-1].reshape(3, 4)


    R = P[:, :3]
    t = P[:, 3]

    U, _, Vt = np.linalg.svd(R)
    R = U @ Vt

    if np.linalg.det(R) < 0:
        R = -R
        t = -t
    C = -R.T @ t

    return C, R