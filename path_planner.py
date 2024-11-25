import numpy as np


def setup_equality_constraints(
    N: int,
    delta_s: float,
    n_states: int,
    n_controls: int,
    yaw_0: np.array,
    kappa_0: np.array,
):
    A = np.zeros((N * n_states, N * n_states + (N - 1) * n_controls))
    b = np.zeros((N * n_states,))
    for i in range(0, N - 1):
        # old state
        A_lin = np.zeros((3, 3))
        A_lin[0, 2] = -delta_s * np.sin(yaw_0[i])
        A_lin[1, 2] = delta_s * np.cos(yaw_0[i])
        A[i * n_states : (i + 1) * n_states, i * n_states : (i + 1) * n_states] = -(
            np.eye(3) + A_lin
        )
        # new state
        A[
            i * n_states : (i + 1) * n_states, (i + 1) * n_states : (i + 2) * n_states
        ] = np.eye(3)
        # control
        A[i * n_states : (i + 1) * n_states, N * n_states + i * n_controls] = -np.array(
            [[0, 0, 1]]
        )
        # right-hand side
        b[i * n_states : (i + 1) * n_states] = (
            delta_s * np.array([np.cos(yaw_0[i]), np.sin(yaw_0[i]), kappa_0[i]])
            - np.array(
                [
                    -delta_s * np.sin(yaw_0[i]) * yaw_0[i],
                    delta_s * np.cos(yaw_0[i]) * yaw_0[i],
                    0,
                ]
            )
            - np.array([0, 0, kappa_0[i]])
        )
    # Initial condition
    A[(N - 1) * n_states : N * n_states, 0:n_states] = np.eye(3)
    b[(N - 1) * n_states : N * n_states] = np.array(
        [0, 0, 0]
    )  # for the sake of completeness
    return A, b


def setup_inequality_constraints(
    N: int, n_states: int, n_controls: int, ay_max: float, vx: float
):
    G = np.zeros((2 * (N - 1), N * n_states + (N - 1) * n_controls))
    h = ay_max * np.ones((2 * (N - 1),))
    for i in range(N - 1):
        G[i * (n_controls + 1), N * n_states + i] = vx**2
        G[i * (n_controls + 1) + 1, N * n_states + i] = -(vx**2)
    return G, h


def setup_cost(
    N: int,
    n_states: int,
    n_controls: int,
    Q: np.ndarray,
    R: np.ndarray,
    x_ref: np.array,
    y_ref: np.array,
):
    P = np.zeros(
        (N * n_states + (N - 1) * n_controls, N * n_states + (N - 1) * n_controls)
    )
    q = np.zeros((N * n_states + (N - 1) * n_controls,))
    # Trajectory tracking cost
    for i in range(0, N - 1):
        P[i * n_states : (i + 1) * n_states, i * n_states : (i + 1) * n_states] = 2 * Q
        q[i * n_states : (i + 1) * n_states] = -2 * np.array(
            [Q[0, 0] * x_ref[i], Q[1, 1] * y_ref[i], Q[2, 2] * 0.0]
        )
    # Terminal cost
    P[(N - 1) * n_states : N * n_states, (N - 1) * n_states : N * n_states] = 2 * Q
    q[(N - 1) * n_states : N * n_states] = -2 * np.array(
        [Q[0, 0] * x_ref[N - 1], Q[1, 1] * y_ref[N - 1], Q[2, 2] * 0.0]
    )
    # Control cost
    for i in range(0, N - 1):
        P[
            N * n_states + i * n_controls : N * n_states + (i + 1) * n_controls,
            N * n_states + i * n_controls : N * n_states + (i + 1) * n_controls,
        ] = R
    return P, q


def calculate_inv_backprop_matrix(
    Q: np.ndarray,
    G: np.ndarray,
    A: np.ndarray,
    h: np.array,
    z_opt: np.array,
    lambda_opt: np.array,
) -> np.ndarray:
    n = Q.shape[0]
    neq = A.shape[0]
    nieq = G.shape[0]
    matrix = np.zeros((n + neq + nieq, n + neq + nieq))
    matrix[:n, :n] = Q
    matrix[:n, n : n + nieq] = np.transpose(G) @ np.diag(lambda_opt)
    matrix[:n, n + nieq : n + nieq + neq] = np.transpose(A)
    matrix[n : n + nieq, :n] = G
    matrix[n : n + nieq, n : n + nieq] = np.diag(np.dot(G, z_opt) - h)
    matrix[n + nieq : n + nieq + neq, :n] = A

    return -np.linalg.inv(matrix)
