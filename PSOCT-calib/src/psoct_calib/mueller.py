"""Mueller matrices for rotations and ideal linear retarders.

Conventions
-----------
Stokes vectors are ordered (I, Q, U, V). All functions broadcast over
leading dimensions: an input of shape (...,) gives an output of shape
(..., 4, 4).

For a linear retarder with fast axis ``theta`` and retardance ``delta``,
the last column of its Mueller matrix is

    M[1, 3] = -sin(2 theta) sin(delta)
    M[2, 3] =  cos(2 theta) sin(delta)
    M[3, 3] =  cos(delta)

so the output Stokes vector for circularly polarized input (0, 0, 0, 1) is
(q, u, v) = (-M[1, 3], M[2, 3], M[3, 3]) = (sin2t sind, cos2t sind, cosd).
This is the same (Q, U, V) convention used in :mod:`psoct_calib.stokes`.
"""

import numpy as np


def rot_mueller(psi):
    """Mueller rotation matrix R(psi), psi in radians. Shape (..., 4, 4)."""
    psi = np.asarray(psi, dtype=float)
    c = np.cos(psi)
    s = np.sin(psi)
    R = np.zeros(psi.shape + (4, 4))
    R[..., 0, 0] = 1.0
    R[..., 1, 1] = c
    R[..., 1, 2] = s
    R[..., 2, 1] = -s
    R[..., 2, 2] = c
    R[..., 3, 3] = 1.0
    return R


def linear_retarder(theta, delta):
    """Mueller matrix of an ideal linear retarder.

    Parameters
    ----------
    theta : array_like
        Fast-axis orientation [rad].
    delta : array_like
        Retardance [rad]. Broadcast against ``theta``.

    Returns
    -------
    ndarray, shape (..., 4, 4)
    """
    theta, delta = np.broadcast_arrays(
        np.asarray(theta, dtype=float), np.asarray(delta, dtype=float)
    )
    c = np.cos(delta)
    s = np.sin(delta)
    M0 = np.zeros(theta.shape + (4, 4))
    M0[..., 0, 0] = 1.0
    M0[..., 1, 1] = 1.0
    M0[..., 2, 2] = c
    M0[..., 2, 3] = s
    M0[..., 3, 2] = -s
    M0[..., 3, 3] = c
    return rot_mueller(-2 * theta) @ M0 @ rot_mueller(2 * theta)


# Name used in the original notebook.
M_linear_retarder = linear_retarder


def retarder_params_from_stokes(Q, U, V):
    """Single-pass retarder (theta, delta) from a normalized round-trip Stokes vector.

    The measured (Q, U, V) is the result of a double pass, so the
    single-pass orientation and retardance are half of the angles
    encoded in the Stokes vector:

        theta = atan2(Q, U) / 2,   delta = arccos(V) / 2
    """
    Q = np.asarray(Q, dtype=float)
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    theta = np.arctan2(Q, U) / 2
    delta = np.arccos(np.clip(V, -1.0, 1.0)) / 2
    return theta, delta


def stokes_from_mueller(M):
    """(q, u, v) produced by M acting on circular input (0, 0, 0, 1)."""
    return -M[..., 1, 3], M[..., 2, 3], M[..., 3, 3]
