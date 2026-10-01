"""Conversion from Jones measurements to retardance, orientation and Stokes."""

import numpy as np
from scipy.ndimage import gaussian_filter

GAUSSIAN_TRUNCATE = 4.0  # scipy.ndimage default


def jones_to_retardance_orientation(J1, J2, offset=0.0):
    """Retardance and orientation maps from the two Jones channels.

    Parameters
    ----------
    J1, J2 : complex ndarray
        Jones channels, same shape.
    offset : float
        Phase offset [rad]; added as ``2 * offset`` to the phase difference.

    Returns
    -------
    R3D, O3D : ndarray
        Retardance and orientation in degrees. O3D lies in [-90, 90).
    """
    R3D = np.degrees(np.arctan2(np.abs(J1), np.abs(J2)))
    phi = np.angle(J1) - np.angle(J2) + 2 * offset
    phi = np.mod(phi + np.pi, 2 * np.pi) - np.pi  # wrap to [-pi, pi)
    O3D = np.degrees(phi) / 2
    return R3D, O3D


def normalize_stokes(Q, U, V, eps=1e-12):
    """Scale (Q, U, V) to unit length, guarding against zero norm."""
    n = np.sqrt(Q**2 + U**2 + V**2)
    n = np.maximum(n, eps)
    return Q / n, U / n, V / n


def stokes_from_orientation_retardance(O3D, R3D, sigma=1.0):
    """Normalized round-trip Stokes vector from orientation and retardance maps.

    Q = sin(2 O) sin(2 R), U = cos(2 O) sin(2 R), V = cos(2 R),
    each Gaussian-filtered with ``sigma`` (in pixels, all axes) and then
    normalized per voxel.
    """
    O = np.radians(O3D)
    R = np.radians(R3D)
    s2R = np.sin(2 * R)
    Q = np.sin(2 * O) * s2R
    U = np.cos(2 * O) * s2R
    V = np.cos(2 * R)
    if sigma and sigma > 0:
        Q = gaussian_filter(Q, sigma)
        U = gaussian_filter(U, sigma)
        V = gaussian_filter(V, sigma)
    return normalize_stokes(Q, U, V)


def filter_radius(sigma, truncate=GAUSSIAN_TRUNCATE):
    """Kernel radius in pixels used by scipy.ndimage.gaussian_filter."""
    if not sigma or sigma <= 0:
        return 0
    return int(truncate * float(sigma) + 0.5)
