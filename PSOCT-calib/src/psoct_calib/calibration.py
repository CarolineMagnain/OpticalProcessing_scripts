"""Reference-profile extraction, calibration fit and correction.

Model
-----
For each column j, the round-trip Stokes profile is converted to an
equivalent single-pass linear retarder M_meas(j). The system is modelled
as a linear retarder M_LR(j) traversed on the way in and
on the way out:

    M_meas(j) = M_LR(j) @ M_ref @ M_LR(j)

where M_ref is the retarder of a reference column. (theta, delta) of
M_LR(j) are found by nonlinear least squares on the 16 matrix elements.
The correction of a measurement is

    M_true = inv(M_LR) @ M_meas @ inv(M_LR),   Mc = M_true @ M_true,
    (q, u, v) = (-Mc[1, 3], Mc[2, 3], Mc[3, 3]).
"""

from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.optimize import least_squares

from .mueller import linear_retarder, retarder_params_from_stokes, stokes_from_mueller
from .stokes import (
    filter_radius,
    jones_to_retardance_orientation,
    normalize_stokes,
    stokes_from_orientation_retardance,
)


# --------------------------------------------------------------------------
# reference profile
# --------------------------------------------------------------------------
def reference_profile(Q, U, V):
    """Average (I, J, K) Stokes volumes over I and K and renormalize -> (J,) each."""
    Qa = Q.mean(axis=(0, 2))
    Ua = U.mean(axis=(0, 2))
    Va = V.mean(axis=(0, 2))
    return normalize_stokes(Qa, Ua, Va)


def extract_profile(jones_file, i_range, k_range, sigma_stokes=1.0, offset=0.0):
    """Reference profile from a region of interest, reading only that region.

    The slab is padded by the Gaussian kernel radius (clipped at the volume
    edges) before filtering and cropped afterwards, so the result equals
    filtering the full volume first and then cropping.

    Parameters
    ----------
    jones_file : psoct_calib.io.JonesFile
    i_range, k_range : (start, stop)
        Half-open, 0-based ranges (Python slicing; MATLAB 4001:5000 is (4000, 5000)).
    """
    I, _, K = jones_file.shape
    i0, i1 = map(int, i_range)
    k0, k1 = map(int, k_range)
    if not (0 <= i0 < i1 <= I):
        raise IndexError(f"i_range {i_range} outside [0, {I}]")
    if not (0 <= k0 < k1 <= K):
        raise IndexError(f"k_range {k_range} outside [0, {K}]")

    r = filter_radius(sigma_stokes)
    i0p, i1p = max(i0 - r, 0), min(i1 + r, I)
    k0p, k1p = max(k0 - r, 0), min(k1 + r, K)

    J1, J2 = jones_file.read(i0p, i1p, k0p, k1p)
    R3D, O3D = jones_to_retardance_orientation(J1, J2, offset)
    del J1, J2
    Q, U, V = stokes_from_orientation_retardance(O3D, R3D, sigma_stokes)
    del R3D, O3D

    crop = (slice(i0 - i0p, i1 - i0p), slice(None), slice(k0 - k0p, k1 - k0p))
    return reference_profile(Q[crop], U[crop], V[crop])


# --------------------------------------------------------------------------
# fit
# --------------------------------------------------------------------------
@dataclass
class CalibrationResult:
    calibration_M: np.ndarray        # (J, 4, 4)
    theta_fit: np.ndarray            # (J,)
    delta_fit: np.ndarray            # (J,)
    cost: np.ndarray                 # (J,) 0.5 * sum of squared residuals
    ref_index: int                   # index of the reference column (in ref_tilt)
    ref_tilt: str                    # '' if the tilt's own profile was used
    ref_params: tuple                # (theta, delta) of M_ref
    sigma_profile: float
    smoothed: tuple = field(repr=False)  # (Qs, Us, Vs) used for the fit

    def to_meta(self, profile_entry=None):
        m = {
            "theta_fit": self.theta_fit,
            "delta_fit": self.delta_fit,
            "cost": self.cost,
            "ref_index": int(self.ref_index),
            "ref_tilt": self.ref_tilt,
            "ref_theta": float(self.ref_params[0]),
            "ref_delta": float(self.ref_params[1]),
            "sigma_profile": float(self.sigma_profile),
        }
        if profile_entry is not None:
            for key in ("source", "i_range", "k_range", "sigma_stokes", "offset"):
                if key in profile_entry:
                    m[key] = np.asarray(profile_entry[key]) if key.endswith("range") \
                        else profile_entry[key]
        return m


def smooth_profile(Qa, Ua, Va, sigma):
    """1-D Gaussian smoothing along J followed by renormalization."""
    if sigma and sigma > 0:
        Qa = gaussian_filter(Qa, sigma)
        Ua = gaussian_filter(Ua, sigma)
        Va = gaussian_filter(Va, sigma)
    return normalize_stokes(Qa, Ua, Va)


def resolve_reference(Vs, reference):
    """'max_v' -> argmax(Vs); an integer (or digit string) -> that index."""
    if isinstance(reference, str) and reference.lower() == "max_v":
        return int(np.argmax(Vs))
    idx = int(reference)
    if not -len(Vs) <= idx < len(Vs):
        raise IndexError(f"reference index {idx} outside profile of length {len(Vs)}")
    return idx % len(Vs)


def _residual(params, M_meas, M_ref):
    M_LR = linear_retarder(params[0], params[1])
    return (M_LR @ M_ref @ M_LR - M_meas).ravel()


def reference_params(Qa, Ua, Va, sigma=2.0, reference="max_v"):
    """(ref_index, (theta, delta)) of the reference column of a profile."""
    Qs, Us, Vs = smooth_profile(Qa, Ua, Va, sigma)
    idx = resolve_reference(Vs, reference)
    theta, delta = retarder_params_from_stokes(Qs[idx], Us[idx], Vs[idx])
    return idx, (float(theta), float(delta))


def fit_calibration(Qa, Ua, Va, sigma=2.0, reference="max_v",
                    ref_params=None, ref_tilt="", ref_index=None, warm_start=False):
    """Fit the per A-line system retarder M_LR(j).

    Parameters
    ----------
    Qa, Ua, Va : (J,) arrays
        Normalized reference profile (output of :func:`extract_profile`).
    sigma : float
        Gaussian smoothing of the profile along J before fitting.
    reference : 'max_v' or int
        Reference column when ``ref_params`` is not given.
    ref_params : (theta, delta), optional
        Use this retarder as M_ref instead (e.g. taken from another tilt).
    warm_start : bool
        Start each column's fit from the previous column's solution instead
        of (0, 0). Default False reproduces the original notebook.
    """
    Qs, Us, Vs = smooth_profile(Qa, Ua, Va, sigma)
    theta_m, delta_m = retarder_params_from_stokes(Qs, Us, Vs)
    M_meas_all = linear_retarder(theta_m, delta_m)          # (J, 4, 4)

    if ref_params is None:
        ref_index = resolve_reference(Vs, reference)
        ref_params = (float(theta_m[ref_index]), float(delta_m[ref_index]))
        ref_tilt = ""
    M_ref = linear_retarder(*ref_params)

    nJ = len(Qs)
    theta_fit = np.empty(nJ)
    delta_fit = np.empty(nJ)
    cost = np.empty(nJ)
    x0 = np.zeros(2)
    for j in range(nJ):
        out = least_squares(_residual, x0, args=(M_meas_all[j], M_ref))
        theta_fit[j], delta_fit[j] = out.x
        cost[j] = out.cost
        if warm_start:
            x0 = out.x

    return CalibrationResult(
        calibration_M=linear_retarder(theta_fit, delta_fit),
        theta_fit=theta_fit,
        delta_fit=delta_fit,
        cost=cost,
        ref_index=-1 if ref_index is None else int(ref_index),
        ref_tilt=ref_tilt,
        ref_params=tuple(ref_params),
        sigma_profile=float(sigma),
        smoothed=(Qs, Us, Vs),
    )


def calibrate_tilts(profiles, sigma=2.0, reference="max_v", reference_tilt=None,
                    warm_start=False):
    """Fit every tilt in ``profiles`` ({tilt: entry with Qa/Ua/Va}).

    If ``reference_tilt`` is given, all tilts share the reference retarder
    taken from that tilt's profile; otherwise each tilt uses its own.
    """
    shared = None
    if reference_tilt:
        if reference_tilt not in profiles:
            raise KeyError(f"reference tilt '{reference_tilt}' not in profiles "
                           f"({list(profiles)})")
        p = profiles[reference_tilt]
        shared = reference_params(p["Qa"], p["Ua"], p["Va"], sigma, reference)

    results = {}
    for tilt, p in profiles.items():
        kwargs = dict(sigma=sigma, reference=reference, warm_start=warm_start)
        if shared is not None:
            kwargs.update(ref_index=shared[0], ref_params=shared[1], ref_tilt=reference_tilt)
        results[tilt] = fit_calibration(p["Qa"], p["Ua"], p["Va"], **kwargs)
    return results


# --------------------------------------------------------------------------
# correction
# --------------------------------------------------------------------------
def reconstruct_profile(Qa, Ua, Va, calibration_M):
    """Apply the calibration to a (J,) profile -> corrected (Qc, Uc, Vc)."""
    theta, delta = retarder_params_from_stokes(Qa, Ua, Va)
    M_meas = linear_retarder(theta, delta)
    B = np.linalg.inv(calibration_M)
    M_true = B @ M_meas @ B
    return stokes_from_mueller(M_true @ M_true)


def apply_calibration(O3D, R3D, calibration_M, dtype=np.float32):
    """Corrected (Q, U, V) volumes from orientation/retardance maps in degrees.

    O3D, R3D : (I, J, K);  calibration_M : (J, 4, 4).
    Processes one I-slice at a time to bound memory.
    """
    I, J, K = O3D.shape
    if calibration_M.shape[0] != J:
        raise ValueError(f"calibration has {calibration_M.shape[0]} columns, data has J={J}")
    B = np.linalg.inv(calibration_M)[:, None]      # (J, 1, 4, 4), broadcast over K

    Q = np.empty((I, J, K), dtype=dtype)
    U = np.empty((I, J, K), dtype=dtype)
    V = np.empty((I, J, K), dtype=dtype)
    for i in range(I):
        M_meas = linear_retarder(np.radians(O3D[i]), np.radians(R3D[i]))  # (J, K, 4, 4)
        M_true = B @ M_meas @ B
        Q[i], U[i], V[i] = stokes_from_mueller(M_true @ M_true)
    return Q, U, V


def apply_calibration_jones(J1, J2, calibration_M, offset=0.0, dtype=np.float32):
    """Convenience wrapper: Jones slab -> corrected (Q, U, V)."""
    R3D, O3D = jones_to_retardance_orientation(J1, J2, offset)
    return apply_calibration(O3D, R3D, calibration_M, dtype=dtype)
