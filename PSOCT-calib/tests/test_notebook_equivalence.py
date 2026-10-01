"""The package must reproduce the original notebook exactly."""
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.optimize import least_squares

from psoct_calib.calibration import (
    apply_calibration, extract_profile, fit_calibration, reconstruct_profile,
)
from psoct_calib.io import JonesFile
from psoct_calib.mueller import linear_retarder


# ---- original notebook code (verbatim logic) -----------------------------
def nb_stokes(J1, J2):
    R3D = np.arctan(np.abs(J1) / np.abs(J2)) / np.pi * 180
    phi = np.angle(J1) - np.angle(J2)
    phi[phi > np.pi] -= 2 * np.pi
    phi[phi < -np.pi] += 2 * np.pi
    O3D = phi / (2 * np.pi) * 180
    Q = gaussian_filter(np.sin(O3D / 180 * np.pi * 2) * np.sin(R3D / 180 * np.pi * 2), 1)
    U = gaussian_filter(np.cos(O3D / 180 * np.pi * 2) * np.sin(R3D / 180 * np.pi * 2), 1)
    V = gaussian_filter(np.cos(R3D / 180 * np.pi * 2), 1)
    n = np.sqrt(Q**2 + U**2 + V**2)
    return Q / n, U / n, V / n, O3D, R3D


def nb_profile(Q, U, V, isl, ksl):
    Qa = np.mean(np.mean(Q[isl, :, ksl], axis=2), axis=0)
    Ua = np.mean(np.mean(U[isl, :, ksl], axis=2), axis=0)
    Va = np.mean(np.mean(V[isl, :, ksl], axis=2), axis=0)
    n = np.sqrt(Qa**2 + Ua**2 + Va**2)
    return Qa / n, Ua / n, Va / n


def nb_calibrate(Qa_o, Ua_o, Va_o):
    Qa, Ua, Va = (gaussian_filter(x, 2) for x in (Qa_o, Ua_o, Va_o))
    n = np.sqrt(Qa**2 + Ua**2 + Va**2)
    Qa, Ua, Va = Qa / n, Ua / n, Va / n

    def res(p, Mm, Mr):
        L = linear_retarder(*p)
        return (L @ Mr @ L - Mm).ravel()

    r = np.argmax(Va)
    M_ref = linear_retarder(np.arctan2(Qa[r], Ua[r]) / 2, np.arccos(Va[r]) / 2)
    Ms, Qc, Uc, Vc = [], [], [], []
    for i in range(len(Qa)):
        M_meas = linear_retarder(np.arctan2(Qa[i], Ua[i]) / 2, np.arccos(Va[i]) / 2)
        out = least_squares(res, np.zeros(2), args=(M_meas, M_ref))
        M_LR = linear_retarder(*out.x)
        Ms.append(M_LR)
        M_meas = linear_retarder(np.arctan2(Qa_o[i], Ua_o[i]) / 2, np.arccos(Va_o[i]) / 2)
        Minv = np.linalg.inv(M_LR)
        Mt = Minv @ M_meas @ Minv
        Mc = Mt @ Mt
        Qc.append(-Mc[1, 3]); Uc.append(Mc[2, 3]); Vc.append(Mc[3, 3])
    return np.array(Ms), np.array(Qc), np.array(Uc), np.array(Vc)
# ---------------------------------------------------------------------------


def test_extract_matches_full_volume(jones_file, jones_volume):
    Qf, Uf, Vf, _, _ = nb_stokes(*jones_volume)
    with JonesFile(jones_file) as jf:
        # interior ROI and ROIs touching the volume edges
        for ir, kr in [((6, 15), (5, 11)), ((0, 5), (0, 4)), ((20, 24), (14, 18))]:
            ref = nb_profile(Qf, Uf, Vf, slice(*ir), slice(*kr))
            got = extract_profile(jf, ir, kr, sigma_stokes=1)
            for a, b in zip(got, ref):
                assert np.allclose(a, b, atol=1e-12)


def test_calibration_matches_notebook(jones_volume):
    Qf, Uf, Vf, _, _ = nb_stokes(*jones_volume)
    prof = nb_profile(Qf, Uf, Vf, slice(4, 20), slice(3, 15))
    M_nb, Qc_nb, Uc_nb, Vc_nb = nb_calibrate(*prof)

    res = fit_calibration(*prof, sigma=2, reference="max_v")
    assert np.allclose(res.calibration_M, M_nb, atol=1e-10)
    Qc, Uc, Vc = reconstruct_profile(*prof, res.calibration_M)
    assert np.allclose(Qc, Qc_nb) and np.allclose(Uc, Uc_nb) and np.allclose(Vc, Vc_nb)


def test_apply_matches_per_voxel_loop(jones_volume):
    _, _, _, O3D, R3D = nb_stokes(*jones_volume)
    O3D, R3D = O3D[:3], R3D[:3]
    rng = np.random.default_rng(3)
    M = linear_retarder(rng.uniform(-1, 1, O3D.shape[1]), rng.uniform(0, 0.5, O3D.shape[1]))
    Q, U, V = apply_calibration(O3D, R3D, M, dtype=np.float64)
    for i, j, k in [(0, 0, 0), (1, 7, 3), (2, 29, 17)]:
        Mm = linear_retarder(np.radians(O3D[i, j, k]), np.radians(R3D[i, j, k]))
        B = np.linalg.inv(M[j])
        Mc = (B @ Mm @ B) @ (B @ Mm @ B)
        assert np.allclose([Q[i, j, k], U[i, j, k], V[i, j, k]],
                           [-Mc[1, 3], Mc[2, 3], Mc[3, 3]])


def test_constant_profile_gives_identity_calibration():
    J = 25
    th, de = 0.4, 0.6   # single-pass retarder
    Qa = np.full(J, np.sin(2 * th) * np.sin(2 * de))
    Ua = np.full(J, np.cos(2 * th) * np.sin(2 * de))
    Va = np.full(J, np.cos(2 * de))
    res = fit_calibration(Qa, Ua, Va)
    assert np.allclose(res.calibration_M, np.eye(4), atol=1e-8)
    Qc, Uc, Vc = reconstruct_profile(Qa, Ua, Va, res.calibration_M)
    assert np.allclose(Qc, Qa) and np.allclose(Uc, Ua) and np.allclose(Vc, Va)
