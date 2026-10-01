"""apply-tiles must reproduce the original per-tile processing script."""
import os

import numpy as np
from numpy import arccos, arctan2, cos, log10, pi, sin, sqrt
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter

from conftest import synthetic_jones, write_v73_like
from psoct_calib.batch import parse_tiles, process_tile, run_tiles
from psoct_calib.cli import main
from psoct_calib.io import save_calibration
from psoct_calib.mueller import linear_retarder


# ---- original script, per tile (verbatim logic) --------------------------
def script_tile(Jones1, Jones2, calibration_M):
    inten = 10 * log10(sqrt(abs(Jones1) ** 2 + abs(Jones2) ** 2))
    R3D = np.arctan(np.abs(Jones1) / np.abs(Jones2)) / np.pi * 180
    phi = np.angle(Jones1) - np.angle(Jones2)
    phi[phi > np.pi] -= 2 * np.pi
    phi[phi < -np.pi] += 2 * np.pi
    O3D = phi / (2 * np.pi) * 180
    Q = gaussian_filter(sin(O3D / 180 * pi * 2) * sin(R3D / 180 * pi * 2), 1)
    U = gaussian_filter(cos(O3D / 180 * pi * 2) * sin(R3D / 180 * pi * 2), 1)
    V = gaussian_filter(cos(R3D / 180 * pi * 2), 1)
    n = sqrt(Q**2 + U**2 + V**2)
    Q, U, V = Q / n, U / n, V / n
    O, R = arctan2(Q, U) / pi * 180 / 2, arccos(V) / pi * 180 / 2

    I, J, K = O.shape
    B = np.linalg.inv(calibration_M)[:, None]
    Qd, Ud, Vd = (np.zeros((I, J, K)) for _ in range(3))
    for i in range(I):
        Mm = linear_retarder(O[i] * pi / 180, R[i] * pi / 180)
        Mt = B @ Mm @ B
        Mc = Mt @ Mt
        Qd[i], Ud[i], Vd[i] = -Mc[..., 1, 3], Mc[..., 2, 3], Mc[..., 3, 3]
    return {k: v.astype("float32") for k, v in
            dict(Q=Qd, U=Ud, V=Vd, inten=inten).items()}
# ---------------------------------------------------------------------------


def _calib(J, seed=5):
    rng = np.random.default_rng(seed)
    return linear_retarder(rng.uniform(-1, 1, J), rng.uniform(0, 0.6, J))


def test_process_tile_matches_script():
    J1, J2 = synthetic_jones((12, 20, 16), seed=7)
    M = _calib(20)
    ref = script_tile(J1, J2, M)
    got = process_tile(J1, J2, M)
    for k in ("Q", "U", "V", "inten"):
        assert got[k].dtype == np.float32
        assert np.allclose(got[k], ref[k], atol=1e-6), k


def test_parse_tiles():
    assert parse_tiles("1-55") == list(range(1, 56))
    assert parse_tiles("1,4,10-12") == [1, 4, 10, 11, 12]


def test_apply_tiles_cli(tmp_path):
    data, out = tmp_path / "data", tmp_path / "out"
    data.mkdir()
    vols = {}
    for tile in (1, 2, 3):
        vols[tile] = synthetic_jones((10, 20, 14), seed=tile)
        write_v73_like(data / f"Jones_normal0deg_slice3_strip{tile:04d}_PD12.mat", *vols[tile])
    cal = tmp_path / "calibration_profile.mat"
    M = _calib(20)
    save_calibration(cal, {"normal0deg": M, "tiltNeg20deg": _calib(20, 6)})

    args = ["apply-tiles", "--tilt", "normal0deg", "--slice", "3", "--path-in", str(data),
            "--path-out", str(out), "--calib", str(cal), "--tiles", "1-3"]
    main(args)

    names = sorted(os.listdir(out))
    assert len(names) == 12 and not any(".part" in n for n in names)
    for tile in (1, 2, 3):
        ref = script_tile(*vols[tile], M)
        for prefix, var in (("Q", "Q"), ("U", "U"), ("V", "V"), ("dBI3D", "inten")):
            d = loadmat(out / f"{prefix}_normal0deg_slice3_strip{tile:04d}.mat")
            assert d[var].shape == (10, 20, 14) and d[var].dtype == np.float32
            assert np.allclose(d[var], ref[var], atol=1e-6)

    # restart: nothing to redo
    assert run_tiles(str(data), str(out), str(cal), "normal0deg", 3, "1-3",
                     skip_existing=True, log=lambda *a: None) == []
    # missing tile handling
    assert run_tiles(str(data), str(out), str(cal), "normal0deg", 3, "3-4",
                     skip_missing=True, log=lambda *a: None) == [3]


def test_apply_tiles_without_slice(tmp_path):
    data, out = tmp_path / "data", tmp_path / "out"
    data.mkdir()
    vol = synthetic_jones((8, 20, 12), seed=11)
    write_v73_like(data / "Jones_normal0deg_strip0001_PD12.mat", *vol)
    cal = tmp_path / "calibration_profile.mat"
    M = _calib(20)
    save_calibration(cal, {"normal0deg": M})

    main(["apply-tiles", "--tilt", "normal0deg", "--tiles", "1", "--path-in", str(data),
          "--path-out", str(out), "--calib", str(cal)])

    assert sorted(os.listdir(out)) == sorted(
        f"{p}_normal0deg_strip0001.mat" for p in ("Q", "U", "V", "dBI3D"))
    ref = script_tile(*vol, M)
    d = loadmat(out / "Q_normal0deg_strip0001.mat")
    assert np.allclose(d["Q"], ref["Q"], atol=1e-6)


def test_slice_pattern_without_slice_is_an_error(tmp_path):
    import pytest
    with pytest.raises(ValueError, match="slice"):
        run_tiles(str(tmp_path), str(tmp_path), {"normal0deg": np.eye(4)[None]},
                  "normal0deg", None, "1",
                  input_pattern="Jones_{tilt}_slice{slice}_strip{tile:04d}_PD12.mat",
                  log=lambda *a: None)
