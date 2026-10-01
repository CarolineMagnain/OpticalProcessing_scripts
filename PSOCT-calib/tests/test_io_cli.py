import h5py
import numpy as np
import pytest
import yaml

from conftest import synthetic_jones, write_v73_like
from psoct_calib.cli import main
from psoct_calib.io import (
    JonesFile, check_key, load_calibration, load_mat_dict, save_calibration,
    tilt_from_filename,
)

TILTS = ["tiltNeg20deg", "tiltPos20deg", "normal0deg"]


def test_tilt_from_filename():
    assert tilt_from_filename("/x/Jones_tiltNeg20deg_strip0023_PD12.mat") == "tiltNeg20deg"
    assert tilt_from_filename("Jones_normal0deg_strip0001_PD3.mat") == "normal0deg"
    with pytest.raises(ValueError):
        tilt_from_filename("something_else.mat")
    with pytest.raises(ValueError):
        check_key("20deg")


def test_jones_reader_order(jones_file, jones_volume):
    with JonesFile(jones_file) as jf:
        assert jf.shape == jones_volume[0].shape
        J1, J2 = jf.read(3, 9, 2, 7)
        assert np.allclose(J1, jones_volume[0][3:9, :, 2:7])
        assert np.allclose(J2, jones_volume[1][3:9, :, 2:7])


def test_calibration_file_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    MMs = {t: rng.standard_normal((12, 4, 4)) for t in TILTS}
    p = tmp_path / "calibration_profile.mat"
    save_calibration(p, MMs, meta={"version": "x"})
    back = load_calibration(p)
    assert set(back) == set(TILTS)
    for t in TILTS:
        assert np.allclose(back[t], MMs[t])


def _make_dataset(tmp_path):
    for n, t in enumerate(TILTS):
        write_v73_like(tmp_path / f"Jones_{t}_strip0023_PD12.mat",
                       *synthetic_jones((24, 30, 18), seed=n))


def test_extract_calibrate_apply(tmp_path):
    _make_dataset(tmp_path)
    prof = str(tmp_path / "profiles.mat")
    cal = str(tmp_path / "calibration_profile.mat")
    files = [str(tmp_path / f"Jones_{t}_strip0023_PD12.mat") for t in TILTS]

    main(["extract", *files, "--i-range", "4:20", "--k-range", "3:15", "-o", prof])
    main(["calibrate", prof, "-o", cal, "--plot", str(tmp_path / "fit.png")])

    raw = load_mat_dict(cal)
    assert set(raw) == set(TILTS) | {"meta"}
    for t in TILTS:
        assert raw[t].shape == (30, 4, 4)
    assert (tmp_path / "fit.png").exists()

    out = str(tmp_path / "corrected.h5")
    main(["apply", files[0], "--calib", cal, "-o", out, "--chunk", "7"])
    with h5py.File(out, "r") as f:
        assert f["Q"].shape == (18, 30, 24)      # MATLAB h5read gives (24, 30, 18)
        assert f.attrs["tilt"] == "tiltNeg20deg"
        assert np.all(np.isfinite(f["V"][...]))


def test_run_config_and_shared_reference(tmp_path, capsys):
    _make_dataset(tmp_path)
    cfg = {
        "data_dir": ".",
        "output": "calibration_profile.mat",
        "reference_tilt": "normal0deg",
        "tilts": {t: {"file": f"Jones_{t}_strip0023_PD12.mat",
                      "i_range": [4, 20], "k_range": [3, 15]} for t in TILTS},
    }
    cfg_path = tmp_path / "calib_config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))

    main(["run", str(cfg_path)])
    main(["run", str(cfg_path)])                  # second run skips extraction
    assert "up to date" in capsys.readouterr().out

    raw = load_mat_dict(tmp_path / "calibration_profile.mat")
    assert set(raw) == set(TILTS) | {"meta"}
    refs = {raw["meta"][t]["ref_tilt"] for t in TILTS}
    assert refs == {"normal0deg"}
