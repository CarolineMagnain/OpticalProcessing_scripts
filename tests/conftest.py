import h5py
import numpy as np
import pytest


def write_v73_like(path, J1, J2):
    """Write J1/J2 (MATLAB order I,J,K) the way MATLAB v7.3 stores complex data."""
    ctype = np.dtype([("real", "<f8"), ("imag", "<f8")])
    with h5py.File(path, "w") as f:
        for name, arr in (("J1", J1), ("J2", J2)):
            raw = np.empty(arr.shape[::-1], dtype=ctype)
            raw["real"] = arr.real.transpose(2, 1, 0)
            raw["imag"] = arr.imag.transpose(2, 1, 0)
            f.create_dataset(name, data=raw)


def synthetic_jones(shape, seed=0, j_trend=True):
    """Jones volume whose orientation/retardance drift smoothly along J."""
    rng = np.random.default_rng(seed)
    I, J, K = shape
    j = np.linspace(0, 1, J)[None, :, None]
    ret = np.radians(20 + 25 * j) + 0.05 * rng.standard_normal(shape)
    ori = np.radians(10 + 60 * j) + 0.05 * rng.standard_normal(shape)
    amp = 1 + 0.1 * rng.random(shape)
    ph = rng.uniform(-np.pi, np.pi, shape)
    J1 = amp * np.sin(ret) * np.exp(1j * (ph + 2 * ori))
    J2 = amp * np.cos(ret) * np.exp(1j * ph)
    return J1, J2


@pytest.fixture
def jones_volume():
    return synthetic_jones((24, 30, 18))


@pytest.fixture
def jones_file(tmp_path, jones_volume):
    p = tmp_path / "Jones_tiltNeg20deg_strip0023_PD12.mat"
    write_v73_like(p, *jones_volume)
    return p
