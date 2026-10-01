import numpy as np

from psoct_calib.mueller import (
    linear_retarder, retarder_params_from_stokes, rot_mueller, stokes_from_mueller,
)


def test_zero_retardance_is_identity():
    assert np.allclose(linear_retarder(0.7, 0.0), np.eye(4))


def test_retarder_is_orthogonal_and_inverse_is_negative_retardance():
    M = linear_retarder(0.3, 1.1)
    assert np.allclose(M @ M.T, np.eye(4))
    assert np.allclose(np.linalg.inv(M), linear_retarder(0.3, -1.1))


def test_batch_matches_scalar():
    rng = np.random.default_rng(1)
    th, de = rng.uniform(-2, 2, (5, 3)), rng.uniform(0, 3, (5, 3))
    Mb = linear_retarder(th, de)
    assert Mb.shape == (5, 3, 4, 4)
    for idx in np.ndindex(5, 3):
        assert np.allclose(Mb[idx], linear_retarder(th[idx], de[idx]))


def test_rotation_composition():
    assert np.allclose(rot_mueller(0.4) @ rot_mueller(0.5), rot_mueller(0.9))


def test_stokes_convention_round_trip():
    # A double pass through the same retarder doubles the retardance, so
    # the circular-input output of M(theta, delta)^2 must reproduce Q, U, V.
    rng = np.random.default_rng(2)
    v = rng.standard_normal((50, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    Q, U, V = v.T
    th, de = retarder_params_from_stokes(Q, U, V)
    M = linear_retarder(th, de)
    q, u, w = stokes_from_mueller(M @ M)
    assert np.allclose(q, Q) and np.allclose(u, U) and np.allclose(w, V)
