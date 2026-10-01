"""File input/output.

Large Jones files are read lazily, slab by slab, and are never written.
Only small files are written: the reference profiles (profiles.mat) and
the calibration matrices (calibration_profile.mat).

Array order
-----------
All arrays returned to Python use MATLAB's index order (I, J, K), i.e. the
same shape you see with ``size(J1)`` in MATLAB and with ``mat73.loadmat``.
MATLAB v7.3 files are HDF5 files in which h5py sees the axes reversed
(K, J, I); :class:`JonesFile` handles that transpose.
"""

import os
import re

import h5py
import numpy as np
from scipy.io import loadmat, savemat

from . import __version__

TILT_PATTERN = re.compile(r"Jones_(?P<tilt>[A-Za-z][A-Za-z0-9]*)_")
_MATLAB_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,62}$")
_RESERVED_KEYS = {"meta"}


# --------------------------------------------------------------------------
# names
# --------------------------------------------------------------------------
def tilt_from_filename(path, pattern=TILT_PATTERN):
    """'Jones_tiltNeg20deg_strip0023_PD12.mat' -> 'tiltNeg20deg'."""
    m = pattern.search(os.path.basename(path))
    if m is None:
        raise ValueError(
            f"Cannot infer the tilt name from '{os.path.basename(path)}'. "
            "Expected 'Jones_<tilt>_...'; pass the name explicitly with --tilt."
        )
    return m.group("tilt")


def check_key(name):
    """Ensure ``name`` can be used as a MATLAB variable / struct field name."""
    if not _MATLAB_NAME.match(name):
        raise ValueError(
            f"'{name}' is not a valid MATLAB variable name "
            "(letter first, then letters/digits/underscores, max 63 chars)."
        )
    if name in _RESERVED_KEYS:
        raise ValueError(f"'{name}' is reserved.")
    return name


# --------------------------------------------------------------------------
# Jones input
# --------------------------------------------------------------------------
def _to_complex(raw):
    if raw.dtype.names and "real" in raw.dtype.names:
        return raw["real"] + 1j * raw["imag"]
    return raw


class JonesFile:
    """Lazy reader for the two Jones channels in a .mat file.

    MATLAB v7.3 (HDF5) files are read slab by slab without loading the
    full volume. Older .mat versions are not HDF5 and are loaded fully with
    scipy as a fallback.

    Use as a context manager::

        with JonesFile(path) as jf:
            I, J, K = jf.shape
            J1, J2 = jf.read(i0, i1, k0, k1)
    """

    def __init__(self, path, names=("J1", "J2")):
        self.path = path
        self.names = tuple(names)
        if len(self.names) != 2:
            raise ValueError("names must contain exactly two variable names")
        self._h5 = None
        self._arrays = None

        if h5py.is_hdf5(path):
            self._h5 = h5py.File(path, "r")
            missing = [n for n in self.names if n not in self._h5]
            if missing:
                raise KeyError(f"{missing} not found in {path}; "
                               f"available: {list(self._h5.keys())}")
            dsets = [self._h5[n] for n in self.names]
            shapes = {d.shape for d in dsets}
            if len(shapes) != 1:
                raise ValueError(f"{self.names} have different shapes: {shapes}")
            if dsets[0].ndim != 3:
                raise ValueError(f"Expected 3-D volumes, got shape {dsets[0].shape}")
            self._dsets = dsets
            self.shape = tuple(reversed(dsets[0].shape))  # MATLAB order
        else:
            d = loadmat(path, variable_names=list(self.names))
            self._arrays = [np.asarray(d[n]) for n in self.names]
            if self._arrays[0].ndim != 3:
                raise ValueError(f"Expected 3-D volumes, got shape {self._arrays[0].shape}")
            self.shape = self._arrays[0].shape

    def read(self, i0, i1, k0, k1):
        """Return (J1, J2) for I-range [i0, i1) and K-range [k0, k1), all of J.

        Output shape is (i1 - i0, J, k1 - k0) in MATLAB order.
        """
        I, _, K = self.shape
        if not (0 <= i0 < i1 <= I and 0 <= k0 < k1 <= K):
            raise IndexError(f"range i=[{i0},{i1}) k=[{k0},{k1}) outside volume {self.shape}")
        if self._h5 is not None:
            out = []
            for d in self._dsets:
                raw = d[k0:k1, :, i0:i1]  # HDF5 order (K, J, I)
                out.append(np.ascontiguousarray(_to_complex(raw).transpose(2, 1, 0)))
            return tuple(out)
        return tuple(a[i0:i1, :, k0:k1] for a in self._arrays)

    def iter_slabs(self, chunk):
        """Yield (i0, i1, J1, J2) over the full volume in I-chunks."""
        I, _, K = self.shape
        for i0 in range(0, I, chunk):
            i1 = min(i0 + chunk, I)
            J1, J2 = self.read(i0, i1, 0, K)
            yield i0, i1, J1, J2

    def close(self):
        if self._h5 is not None:
            self._h5.close()
            self._h5 = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


# --------------------------------------------------------------------------
# small .mat files
# --------------------------------------------------------------------------
def load_mat_dict(path):
    """Load a small .mat file as nested dicts, dropping the __header__ etc. keys."""
    if not os.path.exists(path):
        return {}
    d = loadmat(path, simplify_cells=True)
    return {k: v for k, v in d.items() if not k.startswith("__")}


def make_profile_entry(Qa, Ua, Va, source, i_range, k_range, sigma_stokes, offset=0.0):
    return {
        "Qa": np.asarray(Qa, dtype=float),
        "Ua": np.asarray(Ua, dtype=float),
        "Va": np.asarray(Va, dtype=float),
        "source": os.path.basename(source),
        "i_range": np.asarray(i_range, dtype=np.int64),
        "k_range": np.asarray(k_range, dtype=np.int64),
        "sigma_stokes": float(sigma_stokes),
        "offset": float(offset),
        "version": __version__,
    }


def update_profile(path, tilt, entry):
    """Insert/replace one tilt in profiles.mat, keeping the other tilts."""
    check_key(tilt)
    d = load_mat_dict(path)
    d[tilt] = entry
    savemat(path, d, do_compression=True)


def load_profiles(path):
    """Return {tilt: entry} with Qa/Ua/Va as 1-D float arrays."""
    d = load_mat_dict(path)
    if not d:
        raise FileNotFoundError(f"No profiles found in {path}")
    out = {}
    for tilt, e in d.items():
        e = dict(e)
        for key in ("Qa", "Ua", "Va"):
            e[key] = np.atleast_1d(np.asarray(e[key], dtype=float))
        for key in ("i_range", "k_range"):
            e[key] = tuple(int(x) for x in np.atleast_1d(e[key]))
        e["source"] = str(e.get("source", ""))
        e["sigma_stokes"] = float(e.get("sigma_stokes", np.nan))
        e["offset"] = float(e.get("offset", 0.0))
        out[tilt] = e
    return out


def save_calibration(path, calibration_MMs, meta=None):
    """Write {tilt: (J, 4, 4) array} (plus optional 'meta' struct) to one .mat file."""
    out = {}
    for tilt, M in calibration_MMs.items():
        check_key(tilt)
        M = np.asarray(M, dtype=float)
        if M.ndim != 3 or M.shape[1:] != (4, 4):
            raise ValueError(f"calibration for {tilt} has shape {M.shape}, expected (J, 4, 4)")
        out[tilt] = M
    if meta:
        out["meta"] = meta
    savemat(path, out, do_compression=True)


def load_calibration(path):
    """Return {tilt: (J, 4, 4) array}, ignoring the 'meta' entry."""
    d = load_mat_dict(path)
    d.pop("meta", None)
    return {k: np.asarray(v, dtype=float).reshape(-1, 4, 4) for k, v in d.items()}
