"""Apply a calibration to every tile of one tilt and slice.

This reproduces the per-tile processing script used for the full datasets.
For each tile ``Jones_<tilt>_slice<N>_strip<TTTT>_PD12.mat``:

1. load J1, J2;
2. intensity  dBI3D = 10 log10( sqrt(|J1|^2 + |J2|^2) );
3. Q, U, V from orientation/retardance, Gaussian-filtered (sigma = 1) and
   normalized, as in :func:`psoct_calib.stokes.stokes_from_orientation_retardance`;
4. the smoothed Stokes vector is converted back to a single-pass retarder
   (theta = atan2(Q,U)/2, delta = acos(V)/2);
5. the calibration of that tilt is applied with
   :func:`psoct_calib.calibration.apply_calibration`;
6. Q, U, V and dBI3D are written as separate float32 .mat files:
   ``Q_<tilt>_slice<N>_strip<TTTT>.mat`` (variable ``Q``), likewise ``U``, ``V``,
   and ``dBI3D_...mat`` (variable ``inten``).

Without a slice number, files are named ``Jones_<tilt>_strip<TTTT>_PD12.mat``
and ``Q_<tilt>_strip<TTTT>.mat`` etc. instead.

Each tile is independent, so different tilts/slices can run as parallel jobs.

Command line::

    psoct-calib apply-tiles --tilt normal0deg --slice 3 \\
        --path-in DATA --path-out OUT --calib calibration_profile.mat
    psoct-calib apply-tiles --tilt normal0deg \\
        --path-in DATA --path-out OUT --calib calibration_profile.mat   # no slice
"""

import os
import time

import numpy as np
from scipy.io import savemat

from .calibration import apply_calibration
from .io import JonesFile, check_key, load_calibration
from .mueller import retarder_params_from_stokes
from .stokes import jones_to_retardance_orientation, stokes_from_orientation_retardance

DEFAULT_INPUT_PATTERN = "Jones_{tilt}_slice{slice}_strip{tile:04d}_PD12.mat"
DEFAULT_OUTPUT_PATTERN = "{prefix}_{tilt}_slice{slice}_strip{tile:04d}.mat"
NOSLICE_INPUT_PATTERN = "Jones_{tilt}_strip{tile:04d}_PD12.mat"
NOSLICE_OUTPUT_PATTERN = "{prefix}_{tilt}_strip{tile:04d}.mat"
DEFAULT_TILES = "1-55"

# (file prefix, variable name inside the .mat file)
OUTPUTS = (("Q", "Q"), ("U", "U"), ("V", "V"), ("dBI3D", "inten"))

_MAT5_LIMIT = 2**32 - 1   # bytes per variable scipy.io.savemat can write (v5 format)


# --------------------------------------------------------------------------
# per-tile computation
# --------------------------------------------------------------------------
def intensity_db(J1, J2):
    """10 log10( sqrt(|J1|^2 + |J2|^2) ). Zero signal gives -inf."""
    with np.errstate(divide="ignore"):
        return 10 * np.log10(np.sqrt(np.abs(J1) ** 2 + np.abs(J2) ** 2))


def smoothed_orientation_retardance(J1, J2, sigma=1.0, offset=0.0):
    """Orientation and retardance [deg] of the Gaussian-smoothed Stokes vector.

    Returns (O_s, R_s) with O_s = atan2(Q, U)/2 and R_s = acos(V)/2 in degrees,
    where (Q, U, V) are the filtered, normalized Stokes volumes.
    """
    R3D, O3D = jones_to_retardance_orientation(J1, J2, offset)
    Q, U, V = stokes_from_orientation_retardance(O3D, R3D, sigma)
    del R3D, O3D
    theta, delta = retarder_params_from_stokes(Q, U, V)
    return np.degrees(theta), np.degrees(delta)


def process_tile(J1, J2, calibration_M, sigma=1.0, offset=0.0, intensity=True):
    """Corrected Q, U, V (and intensity) of one tile.

    Returns a dict with float32 arrays of shape (I, J, K):
    ``{'Q', 'U', 'V'}`` and, if ``intensity``, ``'inten'``.
    """
    out = {}
    if intensity:
        out["inten"] = intensity_db(J1, J2).astype(np.float32)
    O_s, R_s = smoothed_orientation_retardance(J1, J2, sigma, offset)
    Q, U, V = apply_calibration(O_s, R_s, calibration_M, dtype=np.float32)
    out.update(Q=Q, U=U, V=V)
    return out


# --------------------------------------------------------------------------
# file handling
# --------------------------------------------------------------------------
def parse_tiles(text):
    """'1-55' -> [1..55] (inclusive); '1,4,10-12' -> [1, 4, 10, 11, 12]."""
    tiles = []
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = (int(x) for x in part.split("-", 1))
            if b < a:
                raise ValueError(f"empty tile range '{part}'")
            tiles.extend(range(a, b + 1))
        else:
            tiles.append(int(part))
    if not tiles:
        raise ValueError(f"no tiles in '{text}'")
    return tiles


def _save_var(path, var, arr, compress=False):
    if arr.nbytes > _MAT5_LIMIT:
        raise ValueError(
            f"{os.path.basename(path)}: '{var}' is {arr.nbytes / 2**30:.2f} GiB, above "
            "the 4 GiB per-variable limit of .mat files written by scipy.io.savemat."
        )
    tmp = path[:-4] + ".part.mat"   # write then rename: no half-written outputs
    savemat(tmp, {var: arr}, do_compression=compress)
    os.replace(tmp, path)


def resolve_patterns(slice_num, input_pattern=None, output_pattern=None):
    """Default file patterns with or without a slice number.

    Explicit patterns are kept as given. A pattern that uses ``{slice}``
    requires a slice number.
    """
    with_slice = slice_num is not None
    if input_pattern is None:
        input_pattern = DEFAULT_INPUT_PATTERN if with_slice else NOSLICE_INPUT_PATTERN
    if output_pattern is None:
        output_pattern = DEFAULT_OUTPUT_PATTERN if with_slice else NOSLICE_OUTPUT_PATTERN
    if not with_slice:
        for name, pat in (("input", input_pattern), ("output", output_pattern)):
            if "{slice" in pat:
                raise ValueError(f"the {name} pattern '{pat}' uses {{slice}} but no slice "
                                 "number was given")
    return input_pattern, output_pattern


def output_paths(path_out, tilt, slice_num, tile, intensity=True,
                 output_pattern=DEFAULT_OUTPUT_PATTERN):
    """{prefix: path} of the files written for one tile."""
    return {
        prefix: os.path.join(path_out, output_pattern.format(
            prefix=prefix, tilt=tilt, slice=slice_num, tile=tile))
        for prefix, _ in OUTPUTS
        if intensity or prefix != "dBI3D"
    }


def run_tiles(path_in, path_out, calib, tilt, slice_num=None, tiles=DEFAULT_TILES,
              sigma=1.0, offset=0.0, names=("J1", "J2"), intensity=True,
              input_pattern=None, output_pattern=None,
              skip_existing=False, skip_missing=False, compress=False, log=print):
    """Process every tile of one tilt and slice.

    Parameters
    ----------
    calib : str or dict
        Path to calibration_profile.mat, or an already loaded {tilt: (J,4,4)} dict.
    slice_num : int or None
        Slice number in the filenames. None -> files named without a slice
        (``Jones_<tilt>_strip<TTTT>_PD12.mat``).
    input_pattern, output_pattern : str or None
        Filename patterns; None chooses the default for ``slice_num``.
    tiles : str or iterable of int
        Tile numbers as in the filenames, e.g. '1-55' (inclusive) or [1, 2, 3].
    skip_existing : bool
        Skip a tile whose output files all exist already (for restarting jobs).
    skip_missing : bool
        Skip missing input files instead of raising FileNotFoundError.

    Returns
    -------
    list of int : the tiles that were processed.
    """
    check_key(tilt)
    if isinstance(calib, (str, os.PathLike)):
        calib = load_calibration(calib)
    if tilt not in calib:
        raise KeyError(f"no calibration for '{tilt}' (have {sorted(calib)})")
    M = np.asarray(calib[tilt], dtype=float)

    input_pattern, output_pattern = resolve_patterns(slice_num, input_pattern, output_pattern)
    tiles = parse_tiles(tiles) if isinstance(tiles, str) else [int(t) for t in tiles]
    os.makedirs(path_out, exist_ok=True)
    where = f"slice {slice_num}" if slice_num is not None else "no slice"
    log(f"Processing {tilt}, {where}: {len(tiles)} tiles")

    done = []
    for n, tile in enumerate(tiles, 1):
        outs = output_paths(path_out, tilt, slice_num, tile, intensity, output_pattern)
        if skip_existing and all(os.path.exists(p) for p in outs.values()):
            log(f"  tile {tile} ({n}/{len(tiles)}): outputs exist, skipping")
            continue

        src = os.path.join(path_in, input_pattern.format(tilt=tilt, slice=slice_num, tile=tile))
        if not os.path.exists(src):
            if skip_missing:
                log(f"  tile {tile} ({n}/{len(tiles)}): {os.path.basename(src)} missing, skipping")
                continue
            raise FileNotFoundError(src)

        t0 = time.time()
        with JonesFile(src, names=names) as jf:
            I, J, K = jf.shape
            if M.shape[0] != J:
                raise ValueError(f"calibration '{tilt}' has {M.shape[0]} columns, "
                                 f"{os.path.basename(src)} has J={J}")
            J1, J2 = jf.read(0, I, 0, K)
        res = process_tile(J1, J2, M, sigma, offset, intensity)
        del J1, J2

        for prefix, var in OUTPUTS:
            if prefix in outs:
                _save_var(outs[prefix], var, res[var], compress)
        done.append(tile)
        log(f"  tile {tile} ({n}/{len(tiles)}): {os.path.basename(src)} "
            f"(I,J,K)={res['Q'].shape}  {time.time() - t0:.1f} s")
    log("Done.")
    return done


# --------------------------------------------------------------------------
# command line: registered in cli.py as 'psoct-calib apply-tiles'
# --------------------------------------------------------------------------
def _cmd_apply_tiles(args):
    run_tiles(args.path_in, args.path_out, args.calib, args.tilt, args.slice,
              tiles=args.tiles, sigma=args.sigma_stokes, offset=args.offset,
              names=args.vars, intensity=not args.no_intensity,
              input_pattern=args.input_pattern, output_pattern=args.output_pattern,
              skip_existing=args.skip_existing, skip_missing=args.skip_missing,
              compress=args.compress)


def add_cli(subparsers):
    t = subparsers.add_parser(
        "apply-tiles",
        help="apply a calibration to all tiles of one tilt/slice (Q, U, V, dBI3D .mat files)",
        description=__doc__.split("\n\n")[0],
    )
    t.add_argument("--tilt", required=True, help="e.g. normal0deg, tiltNeg20deg, tiltPos20deg")
    t.add_argument("--slice", type=int, default=None,
                   help="slice number in the filenames; omit for files named "
                        "Jones_<tilt>_strip<TTTT>_PD12.mat")
    t.add_argument("--path-in", required=True, help="folder with the Jones .mat files")
    t.add_argument("--path-out", required=True, help="folder for the Q/U/V/dBI3D files")
    t.add_argument("--calib", required=True, help="calibration_profile.mat")
    t.add_argument("--tiles", default=DEFAULT_TILES,
                   help=f"tile numbers, inclusive: '1-55', '3', '1,4,10-12' (default {DEFAULT_TILES})")
    t.add_argument("--sigma-stokes", type=float, default=1.0,
                   help="Gaussian sigma for the Q/U/V volumes (default: 1)")
    t.add_argument("--offset", type=float, default=0.0, help="phase offset in radians")
    t.add_argument("--vars", nargs=2, default=("J1", "J2"), metavar=("J1", "J2"))
    t.add_argument("--no-intensity", action="store_true", help="do not write dBI3D files")
    t.add_argument("--input-pattern", default=None,
                   help=f"default: {DEFAULT_INPUT_PATTERN} (with --slice) "
                        f"or {NOSLICE_INPUT_PATTERN} (without)")
    t.add_argument("--output-pattern", default=None,
                   help=f"default: {DEFAULT_OUTPUT_PATTERN} (with --slice) "
                        f"or {NOSLICE_OUTPUT_PATTERN} (without)")
    t.add_argument("--skip-existing", action="store_true",
                   help="skip tiles whose outputs already exist (restart an interrupted job)")
    t.add_argument("--skip-missing", action="store_true",
                   help="skip missing input tiles instead of stopping")
    t.add_argument("--compress", action="store_true", help="compress the .mat files")
    t.set_defaults(func=_cmd_apply_tiles)
    return t
