"""Command-line interface.

    psoct-calib extract   FILE... --i-range A:B --k-range A:B -o profiles.mat
    psoct-calib calibrate profiles.mat -o calibration_profile.mat [--plot fig.png]
    psoct-calib apply     FILE --calib calibration_profile.mat -o corrected.h5
    psoct-calib run       config.yaml
    psoct-calib apply-tiles --tilt T --slice N --path-in DIR --path-out DIR --calib FILE

Ranges are 0-based and half-open (Python slicing): MATLAB 4001:5000 -> 4000:5000.
"""

import argparse
import os
import sys

import numpy as np

from . import __version__
from .calibration import (
    apply_calibration_jones,
    calibrate_tilts,
    extract_profile,
)
from .batch import add_cli as add_apply_tiles_cli
from .io import (
    JonesFile,
    check_key,
    load_calibration,
    load_profiles,
    make_profile_entry,
    save_calibration,
    tilt_from_filename,
    update_profile,
)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def parse_range(text):
    try:
        a, b = text.split(":")
        a, b = int(a), int(b)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected START:STOP, got '{text}'")
    if b <= a:
        raise argparse.ArgumentTypeError(f"empty range '{text}'")
    return a, b


def parse_reference(text):
    if isinstance(text, str) and text.lower() == "max_v":
        return "max_v"
    return int(text)


def _extract_one(path, tilt, i_range, k_range, sigma_stokes, offset, names, profiles_path):
    check_key(tilt)
    with JonesFile(path, names=names) as jf:
        print(f"[extract] {tilt}: {os.path.basename(path)}  volume (I,J,K)={jf.shape}")
        Qa, Ua, Va = extract_profile(jf, i_range, k_range, sigma_stokes, offset)
    entry = make_profile_entry(Qa, Ua, Va, path, i_range, k_range, sigma_stokes, offset)
    update_profile(profiles_path, tilt, entry)
    print(f"[extract] {tilt}: profile of length {len(Qa)} -> {profiles_path}")


def _calibrate_and_save(profiles, output, sigma, reference, reference_tilt,
                        warm_start, plot=None, plot_skip=10, with_meta=True):
    results = calibrate_tilts(profiles, sigma=sigma, reference=reference,
                              reference_tilt=reference_tilt, warm_start=warm_start)
    calibration_MMs = {t: r.calibration_M for t, r in results.items()}
    meta = None
    if with_meta:
        meta = {t: r.to_meta(profiles[t]) for t, r in results.items()}
        meta["version"] = __version__
    save_calibration(output, calibration_MMs, meta)

    for t, r in results.items():
        ref = f"{r.ref_tilt}[{r.ref_index}]" if r.ref_tilt else f"[{r.ref_index}]"
        print(f"[calibrate] {t}: J={len(r.cost)}  ref={ref}  "
              f"max cost={r.cost.max():.3e}")
    print(f"[calibrate] wrote {sorted(calibration_MMs)} -> {output}")

    if plot:
        import matplotlib
        matplotlib.use("Agg")
        from .plotting import plot_calibrations
        plot_calibrations(results, profiles, plot, skip=plot_skip)
        print(f"[calibrate] figure -> {plot}")
    return results


# --------------------------------------------------------------------------
# subcommands
# --------------------------------------------------------------------------
def cmd_extract(args):
    if args.tilt and len(args.files) > 1:
        sys.exit("--tilt can only be used with a single input file")
    for f in args.files:
        tilt = args.tilt or tilt_from_filename(f)
        _extract_one(f, tilt, args.i_range, args.k_range, args.sigma_stokes,
                     args.offset, args.vars, args.output)


def cmd_calibrate(args):
    profiles = load_profiles(args.profiles)
    if args.tilts:
        missing = set(args.tilts) - set(profiles)
        if missing:
            sys.exit(f"tilts {sorted(missing)} not in {args.profiles} ({sorted(profiles)})")
        profiles = {t: profiles[t] for t in args.tilts}
    _calibrate_and_save(profiles, args.output, args.sigma, args.ref, args.ref_tilt,
                        args.warm_start, args.plot, args.plot_skip, not args.no_meta)


def cmd_apply(args):
    import h5py
    from tqdm import tqdm

    calib = load_calibration(args.calib)
    tilt = args.tilt or tilt_from_filename(args.file)
    if tilt not in calib:
        sys.exit(f"no calibration for '{tilt}' in {args.calib} (have {sorted(calib)})")
    M = calib[tilt]
    dtype = np.float64 if args.float64 else np.float32

    with JonesFile(args.file, names=args.vars) as jf, h5py.File(args.output, "w") as out:
        I, J, K = jf.shape
        if M.shape[0] != J:
            sys.exit(f"calibration '{tilt}' has {M.shape[0]} columns but data has J={J}")
        # Stored like MATLAB v7.3 (h5py order K,J,I) so that h5read in MATLAB
        # returns (I,J,K). In Python, transpose(2,1,0) after reading.
        comp = "gzip" if args.compress else None
        dsets = {n: out.create_dataset(n, shape=(K, J, I), dtype=dtype,
                                       chunks=True, compression=comp) for n in "QUV"}
        out.attrs.update(tilt=tilt, source=os.path.basename(args.file),
                         calibration=os.path.basename(args.calib), version=__version__,
                         layout="h5py order (K,J,I); MATLAB h5read order (I,J,K)")
        n_chunks = -(-I // args.chunk)
        for i0, i1, J1, J2 in tqdm(jf.iter_slabs(args.chunk), total=n_chunks, desc=tilt):
            Q, U, V = apply_calibration_jones(J1, J2, M, args.offset, dtype)
            for name, arr in zip("QUV", (Q, U, V)):
                dsets[name][:, :, i0:i1] = arr.transpose(2, 1, 0)
    print(f"[apply] {tilt}: corrected Q,U,V -> {args.output}")


def _profile_is_current(entry, source, i_range, k_range, sigma_stokes, offset):
    return (entry.get("source") == os.path.basename(source)
            and tuple(entry.get("i_range", ())) == tuple(i_range)
            and tuple(entry.get("k_range", ())) == tuple(k_range)
            and np.isclose(entry.get("sigma_stokes", np.nan), sigma_stokes)
            and np.isclose(entry.get("offset", 0.0), offset))


def cmd_run(args):
    import yaml

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)
    base = os.path.dirname(os.path.abspath(args.config))

    def resolve(p):
        return p if os.path.isabs(p) else os.path.join(base, p)

    data_dir = resolve(cfg.get("data_dir", "."))
    profiles_path = resolve(cfg.get("profiles", "profiles.mat"))
    output = resolve(cfg.get("output", "calibration_profile.mat"))
    plot = resolve(cfg["plot"]) if cfg.get("plot") else None
    sigma_stokes = float(cfg.get("sigma_stokes", 1.0))
    sigma_profile = float(cfg.get("sigma_profile", 2.0))
    reference = parse_reference(cfg.get("reference", "max_v"))
    reference_tilt = cfg.get("reference_tilt") or None
    offset = float(cfg.get("offset", 0.0))
    names = tuple(cfg.get("variables", ["J1", "J2"]))
    tilts = cfg.get("tilts") or {}
    if not tilts:
        sys.exit("config has no 'tilts' section")

    existing = {}
    if os.path.exists(profiles_path):
        existing = load_profiles(profiles_path)

    for tilt, tc in tilts.items():
        path = os.path.join(data_dir, tc["file"])
        i_range, k_range = tuple(tc["i_range"]), tuple(tc["k_range"])
        s = float(tc.get("sigma_stokes", sigma_stokes))
        if (not args.force and tilt in existing
                and _profile_is_current(existing[tilt], path, i_range, k_range, s, offset)):
            print(f"[extract] {tilt}: up to date in {profiles_path}, skipping")
            continue
        _extract_one(path, tilt, i_range, k_range, s, offset, names, profiles_path)

    profiles = load_profiles(profiles_path)
    profiles = {t: profiles[t] for t in tilts}
    _calibrate_and_save(profiles, output, sigma_profile, reference, reference_tilt,
                        bool(cfg.get("warm_start", False)), plot,
                        int(cfg.get("plot_skip", 10)), bool(cfg.get("save_meta", True)))


# --------------------------------------------------------------------------
# parser
# --------------------------------------------------------------------------
def build_parser():
    p = argparse.ArgumentParser(prog="psoct-calib", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="command", required=True)

    def add_jones_opts(sp):
        sp.add_argument("--vars", nargs=2, default=("J1", "J2"), metavar=("J1", "J2"),
                        help="variable names of the Jones channels (default: J1 J2)")
        sp.add_argument("--offset", type=float, default=0.0,
                        help="phase offset in radians (default: 0)")

    e = sub.add_parser("extract", help="Jones file(s) -> reference profile(s) in a small .mat")
    e.add_argument("files", nargs="+")
    e.add_argument("--i-range", type=parse_range, required=True, help="START:STOP along I")
    e.add_argument("--k-range", type=parse_range, required=True, help="START:STOP along K")
    e.add_argument("--sigma-stokes", type=float, default=1.0,
                   help="Gaussian sigma for the Q/U/V volumes (default: 1)")
    e.add_argument("--tilt", help="key name (default: parsed from 'Jones_<tilt>_...')")
    e.add_argument("-o", "--output", default="profiles.mat")
    add_jones_opts(e)
    e.set_defaults(func=cmd_extract)

    c = sub.add_parser("calibrate", help="profiles.mat -> calibration_profile.mat")
    c.add_argument("profiles")
    c.add_argument("-o", "--output", default="calibration_profile.mat")
    c.add_argument("--sigma", type=float, default=2.0,
                   help="Gaussian sigma along J for the profile (default: 2)")
    c.add_argument("--ref", type=parse_reference, default="max_v",
                   help="reference column: 'max_v' (default) or an index")
    c.add_argument("--ref-tilt", default=None,
                   help="take the reference retarder from this tilt for all tilts")
    c.add_argument("--tilts", nargs="+", help="only calibrate these tilts")
    c.add_argument("--warm-start", action="store_true",
                   help="start each column's fit from the previous solution")
    c.add_argument("--no-meta", action="store_true", help="do not store the 'meta' struct")
    c.add_argument("--plot", help="save a diagnostic figure to this path")
    c.add_argument("--plot-skip", type=int, default=10,
                   help="omit the first N columns in the plot (default: 10)")
    c.set_defaults(func=cmd_calibrate)

    a = sub.add_parser("apply", help="apply a calibration to a full Jones volume")
    a.add_argument("file")
    a.add_argument("--calib", required=True)
    a.add_argument("--tilt", help="calibration key (default: parsed from filename)")
    a.add_argument("-o", "--output", required=True, help="output HDF5 file")
    a.add_argument("--chunk", type=int, default=200, help="I-slices per read (default: 200)")
    a.add_argument("--float64", action="store_true", help="store float64 (default float32)")
    a.add_argument("--compress", action="store_true", help="gzip the output datasets")
    add_jones_opts(a)
    a.set_defaults(func=cmd_apply)

    r = sub.add_parser("run", help="extract + calibrate all tilts from a YAML config")
    r.add_argument("config")
    r.add_argument("--force", action="store_true",
                   help="re-extract profiles even if they are up to date")
    r.set_defaults(func=cmd_run)

    add_apply_tiles_cli(sub)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
