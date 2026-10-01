# psoct-calib

Per A-line system calibration for polarization-sensitive OCT (PS-OCT).

A reference region of PSOCT measurement is averaged into a Stokes
profile along the surface of the sample. For each depth profile, the system is modelled as
a linear retarder traversed twice around the sample, and its orientation and
retardance are fitted by least squares. Calibrations for several sample tilts
are stored in a single small `.mat` file.

## Install

```bash
git clone https://github.com/CarolineMagnain/OpticalProcessing_scripts.git
cd OCTAnalysis/psoct-calib
pip install -e ".[test]"
```

Requires Python ≥ 3.9, numpy, scipy, h5py, matplotlib, pyyaml, tqdm.

## Quick start

The simplest route is one config file for all tilts:

```bash
cp calib_config.example.yaml calib_config.yaml   # edit data_dir and ROIs
psoct-calib run calib_config.yaml
```

This produces:

| file | size | contents |
|---|---|---|
| `profiles.mat` | kB | per tilt: `Qa, Ua, Va` + ROI, sigma, source file |
| `calibration_profile.mat` | kB | per tilt: `(J,4,4)` calibration matrices; `meta` struct |
| `calibration_fit.png` | – | measured vs. reconstructed profile per tilt |

Re-running `run` skips any tilt whose profile is already up to date, so changing
the fit settings does not re-read the large Jones files. Use `--force` to
re-extract everything.

## Individual steps

Example session: three tilts, one Jones file each. `DATA` is the folder with
the Jones `.mat` files and `OUT` is where the small result files go.

```console
$ DATA=/path/to/Processed3D
$ OUT=/path/to/calibration_output
$ cd $DATA

$ # 1. Jones volumes -> reference profiles (heavy step; reads only the ROI + filter margin)
$ psoct-calib extract Jones_tiltNeg20deg_strip0023_PD12.mat --i-range 4000:5000 --k-range 190:240 -o $OUT/profiles.mat
[extract] tiltNeg20deg: Jones_tiltNeg20deg_strip0023_PD12.mat  volume (I,J,K)=(9000, 150, 500)
[extract] tiltNeg20deg: profile of length 150 -> /path/to/calibration_output/profiles.mat
$ psoct-calib extract Jones_normal0deg_strip0023_PD12.mat   --i-range 4000:5000 --k-range 190:240 -o $OUT/profiles.mat
[extract] normal0deg: Jones_normal0deg_strip0023_PD12.mat  volume (I,J,K)=(9000, 150, 500)
[extract] normal0deg: profile of length 150 -> /path/to/calibration_output/profiles.mat
$ psoct-calib extract Jones_tiltPos20deg_strip0023_PD12.mat --i-range 4000:5000 --k-range 190:240 -o $OUT/profiles.mat
[extract] tiltPos20deg: Jones_tiltPos20deg_strip0023_PD12.mat  volume (I,J,K)=(9000, 150, 500)
[extract] tiltPos20deg: profile of length 150 -> /path/to/calibration_output/profiles.mat

$ # 2. profiles -> calibration for every tilt (seconds; no large data needed)
$ psoct-calib calibrate $OUT/profiles.mat -o $OUT/calibration_profile.mat --plot $OUT/fit.png
[calibrate] tiltNeg20deg: J=150  ref=[93]  max cost=1.073e-17
[calibrate] tiltPos20deg: J=150  ref=[109]  max cost=1.408e-17
[calibrate] normal0deg: J=150  ref=[112]  max cost=2.631e-17
[calibrate] wrote ['normal0deg', 'tiltNeg20deg', 'tiltPos20deg'] -> /path/to/calibration_output/calibration_profile.mat
[calibrate] figure -> /path/to/calibration_output/fit.png

$ # 3. (optional) apply to a full volume, streamed slab by slab
$ psoct-calib apply Jones_tiltNeg20deg_strip0023_PD12.mat --calib $OUT/calibration_profile.mat -o $OUT/corrected.h5
```

Each `extract` call adds (or replaces) one tilt in `profiles.mat` and keeps the
others, so a single tilt can be re-extracted with a different ROI. Run the
`extract` calls one after another, not as simultaneous jobs writing to the same
`profiles.mat`. If all tilts share one ROI, a single call also works:
`psoct-calib extract Jones_*_strip0023_PD12.mat --i-range 4000:5000 --k-range 190:240 -o $OUT/profiles.mat`.

`ref=[j]` is the reference column (maximal V) chosen for each tilt; `max cost`
is the largest least-squares residual over all columns.

The tilt name is parsed from `Jones_<tilt>_...`; use `--tilt NAME` otherwise.
Run `psoct-calib <command> --help` for all options.

**Ranges are 0-based and half-open**, as in Python slicing. MATLAB `4001:5000` corresponds to `4000:5000`.

## Applying the calibration to tiled data

`apply-tiles` processes every tile of one tilt and slice, reading
`Jones_<tilt>_slice<N>_strip<TTTT>_PD12.mat` and writing four float32 `.mat`
files per tile:

| file | variable | contents |
|---|---|---|
| `Q_<tilt>_slice<N>_strip<TTTT>.mat` | `Q` | corrected Q |
| `U_<tilt>_slice<N>_strip<TTTT>.mat` | `U` | corrected U |
| `V_<tilt>_slice<N>_strip<TTTT>.mat` | `V` | corrected V |
| `dBI3D_<tilt>_slice<N>_strip<TTTT>.mat` | `inten` | `10·log10(sqrt(|J1|²+|J2|²))` |

```console
$ psoct-calib apply-tiles --tilt normal0deg --slice 3 \
      --path-in /path/to/jones_tiles --path-out /path/to/Stokes3D \
      --calib /path/to/calibration_output/calibration_profile.mat
```

Useful options:

* `--tiles 1-55` (default; inclusive), a single tile `--tiles 7`, or a list `--tiles 1,4,10-12`.
* `--skip-existing` resumes an interrupted job. Files are written under a
  temporary name and renamed when complete, so partial outputs are never left behind.
* `--skip-missing` continues past missing input tiles instead of stopping.
* `--no-intensity` skips the `dBI3D` files; `--compress` compresses the outputs.
* `--input-pattern` / `--output-pattern` adapt to other file-naming schemes.

Here the Stokes volumes are Gaussian-filtered (`--sigma-stokes`, default 1) and
normalized before the calibration is applied. Each tilt/slice is independent,
so they can run as parallel cluster jobs, e.g.

```bash
for tilt in normal0deg tiltNeg20deg tiltPos20deg; do
  for s in 1 2 3; do
    psoct-calib apply-tiles --tilt $tilt --slice $s --path-in $DATA --path-out $OUT_STOKES \
        --calib $OUT/calibration_profile.mat --skip-existing
  done
done
```

The same processing is available in Python:

```python
from psoct_calib.batch import process_tile, run_tiles

out = process_tile(J1, J2, calibration_M)          # dict: Q, U, V, inten (float32)
run_tiles(path_in, path_out, "calibration_profile.mat", "normal0deg", slice_num=3, tiles="1-55")
```

## Method and conventions

* Retardance and orientation (degrees):
  `R = atan(|J1|/|J2|)`, `O = (arg J1 − arg J2 )/2`.
* Round-trip Stokes: `Q = sin2O·sin2R`, `U = cos2O·sin2R`, `V = cos2R`,
  Gaussian-filtered (`sigma_stokes`) and normalized per voxel.
* Profile: mean over the I and K ranges of the ROI, normalized, then smoothed
  along J (`sigma_profile`) and normalized again for the fit.
* Single-pass retarder of a Stokes vector: `θ = atan2(Q,U)/2`, `δ = acos(V)/2`.
* Fit for each column: `M_meas(j) ≈ M_LR(j) · M_ref · M_LR(j)`, where `M_ref` is the
  column with maximal V (`reference: max_v`), a chosen index, or a column
  taken from another tilt (`reference_tilt`).
* Correction: `M_true = M_LR⁻¹ M_meas M_LR⁻¹`, `Mc = M_true²`,
  `(q,u,v) = (−Mc[1,3], Mc[2,3], Mc[3,3])`.

## Package layout

```
src/psoct_calib/
  mueller.py      rotation and linear-retarder Mueller matrices, Stokes <-> retarder
  stokes.py       Jones -> (R, O) -> normalized Q, U, V
  io.py           lazy v7.3 (HDF5) reader, small .mat read/write
  calibration.py  profile extraction, fit, reconstruction, volume correction
  batch.py        apply-tiles: calibration of all tiles of a tilt/slice
  plotting.py     diagnostic figure
  cli.py          command-line interface
tests/            unit tests + equivalence with the original notebook
```

## Citation
