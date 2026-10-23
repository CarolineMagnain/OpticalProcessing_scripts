"""psoct_calib: per-column system calibration for PS-OCT."""

__version__ = "0.1.0"

from .batch import (
    parse_tiles,
    process_tile,
    run_tiles,
)
from .calibration import (
    apply_calibration,
    apply_calibration_jones,
    calibrate_tilts,
    extract_profile,
    fit_calibration,
    reconstruct_profile,
    reference_profile,
)
from .exceptions import (
    CalibrationError,
    InvalidJonesFileError,
    PSOCTError,
)
from .io import (
    JonesFile,
    load_calibration,
    load_profiles,
    save_calibration,
)
from .mueller import (
    linear_retarder,
    retarder_params_from_stokes,
    rot_mueller,
    stokes_from_mueller,
)
from .stokes import (
    jones_to_retardance_orientation,
    normalize_stokes,
    stokes_from_orientation_retardance,
)

__all__ = [
    "rot_mueller",
    "linear_retarder",
    "retarder_params_from_stokes",
    "stokes_from_mueller",
    "jones_to_retardance_orientation",
    "stokes_from_orientation_retardance",
    "normalize_stokes",
    "extract_profile",
    "reference_profile",
    "fit_calibration",
    "calibrate_tilts",
    "reconstruct_profile",
    "apply_calibration",
    "apply_calibration_jones",
    "process_tile",
    "run_tiles",
    "parse_tiles",
    "JonesFile",
    "load_calibration",
    "save_calibration",
    "load_profiles",
    "PSOCTError",
    "CalibrationError",
    "InvalidJonesFileError",
]
