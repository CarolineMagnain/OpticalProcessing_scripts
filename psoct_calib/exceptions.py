"""Domain exceptions for psoct_calib."""


class PSOCTError(Exception):
    """Base exception for all psoct_calib errors."""


class CalibrationError(PSOCTError, ValueError):
    """Raised when calibration fit fails or arguments are invalid."""


class InvalidJonesFileError(PSOCTError, ValueError):
    """Raised when Jones data file format, shape, or naming is invalid."""
