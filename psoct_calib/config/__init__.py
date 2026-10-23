"""Bundled configuration templates and utilities."""

import sys
from pathlib import Path

if sys.version_info >= (3, 9):
    from importlib.resources import files
else:
    from importlib_resources import files  # pragma: no cover


def get_config_template_path(name: str = "calib_config.yaml") -> Path:
    """Return the Path to a bundled configuration template."""
    return files("psoct_calib.config").joinpath(name)


def get_config_template(name: str = "calib_config.yaml") -> str:
    """Return the string content of a bundled configuration template."""
    template_path = get_config_template_path(name)
    return template_path.read_text(encoding="utf-8")
