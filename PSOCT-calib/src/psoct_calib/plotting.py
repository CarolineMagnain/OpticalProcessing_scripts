"""Diagnostic figures: smoothed measured profile vs. calibrated reconstruction."""

import numpy as np

from .calibration import reconstruct_profile

_COLORS = {"Q": "tab:red", "U": "tab:blue", "V": "tab:green"}


def plot_profile_fit(ax, measured, reconstructed, skip=10, title=None):
    """Solid lines: smoothed measured profile. Dash-dot: reconstruction."""
    x = np.arange(skip, len(measured[0]))
    for name, m, r in zip("QUV", measured, reconstructed):
        ax.plot(x, m[skip:], color=_COLORS[name], label=name)
        ax.plot(x, r[skip:], "-.", color=_COLORS[name], label=f"{name} reconstructed")
    ax.set_xlabel("column index j")
    ax.set_ylim(-1.05, 1.05)
    if title:
        ax.set_title(title)


def plot_calibrations(results, profiles, path, skip=10, dpi=150):
    """One panel per tilt, saved to ``path``."""
    import matplotlib.pyplot as plt

    tilts = list(results)
    fig, axes = plt.subplots(1, len(tilts), figsize=(4.5 * len(tilts), 3.6),
                             sharey=True, squeeze=False)
    for ax, tilt in zip(axes[0], tilts):
        p = profiles[tilt]
        rec = reconstruct_profile(p["Qa"], p["Ua"], p["Va"], results[tilt].calibration_M)
        plot_profile_fit(ax, results[tilt].smoothed, rec, skip=skip, title=tilt)
    axes[0, 0].set_ylabel("normalized Stokes")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
