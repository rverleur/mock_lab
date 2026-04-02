"""Plotting utilities for report-ready figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray


Array1D = NDArray[np.float64]


def _configure_report_style() -> None:
    """Apply a consistent report-friendly plotting style."""

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 12,
            "axes.labelsize": 12,
            "axes.titlesize": 12,
            "legend.fontsize": 10,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "figure.dpi": 160,
            "savefig.dpi": 400,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
            "axes.spines.top": True,
            "axes.spines.right": True,
        }
    )


def _new_figure() -> tuple[Figure, Axes]:
    """Create a figure after the standard style has been applied."""

    _configure_report_style()
    return plt.subplots(figsize=(7.0, 4.25), constrained_layout=True)


def save_figure(fig: Figure, path: Path | str) -> None:
    """Save a figure with a consistent high-resolution export."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")


def plot_single_sweep(
    time_s: Array1D,
    signal: Array1D,
    *,
    peak_indices: NDArray[np.int64] | None = None,
    signal_label: str = "Signal",
    ylabel: str = "Detector Signal [V]",
    peak_label: str = "Detected Peaks",
) -> Figure:
    """Plot one time-domain sweep with optional peak markers."""

    fig, ax = _new_figure()
    time_us = 1.0e6 * time_s
    ax.plot(time_us, signal, linewidth=1.8, label=signal_label)

    if peak_indices is not None and peak_indices.size > 0:
        ax.scatter(
            time_us[peak_indices],
            signal[peak_indices],
            s=28,
            c="tab:red",
            label=peak_label,
            zorder=3,
        )

    ax.set_xlabel(r"Time [$\mu$s]")
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False)
    return fig


def plot_time_overlay(
    time_s: Array1D,
    first_signal: Array1D,
    second_signal: Array1D,
    *,
    first_label: str,
    second_label: str,
    ylabel: str = "Detector Signal [V]",
) -> Figure:
    """Plot two time-domain traces on the same axes."""

    fig, ax = _new_figure()
    time_us = 1.0e6 * time_s
    ax.plot(time_us, first_signal, linewidth=1.8, label=first_label)
    ax.plot(time_us, second_signal, linewidth=1.8, label=second_label)
    ax.set_xlabel(r"Time [$\mu$s]")
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False)
    return fig


def plot_etalon_calibration(
    time_s: Array1D,
    peak_indices: NDArray[np.int64],
    peak_wavenumber_cm_inv: Array1D,
    fitted_wavenumber_cm_inv: Array1D,
) -> Figure:
    """Plot detected etalon peak locations and the polynomial calibration."""

    fig, ax = _new_figure()
    time_us = 1.0e6 * time_s
    ax.plot(
        time_us,
        fitted_wavenumber_cm_inv,
        linewidth=1.8,
        color="black",
        label="2nd-order fit",
    )
    ax.scatter(
        time_us[peak_indices],
        peak_wavenumber_cm_inv,
        s=28,
        c="tab:red",
        label="Etalon peaks",
        zorder=3,
    )
    ax.set_xlabel(r"Time [$\mu$s]")
    ax.set_ylabel(r"Relative Wavenumber [cm$^{-1}$]")
    ax.legend(frameon=False)
    return fig


def plot_frequency_domain_sweep(
    relative_wavenumber_cm_inv: Array1D,
    signal: Array1D,
    *,
    signal_label: str = "Signal",
    ylabel: str = "Detector Signal [V]",
) -> Figure:
    """Plot one sweep after converting the horizontal axis to wavenumber."""

    fig, ax = _new_figure()
    ax.plot(relative_wavenumber_cm_inv, signal, linewidth=1.8, label=signal_label)
    ax.set_xlabel(r"Relative Wavenumber [cm$^{-1}$]")
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False)
    return fig
