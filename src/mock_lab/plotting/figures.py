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
    peak_residual_cm_inv: Array1D,
    fit_label: str = "Polynomial fit",
    peak_label: str = "Average etalon peaks",
    time_limits_us: tuple[float, float] = (0.5, 3.0),
    residual_limits_cm_inv: tuple[float, float] = (-1.0e-3, 1.0e-3),
) -> Figure:
    """Plot the etalon calibration with a residual panel underneath."""

    _configure_report_style()
    fig, (ax, residual_ax) = plt.subplots(
        2,
        1,
        figsize=(7.0, 5.0),
        constrained_layout=True,
        sharex=True,
        gridspec_kw={"height_ratios": [4.0, 1.1]},
    )
    time_us = 1.0e6 * time_s
    ax.plot(
        time_us,
        fitted_wavenumber_cm_inv,
        linewidth=1.8,
        color="black",
        label=fit_label,
    )
    ax.scatter(
        time_us[peak_indices],
        peak_wavenumber_cm_inv,
        s=28,
        c="tab:red",
        label=peak_label,
        zorder=3,
    )
    ax.set_ylabel(r"Relative Wavenumber [cm$^{-1}$]")
    ax.legend(frameon=False)
    ax.set_xlim(*time_limits_us)

    residual_ax.axhline(0.0, linewidth=1.2, color="black")
    residual_ax.scatter(
        time_us[peak_indices],
        peak_residual_cm_inv,
        s=24,
        c="tab:red",
        zorder=3,
    )
    residual_ax.set_xlabel(r"Time [$\mu$s]")
    residual_ax.set_ylabel(r"Residual [cm$^{-1}$]")
    residual_ax.set_ylim(*residual_limits_cm_inv)
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


def plot_voigt_fit(
    relative_wavenumber_cm_inv: Array1D,
    absorbance: Array1D,
    fitted_absorbance: Array1D,
    component_absorbance: NDArray[np.float64],
    *,
    component_labels: tuple[str, ...],
    absorbance_limits: tuple[float, float] | None = None,
    residual_limits: tuple[float, float] | None = None,
    annotation_text: str | None = None,
) -> Figure:
    """Plot measured absorbance, the total Voigt fit, and residuals."""

    _configure_report_style()
    fig, (ax, residual_ax) = plt.subplots(
        2,
        1,
        figsize=(7.0, 5.0),
        constrained_layout=True,
        sharex=True,
        gridspec_kw={"height_ratios": [4.0, 1.2]},
    )
    ax.plot(
        relative_wavenumber_cm_inv,
        absorbance,
        color="black",
        linewidth=1.8,
        label="Measured absorbance",
    )
    ax.plot(
        relative_wavenumber_cm_inv,
        fitted_absorbance,
        color="tab:red",
        linewidth=1.8,
        linestyle="--",
        label="Voigt fit",
    )

    for component_signal, component_label in zip(component_absorbance, component_labels):
        ax.plot(
            relative_wavenumber_cm_inv,
            component_signal,
            linewidth=1.2,
            label=component_label,
        )

    ax.set_ylabel("Absorbance [-]")
    if absorbance_limits is not None:
        ax.set_ylim(*absorbance_limits)
    ax.legend(frameon=False, ncol=2)
    if annotation_text is not None:
        ax.text(
            0.985,
            0.97,
            annotation_text,
            transform=ax.transAxes,
            ha="right",
            va="top",
            bbox={"facecolor": "white", "edgecolor": "0.75", "alpha": 0.9},
        )

    residual_ax.axhline(0.0, linewidth=1.0, color="black")
    residual_ax.plot(
        relative_wavenumber_cm_inv,
        absorbance - fitted_absorbance,
        color="tab:blue",
        linewidth=1.4,
    )
    residual_ax.set_xlabel(r"Relative Wavenumber [cm$^{-1}$]")
    residual_ax.set_ylabel("Residual [-]")
    if residual_limits is not None:
        residual_ax.set_ylim(*residual_limits)
    return fig


def plot_state_history(
    scan_index: Array1D,
    temperature_k: Array1D,
    pressure_atm: Array1D,
    co_mole_fraction: Array1D,
) -> Figure:
    """Plot scan-by-scan temperature, pressure, and CO mole fraction."""

    _configure_report_style()
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(7.0, 7.2),
        constrained_layout=True,
        sharex=True,
    )
    ax_temperature, ax_pressure, ax_mole_fraction = axes

    ax_temperature.plot(scan_index, temperature_k, color="tab:red", linewidth=1.4)
    ax_temperature.set_ylabel("Temperature [K]")

    ax_pressure.plot(scan_index, pressure_atm, color="tab:blue", linewidth=1.4)
    ax_pressure.set_ylabel("Pressure [atm]")

    ax_mole_fraction.plot(scan_index, co_mole_fraction, color="tab:green", linewidth=1.4)
    ax_mole_fraction.set_ylabel("CO Mole Fraction [-]")
    ax_mole_fraction.set_xlabel("Scan Index [-]")

    return fig
