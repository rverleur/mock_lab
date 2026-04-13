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
    component_centers_cm_inv: Array1D | None = None,
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

    component_colors = ("tab:blue", "tab:green", "tab:orange", "tab:purple", "tab:brown")

    for component_index, (component_signal, component_label) in enumerate(
        zip(component_absorbance, component_labels)
    ):
        color = component_colors[component_index % len(component_colors)]
        ax.plot(
            relative_wavenumber_cm_inv,
            component_signal,
            color=color,
            linewidth=1.2,
            label=component_label,
        )
        if component_centers_cm_inv is not None and component_index < len(component_centers_cm_inv):
            ax.axvline(
                float(component_centers_cm_inv[component_index]),
                color=color,
                linewidth=1.0,
                linestyle=":",
                alpha=0.9,
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
    x_values: Array1D,
    temperature_k: Array1D,
    pressure_atm: Array1D,
    co_mole_fraction: Array1D,
    *,
    temperature_lower: Array1D | None = None,
    temperature_upper: Array1D | None = None,
    pressure_lower: Array1D | None = None,
    pressure_upper: Array1D | None = None,
    co_mole_fraction_lower: Array1D | None = None,
    co_mole_fraction_upper: Array1D | None = None,
    uncertainty_label: str = "95% CI",
    xlabel: str = r"Time [$\mu$s]",
    y_limit_coverage: float | None = None,
    y_limit_padding_fraction: float = 0.05,
    temperature_ylim: tuple[float, float] | None = None,
    pressure_ylim: tuple[float, float] | None = None,
    co_mole_fraction_percent_ylim: tuple[float, float] | None = None,
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

    if temperature_lower is not None and temperature_upper is not None:
        ax_temperature.fill_between(
            x_values,
            temperature_lower,
            temperature_upper,
            color="tab:red",
            alpha=0.18,
            linewidth=0.0,
            label=uncertainty_label,
        )
    ax_temperature.plot(x_values, temperature_k, color="tab:red", linewidth=1.4)
    ax_temperature.set_ylabel("Temperature [K]")
    if temperature_lower is not None and temperature_upper is not None:
        ax_temperature.legend(frameon=False, loc="best")

    if pressure_lower is not None and pressure_upper is not None:
        ax_pressure.fill_between(
            x_values,
            pressure_lower,
            pressure_upper,
            color="tab:blue",
            alpha=0.18,
            linewidth=0.0,
        )
    ax_pressure.plot(x_values, pressure_atm, color="tab:blue", linewidth=1.4)
    ax_pressure.set_ylabel("Pressure [atm]")

    if co_mole_fraction_lower is not None and co_mole_fraction_upper is not None:
        ax_mole_fraction.fill_between(
            x_values,
            100.0 * co_mole_fraction_lower,
            100.0 * co_mole_fraction_upper,
            color="tab:green",
            alpha=0.18,
            linewidth=0.0,
        )
    ax_mole_fraction.plot(
        x_values,
        100.0 * co_mole_fraction,
        color="tab:green",
        linewidth=1.4,
    )
    ax_mole_fraction.set_ylabel("CO Mole Fraction [%]")
    ax_mole_fraction.set_xlabel(xlabel)

    if y_limit_coverage is not None:
        if not (0.0 < y_limit_coverage <= 1.0):
            raise ValueError("y_limit_coverage must lie in (0, 1].")

        def apply_percentile_limits(
            axis: plt.Axes,
            *series: Array1D | None,
        ) -> None:
            """Set percentile-based y limits from the finite plotted values."""

            finite_values = [
                np.asarray(values, dtype=float)[np.isfinite(np.asarray(values, dtype=float))]
                for values in series
                if values is not None
            ]
            finite_values = [values for values in finite_values if values.size > 0]

            if not finite_values:
                return

            combined = np.concatenate(finite_values)
            lower_tail = 50.0 * (1.0 - y_limit_coverage)
            upper_tail = 100.0 - lower_tail
            lower_value = float(np.nanpercentile(combined, lower_tail))
            upper_value = float(np.nanpercentile(combined, upper_tail))

            if not (np.isfinite(lower_value) and np.isfinite(upper_value)):
                return

            if np.isclose(lower_value, upper_value):
                center = lower_value
                span = max(abs(center) * y_limit_padding_fraction, 1.0)
                axis.set_ylim(center - span, center + span)
                return

            span = upper_value - lower_value
            padding = max(span * y_limit_padding_fraction, 1.0e-12)
            axis.set_ylim(lower_value - padding, upper_value + padding)

        apply_percentile_limits(
            ax_temperature,
            temperature_k,
            temperature_lower,
            temperature_upper,
        )
        apply_percentile_limits(
            ax_pressure,
            pressure_atm,
            pressure_lower,
            pressure_upper,
        )
        apply_percentile_limits(
            ax_mole_fraction,
            100.0 * co_mole_fraction,
            None if co_mole_fraction_lower is None else 100.0 * co_mole_fraction_lower,
            None if co_mole_fraction_upper is None else 100.0 * co_mole_fraction_upper,
        )

    if temperature_ylim is not None:
        ax_temperature.set_ylim(*temperature_ylim)
    if pressure_ylim is not None:
        ax_pressure.set_ylim(*pressure_ylim)
    if co_mole_fraction_percent_ylim is not None:
        ax_mole_fraction.set_ylim(*co_mole_fraction_percent_ylim)

    return fig
