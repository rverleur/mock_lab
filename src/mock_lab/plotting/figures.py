"""Plotting utilities for report-ready figures."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from mock_lab.plotting.mpl import plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray

Array1D = NDArray[np.float64]


def _configure_report_style(
    *,
    font_size: float = 12,
    axes_labelsize: float = 12,
    legend_fontsize: float = 10,
    tick_labelsize: float = 11,
    figure_dpi: float = 160,
    savefig_dpi: float = 400,
) -> None:
    """Apply a consistent report-friendly plotting style."""

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": font_size,
            "axes.labelsize": axes_labelsize,
            "axes.titlesize": axes_labelsize,
            "legend.fontsize": legend_fontsize,
            "xtick.labelsize": tick_labelsize,
            "ytick.labelsize": tick_labelsize,
            "figure.dpi": figure_dpi,
            "savefig.dpi": savefig_dpi,
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


def plot_etalon_report_figure(
    time_s: Array1D,
    representative_sweep: Array1D,
    peak_indices: NDArray[np.int64],
    peak_wavenumber_cm_inv: Array1D,
    fitted_wavenumber_cm_inv: Array1D,
    peak_residual_cm_inv: Array1D,
    *,
    time_limits_us: tuple[float, float] = (0.5, 3.0),
    residual_limits_cm_inv: tuple[float, float] = (-1.0e-3, 1.0e-3),
) -> Figure:
    """Create the combined etalon report figure with sweep and calibration."""

    _configure_report_style(
        font_size=9,
        axes_labelsize=9,
        legend_fontsize=9,
        tick_labelsize=9,
        figure_dpi=160,
        savefig_dpi=400,
    )
    main_curve_color = "black"
    peak_color = "#c44e52"

    fig = plt.figure(figsize=(5.9, 2.45), constrained_layout=True)
    grid = fig.add_gridspec(
        2,
        2,
        width_ratios=(1.0, 1.05),
        height_ratios=(4.0, 1.1),
    )
    sweep_ax = fig.add_subplot(grid[:, 0])
    calibration_ax = fig.add_subplot(grid[0, 1])
    residual_ax = fig.add_subplot(grid[1, 1], sharex=calibration_ax)

    time_us = 1.0e6 * time_s
    for axis in (sweep_ax, calibration_ax, residual_ax):
        axis.grid(False)

    sweep_ax.plot(
        time_us,
        representative_sweep,
        color=main_curve_color,
        linewidth=0.9,
        label="etalon sweep",
    )
    sweep_ax.scatter(
        time_us[peak_indices],
        representative_sweep[peak_indices],
        s=8,
        facecolors="none",
        edgecolors=peak_color,
        linewidths=0.7,
        marker="o",
        zorder=3,
        label="peaks",
    )
    sweep_ax.set_xlabel("time [µs]")
    sweep_ax.set_ylabel("etalon signal [V]")
    sweep_ax.legend(frameon=False, loc="best", handlelength=2.0)

    calibration_ax.plot(
        time_us,
        fitted_wavenumber_cm_inv,
        color=main_curve_color,
        linewidth=0.9,
        label="quartic fit",
    )
    calibration_ax.scatter(
        time_us[peak_indices],
        peak_wavenumber_cm_inv,
        s=8,
        facecolors="none",
        edgecolors=peak_color,
        linewidths=0.7,
        marker="o",
        zorder=3,
        label="peaks",
    )
    calibration_ax.set_ylabel(r"relative wavenumber [cm$^{-1}$]")
    calibration_ax.set_xlim(*time_limits_us)
    calibration_ax.legend(frameon=False, loc="best", handlelength=2.0)
    calibration_ax.tick_params(labelbottom=False)

    residual_ax.scatter(
        time_us[peak_indices],
        peak_residual_cm_inv,
        s=7,
        facecolors="none",
        edgecolors=peak_color,
        linewidths=0.7,
        marker="o",
        zorder=3,
    )
    residual_ax.set_xlabel("time [µs]")
    residual_ax.set_ylabel("residual")
    residual_ax.set_ylim(*residual_limits_cm_inv)
    residual_ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, -3), useMathText=False)
    residual_ax.yaxis.get_offset_text().set_size(8)

    return fig


def plot_shock_time_report_figure(
    display_time_s: Array1D,
    baseline_signal: Array1D,
    shock_signal: Array1D,
    absorbance_time_s: Array1D,
    absorbance: Array1D,
) -> Figure:
    """Create the combined baseline/absorbance time-domain report figure."""

    _configure_report_style(
        font_size=9,
        axes_labelsize=9,
        legend_fontsize=9,
        tick_labelsize=9,
        figure_dpi=160,
        savefig_dpi=400,
    )

    fig = plt.figure(figsize=(5.9, 2.15), constrained_layout=True)
    grid = fig.add_gridspec(1, 2, width_ratios=(1.0, 1.0))
    overlay_ax = fig.add_subplot(grid[0, 0])
    absorbance_ax = fig.add_subplot(grid[0, 1])

    display_time_us = 1.0e6 * display_time_s
    absorbance_time_us = 1.0e6 * absorbance_time_s

    for axis in (overlay_ax, absorbance_ax):
        axis.grid(False)

    overlay_ax.plot(
        display_time_us,
        baseline_signal,
        color="black",
        linewidth=0.9,
        label="baseline",
    )
    overlay_ax.plot(
        display_time_us,
        shock_signal,
        color="#c44e52",
        linewidth=0.9,
        label="shock sweep",
    )
    overlay_ax.set_xlabel("time [µs]")
    overlay_ax.set_ylabel("line-corrected signal [V]")
    overlay_ax.legend(frameon=False, loc="best", handlelength=2.0)

    absorbance_ax.plot(
        absorbance_time_us,
        absorbance,
        color="black",
        linewidth=0.9,
    )
    absorbance_ax.set_xlabel("time [µs]")
    absorbance_ax.set_ylabel("absorbance [-]")
    absorbance_ax.set_xlim(float(absorbance_time_us[0]), float(absorbance_time_us[-1]))

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
    report_style: bool = False,
) -> Figure:
    """Plot measured absorbance, the total Voigt fit, and residuals."""

    if report_style:
        _configure_report_style(
            font_size=9,
            axes_labelsize=9,
            legend_fontsize=9,
            tick_labelsize=9,
            figure_dpi=160,
            savefig_dpi=400,
        )
        figsize = (4.0, 2.44)
        top_linewidth = 1.0
        fit_linewidth = 1.0
        component_linewidth = 0.9
        center_linewidth = 0.8
        residual_linewidth = 0.9
        measured_color = "black"
        fit_color = "#c44e52"
        residual_color = "black"
        component_colors = ("#1f77b4", "#2ca02c", "#ff7f0e", "#d627c1", "#9467bd")
        measured_label = "measured"
        fit_label = "fit"
    else:
        _configure_report_style()
        figsize = (7.0, 5.0)
        top_linewidth = 1.8
        fit_linewidth = 1.8
        component_linewidth = 1.2
        center_linewidth = 1.0
        residual_linewidth = 1.4
        measured_color = "black"
        fit_color = "tab:red"
        residual_color = "tab:blue"
        component_colors = ("tab:blue", "tab:green", "tab:orange", "tab:purple", "tab:brown")
        measured_label = "Measured absorbance"
        fit_label = "Voigt fit"

    fig, (ax, residual_ax) = plt.subplots(
        2,
        1,
        figsize=figsize,
        constrained_layout=True,
        sharex=True,
        gridspec_kw={"height_ratios": [4.0, 1.2]},
    )
    if report_style:
        ax.grid(False)
        residual_ax.grid(False)
    ax.plot(
        relative_wavenumber_cm_inv,
        absorbance,
        color=measured_color,
        linewidth=top_linewidth,
        label=measured_label,
    )
    ax.plot(
        relative_wavenumber_cm_inv,
        fitted_absorbance,
        color=fit_color,
        linewidth=fit_linewidth,
        linestyle="--",
        label=fit_label,
    )

    for component_index, (component_signal, component_label) in enumerate(
        zip(component_absorbance, component_labels)
    ):
        color = component_colors[component_index % len(component_colors)]
        ax.plot(
            relative_wavenumber_cm_inv,
            component_signal,
            color=color,
            linewidth=component_linewidth,
            label=component_label,
        )
        if component_centers_cm_inv is not None and component_index < len(component_centers_cm_inv):
            ax.axvline(
                float(component_centers_cm_inv[component_index]),
                color=color,
                linewidth=center_linewidth,
                linestyle=":",
                alpha=0.9,
            )

    ax.set_ylabel("absorbance [-]" if report_style else "Absorbance [-]")
    if absorbance_limits is not None:
        ax.set_ylim(*absorbance_limits)
    ax.legend(
        frameon=False,
        ncol=1 if report_style else 2,
        handlelength=2.0,
        loc="upper left" if report_style else "best",
    )
    if annotation_text is not None:
        ax.text(
            0.985,
            0.97,
            annotation_text,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8 if report_style else None,
            bbox={"facecolor": "white", "edgecolor": "0.75", "alpha": 0.9},
        )

    if not report_style:
        residual_ax.axhline(0.0, linewidth=1.0, color="black")
    residual_ax.plot(
        relative_wavenumber_cm_inv,
        absorbance - fitted_absorbance,
        color=residual_color,
        linewidth=residual_linewidth,
    )
    residual_ax.set_xlabel(
        r"relative wavenumber [cm$^{-1}$]" if report_style else r"Relative Wavenumber [cm$^{-1}$]"
    )
    residual_ax.set_ylabel("residual" if report_style else "Residual [-]")
    if report_style:
        plot_residual_limits = residual_limits if residual_limits is not None else (-25.0e-3, 25.0e-3)
        residual_ax.set_ylim(*plot_residual_limits)
        residual_ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, -3), useMathText=False)
        residual_ax.yaxis.get_offset_text().set_size(8)
    elif residual_limits is not None:
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
    co_mole_fraction_ylim: tuple[float, float] | None = None,
    co_mole_fraction_percent_ylim: tuple[float, float] | None = None,
    co_mole_fraction_scale: float = 100.0,
    co_mole_fraction_label: str | None = None,
    xlim: tuple[float, float] | None = None,
    report_style: bool = False,
) -> Figure:
    """Plot scan-by-scan temperature, pressure, and CO mole fraction."""

    if report_style:
        _configure_report_style(
            font_size=9,
            axes_labelsize=9,
            legend_fontsize=9,
            tick_labelsize=9,
            figure_dpi=160,
            savefig_dpi=400,
        )
        figsize = (4.0, 2.92)
        line_width = 1.0
        fill_alpha = 0.16
        temperature_color = "#c44e52"
        pressure_color = "#4c72b0"
        mole_fraction_color = "#55a868"
        if co_mole_fraction_label is None:
            co_mole_fraction_label = "CO mole fraction [-]"
    else:
        _configure_report_style()
        figsize = (7.0, 7.2)
        line_width = 1.4
        fill_alpha = 0.18
        temperature_color = "tab:red"
        pressure_color = "tab:blue"
        mole_fraction_color = "tab:green"
        if co_mole_fraction_label is None:
            co_mole_fraction_label = "CO Mole Fraction [%]" if co_mole_fraction_scale == 100.0 else "CO Mole Fraction [-]"

    fig, axes = plt.subplots(
        3,
        1,
        figsize=figsize,
        constrained_layout=True,
        sharex=True,
    )
    ax_temperature, ax_pressure, ax_mole_fraction = axes
    if report_style:
        for axis in axes:
            axis.grid(False)

    if temperature_lower is not None and temperature_upper is not None:
        ax_temperature.fill_between(
            x_values,
            temperature_lower,
            temperature_upper,
            color=temperature_color,
            alpha=fill_alpha,
            linewidth=0.0,
            label=uncertainty_label,
        )
    ax_temperature.plot(x_values, temperature_k, color=temperature_color, linewidth=line_width)
    ax_temperature.set_ylabel("temperature\n[K]" if report_style else "Temperature [K]")
    if temperature_lower is not None and temperature_upper is not None:
        ax_temperature.legend(frameon=False, loc="upper right" if report_style else "best")

    if pressure_lower is not None and pressure_upper is not None:
        ax_pressure.fill_between(
            x_values,
            pressure_lower,
            pressure_upper,
            color=pressure_color,
            alpha=fill_alpha,
            linewidth=0.0,
        )
    ax_pressure.plot(x_values, pressure_atm, color=pressure_color, linewidth=line_width)
    ax_pressure.set_ylabel("pressure\n[atm]" if report_style else "Pressure [atm]")

    if co_mole_fraction_lower is not None and co_mole_fraction_upper is not None:
        ax_mole_fraction.fill_between(
            x_values,
            co_mole_fraction_scale * co_mole_fraction_lower,
            co_mole_fraction_scale * co_mole_fraction_upper,
            color=mole_fraction_color,
            alpha=fill_alpha,
            linewidth=0.0,
        )
    ax_mole_fraction.plot(
        x_values,
        co_mole_fraction_scale * co_mole_fraction,
        color=mole_fraction_color,
        linewidth=line_width,
    )
    ax_mole_fraction.set_ylabel(
        co_mole_fraction_label.replace(" mole fraction ", " mole\nfraction ")
        if report_style
        else co_mole_fraction_label
    )
    ax_mole_fraction.set_xlabel(xlabel)

    if report_style:
        for axis in axes:
            axis.yaxis.set_label_coords(-0.18, 0.5)

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
            co_mole_fraction_scale * co_mole_fraction,
            None if co_mole_fraction_lower is None else co_mole_fraction_scale * co_mole_fraction_lower,
            None if co_mole_fraction_upper is None else co_mole_fraction_scale * co_mole_fraction_upper,
        )

    if temperature_ylim is not None:
        ax_temperature.set_ylim(*temperature_ylim)
    if pressure_ylim is not None:
        ax_pressure.set_ylim(*pressure_ylim)
    if co_mole_fraction_ylim is not None:
        ax_mole_fraction.set_ylim(*co_mole_fraction_ylim)
    elif co_mole_fraction_percent_ylim is not None:
        ax_mole_fraction.set_ylim(*co_mole_fraction_percent_ylim)
    if xlim is not None:
        ax_mole_fraction.set_xlim(*xlim)

    return fig
