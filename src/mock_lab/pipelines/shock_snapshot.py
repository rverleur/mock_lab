"""Pipeline for converting shock sweeps from time to relative wavenumber."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from mock_lab.io.matlab_loader import load_mat_trace, split_trace
from mock_lab.pipelines.baseline import load_baseline_products, run_baseline_pipeline
from mock_lab.plotting.figures import (
    plot_frequency_domain_sweep,
    plot_single_sweep,
    plot_time_overlay,
    save_figure,
)
from mock_lab.plotting.mpl import plt
from mock_lab.spectroscopy.absorbance import (
    beer_lambert_absorbance,
    find_analysis_window,
    fit_edge_lines,
    scale_sweeps_to_reference_peak,
    subtract_edge_lines,
)
from mock_lab.spectroscopy.etalon_calibration import (
    evaluate_relative_wavenumber,
    load_etalon_fit,
)


Array1D = NDArray[np.float64]
Array2D = NDArray[np.float64]


@dataclass(frozen=True)
class ShockFrequencyDomainResult:
    """Shock traces expressed on a relative-wavenumber axis."""

    display_time_axis_s: Array1D
    relative_wavenumber_cm_inv: Array1D
    corrected_baseline_sweep: Array1D
    scaled_signal_sweeps: Array2D
    absorbance_sweeps: Array2D
    analysis_window: slice
    plot_sweep_index: int


def run_shock_snapshot_pipeline(
    raw_data: Path | str,
    baseline_dir: Path | str,
    etalon_dir: Path | str,
    output_dir: Path | str,
    plot_sweep_index: int = 0,
    figure_output_dir: Path | str | None = None,
) -> ShockFrequencyDomainResult:
    """Convert phase-aligned shock sweeps from time to relative wavenumber.

    The baseline trace is averaged over all available phase-aligned baseline
    sweeps. A straight line is fit through the first and last 32 samples of
    that average sweep, and the same style of wing-line subtraction is applied
    to each individual shock sweep. Each corrected shock sweep is then scaled
    so its peak matches the peak of the corrected average baseline before the
    Beer-Lambert absorbance is evaluated over the user-selected voltage range.
    """

    raw_data = Path(raw_data)
    baseline_dir = Path(baseline_dir)
    etalon_dir = Path(etalon_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_output_dir = output_dir if figure_output_dir is None else Path(figure_output_dir)
    figure_output_dir.mkdir(parents=True, exist_ok=True)

    calibration = load_etalon_fit(etalon_dir / "etalon_fit.npz")
    baseline_output_path = baseline_dir / "baseline_average.npz"
    if baseline_output_path.is_file():
        baseline_result = load_baseline_products(baseline_output_path)
    else:
        baseline_result = run_baseline_pipeline(
            raw_data=raw_data.with_name("MockLabData_Baseline.mat"),
            output_dir=baseline_dir,
        )

    trace = load_mat_trace(raw_data)
    sweeps = split_trace(trace)
    shock_lines = fit_edge_lines(sweeps.signal, edge_samples=32)
    corrected_signal_sweeps = subtract_edge_lines(sweeps.signal, edge_samples=32)
    scaled_signal_sweeps, scale_factors = scale_sweeps_to_reference_peak(
        baseline_result.corrected_baseline_sweep,
        corrected_signal_sweeps,
    )
    display_time_axis_s = sweeps.time_axis_s - sweeps.time_axis_s[0]
    relative_wavenumber_cm_inv = evaluate_relative_wavenumber(
        sweeps.time_axis_s,
        calibration.coefficients,
    )

    clamped_index = int(np.clip(plot_sweep_index, 0, sweeps.signal.shape[0] - 1))
    absorbance_sweeps = np.full_like(scaled_signal_sweeps, np.nan)
    analysis_window = find_analysis_window(
        baseline_result.corrected_baseline_sweep,
        start_index=32,
        minimum_reference_signal=1.0,
        maximum_reference_signal=2.75,
    )

    for sweep_index, scaled_sweep in enumerate(scaled_signal_sweeps):
        absorbance_sweeps[sweep_index, analysis_window] = beer_lambert_absorbance(
            baseline_result.corrected_baseline_sweep[analysis_window],
            scaled_sweep[analysis_window],
        )

    selected_absorbance = absorbance_sweeps[clamped_index, analysis_window]

    np.savez(
        output_dir / "shock_frequency_domain.npz",
        time_axis_s=sweeps.time_axis_s,
        relative_wavenumber_cm_inv=relative_wavenumber_cm_inv,
        average_baseline_sweep=baseline_result.average_baseline_sweep,
        baseline_line=baseline_result.baseline_line,
        corrected_baseline_sweep=baseline_result.corrected_baseline_sweep,
        shock_lines=shock_lines,
        corrected_signal_sweeps=corrected_signal_sweeps,
        scaled_signal_sweeps=scaled_signal_sweeps,
        scale_factors=scale_factors,
        absorbance_sweeps=absorbance_sweeps,
        reference_sweeps=sweeps.reference,
        etalon_coefficients=calibration.coefficients,
    )

    overlay_figure = plot_time_overlay(
        display_time_axis_s,
        baseline_result.corrected_baseline_sweep,
        scaled_signal_sweeps[clamped_index],
        first_label="Average baseline",
        second_label="Shock sweep",
        ylabel="Line-Corrected Signal [V]",
    )
    save_figure(overlay_figure, figure_output_dir / "baseline_shock_overlay.png")
    plt.close(overlay_figure)

    absorbance_time_figure = plot_single_sweep(
        sweeps.time_axis_s[analysis_window],
        selected_absorbance,
        signal_label="Absorbance",
        ylabel="Absorbance [-]",
    )
    save_figure(absorbance_time_figure, figure_output_dir / "shock_absorbance_time.png")
    plt.close(absorbance_time_figure)

    absorbance_frequency_figure = plot_frequency_domain_sweep(
        relative_wavenumber_cm_inv[analysis_window],
        selected_absorbance,
        signal_label="Absorbance",
        ylabel="Absorbance [-]",
    )
    save_figure(
        absorbance_frequency_figure,
        figure_output_dir / "shock_absorbance_frequency.png",
    )
    plt.close(absorbance_frequency_figure)

    return ShockFrequencyDomainResult(
        display_time_axis_s=display_time_axis_s,
        relative_wavenumber_cm_inv=relative_wavenumber_cm_inv,
        corrected_baseline_sweep=baseline_result.corrected_baseline_sweep,
        scaled_signal_sweeps=scaled_signal_sweeps,
        absorbance_sweeps=absorbance_sweeps,
        analysis_window=analysis_window,
        plot_sweep_index=clamped_index,
    )
