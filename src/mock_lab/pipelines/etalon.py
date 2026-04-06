"""Pipeline for fitting relative wavenumber as a function of sweep time."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from mock_lab.io.matlab_loader import load_mat_trace, split_trace
from mock_lab.plotting.figures import (
    plot_etalon_calibration,
    plot_single_sweep,
    save_figure,
)
from mock_lab.spectroscopy.etalon_calibration import (
    EtalonFit,
    build_representative_sweep,
    find_etalon_peaks,
    fit_relative_wavenumber,
    remove_edge_baseline,
    save_etalon_fit,
)


Array2D = NDArray[np.float64]


@dataclass(frozen=True)
class EtalonPipelineResult:
    """Outputs from the etalon calibration pipeline."""

    fit: EtalonFit
    corrected_sweeps: Array2D
    representative_sweep: NDArray[np.float64]
    plot_sweep_index: int


def _ordinal_label(value: int) -> str:
    """Format a small positive integer as an ordinal label."""

    if 10 <= value % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(value % 10, "th")

    return f"{value}{suffix}"


def run_etalon_pipeline(
    raw_data: Path | str,
    output_dir: Path | str,
    plot_sweep_index: int = 0,
    figure_output_dir: Path | str | None = None,
    polynomial_order: int = 4,
) -> EtalonPipelineResult:
    """Fit a polynomial relative-wavenumber calibration from etalon data.

    The calibration is built from a representative sweep obtained by averaging
    all phase-aligned etalon ramps after removing their edge baseline.
    """

    raw_data = Path(raw_data)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_output_dir = output_dir if figure_output_dir is None else Path(figure_output_dir)
    figure_output_dir.mkdir(parents=True, exist_ok=True)

    trace = load_mat_trace(raw_data)
    sweeps = split_trace(trace)
    corrected_sweeps = remove_edge_baseline(sweeps.signal)
    representative_sweep = build_representative_sweep(corrected_sweeps)
    peak_indices = find_etalon_peaks(representative_sweep)
    fit = fit_relative_wavenumber(
        sweeps.time_axis_s,
        peak_indices,
        polynomial_order=polynomial_order,
    )
    display_time_axis_s = sweeps.time_axis_s - sweeps.time_axis_s[0]

    clamped_index = int(np.clip(plot_sweep_index, 0, corrected_sweeps.shape[0] - 1))
    plot_sweep = corrected_sweeps[clamped_index]

    save_etalon_fit(
        output_dir / "etalon_fit.npz",
        fit=fit,
        representative_sweep=representative_sweep,
        plot_sweep=plot_sweep,
    )

    sweep_figure = plot_single_sweep(
        display_time_axis_s,
        representative_sweep,
        peak_indices=peak_indices,
        signal_label="Average etalon sweep",
        peak_label="Average etalon peaks",
    )
    save_figure(sweep_figure, figure_output_dir / "etalon_sweep.png")
    plt.close(sweep_figure)

    calibration_figure = plot_etalon_calibration(
        display_time_axis_s,
        fit.peak_indices,
        fit.peak_wavenumber_cm_inv,
        fit.relative_wavenumber_cm_inv,
        fit.peak_residual_cm_inv,
        fit_label=f"{_ordinal_label(polynomial_order)}-order fit",
    )
    save_figure(calibration_figure, figure_output_dir / "etalon_calibration.png")
    plt.close(calibration_figure)

    return EtalonPipelineResult(
        fit=fit,
        corrected_sweeps=corrected_sweeps,
        representative_sweep=representative_sweep,
        plot_sweep_index=clamped_index,
    )
