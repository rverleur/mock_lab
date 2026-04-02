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
    plot_sweep_index: int


def run_etalon_pipeline(
    raw_data: Path | str,
    output_dir: Path | str,
    plot_sweep_index: int = 0,
    figure_output_dir: Path | str | None = None,
    polynomial_order: int = 2,
) -> EtalonPipelineResult:
    """Fit a 2nd-order relative-wavenumber calibration from etalon data.

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
    # The handout asks for a polynomial fit to the etalon peaks but does not
    # specify the order, and the bundled MATLAB example code does not include
    # etalon calibration. Keep the current second-order fit for now.
    fit = fit_relative_wavenumber(
        sweeps.time_axis_s,
        peak_indices,
        polynomial_order=polynomial_order,
    )

    clamped_index = int(np.clip(plot_sweep_index, 0, corrected_sweeps.shape[0] - 1))
    plot_sweep = corrected_sweeps[clamped_index]

    save_etalon_fit(
        output_dir / "etalon_fit.npz",
        fit=fit,
        representative_sweep=representative_sweep,
        plot_sweep=plot_sweep,
    )

    sweep_figure = plot_single_sweep(
        sweeps.time_axis_s,
        plot_sweep,
        peak_indices=peak_indices,
        signal_label="Etalon sweep",
    )
    save_figure(sweep_figure, figure_output_dir / "etalon_sweep.png")
    plt.close(sweep_figure)

    calibration_figure = plot_etalon_calibration(
        sweeps.time_axis_s,
        fit.peak_indices,
        fit.peak_wavenumber_cm_inv,
        fit.relative_wavenumber_cm_inv,
    )
    save_figure(calibration_figure, figure_output_dir / "etalon_calibration.png")
    plt.close(calibration_figure)

    return EtalonPipelineResult(
        fit=fit,
        corrected_sweeps=corrected_sweeps,
        plot_sweep_index=clamped_index,
    )
