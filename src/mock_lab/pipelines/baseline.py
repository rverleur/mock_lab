"""Pipeline for building the reusable average baseline sweep products."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from mock_lab.io.matlab_loader import load_mat_trace, split_trace
from mock_lab.plotting.figures import plot_time_overlay, save_figure
from mock_lab.plotting.mpl import plt
from mock_lab.spectroscopy.absorbance import (
    average_sweeps,
    fit_edge_line,
    subtract_edge_line,
)


Array1D = NDArray[np.float64]


@dataclass(frozen=True)
class BaselinePipelineResult:
    """Saved products from the baseline-averaging stage."""

    time_axis_s: Array1D
    average_baseline_sweep: Array1D
    baseline_line: Array1D
    corrected_baseline_sweep: Array1D
    sweep_count: int
    sample_period_s: float
    phase_start_s: float


def load_baseline_products(path: Path | str) -> BaselinePipelineResult:
    """Load a saved baseline-average product bundle from disk."""

    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(f"Baseline product file not found: {path}")

    with np.load(path) as data:
        return BaselinePipelineResult(
            time_axis_s=np.asarray(data["time_axis_s"], dtype=float),
            average_baseline_sweep=np.asarray(data["average_baseline_sweep"], dtype=float),
            baseline_line=np.asarray(data["baseline_line"], dtype=float),
            corrected_baseline_sweep=np.asarray(data["corrected_baseline_sweep"], dtype=float),
            sweep_count=int(np.asarray(data["sweep_count"]).item()),
            sample_period_s=float(np.asarray(data["sample_period_s"]).item()),
            phase_start_s=float(np.asarray(data["phase_start_s"]).item()),
        )


def run_baseline_pipeline(
    raw_data: Path | str,
    output_dir: Path | str,
    *,
    figure_output_dir: Path | str | None = None,
) -> BaselinePipelineResult:
    """Average the no-absorption baseline sweeps and save reusable corrections."""

    raw_data = Path(raw_data)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_output_dir = None if figure_output_dir is None else Path(figure_output_dir)

    trace = load_mat_trace(raw_data)
    sweeps = split_trace(trace)
    average_baseline_sweep = average_sweeps(sweeps.signal)
    baseline_line = fit_edge_line(average_baseline_sweep, edge_samples=32)
    corrected_baseline_sweep = subtract_edge_line(average_baseline_sweep, edge_samples=32)

    result = BaselinePipelineResult(
        time_axis_s=np.asarray(sweeps.time_axis_s, dtype=float),
        average_baseline_sweep=np.asarray(average_baseline_sweep, dtype=float),
        baseline_line=np.asarray(baseline_line, dtype=float),
        corrected_baseline_sweep=np.asarray(corrected_baseline_sweep, dtype=float),
        sweep_count=int(sweeps.signal.shape[0]),
        sample_period_s=float(sweeps.sample_period_s),
        phase_start_s=float(sweeps.phase_start_s),
    )

    np.savez(
        output_dir / "baseline_average.npz",
        time_axis_s=result.time_axis_s,
        average_baseline_sweep=result.average_baseline_sweep,
        baseline_line=result.baseline_line,
        corrected_baseline_sweep=result.corrected_baseline_sweep,
        sweep_count=result.sweep_count,
        sample_period_s=result.sample_period_s,
        phase_start_s=result.phase_start_s,
    )

    if figure_output_dir is not None:
        figure_output_dir.mkdir(parents=True, exist_ok=True)
        display_time_axis_s = result.time_axis_s - result.time_axis_s[0]
        figure = plot_time_overlay(
            display_time_axis_s,
            result.average_baseline_sweep,
            result.baseline_line,
            first_label="Average baseline",
            second_label="Edge-line fit",
            ylabel="Detector Signal [V]",
        )
        save_figure(figure, figure_output_dir / "baseline_average.png")
        plt.close(figure)

    return result
