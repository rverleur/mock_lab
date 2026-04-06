"""State-history pipeline for scan-by-scan temperature, pressure, and CO."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from mock_lab.io.matlab_loader import DEFAULT_REFERENCE_FREQUENCY_HZ
from mock_lab.plotting.figures import plot_state_history, save_figure
from mock_lab.spectroscopy.state_estimation import (
    DEFAULT_OPTICAL_PATH_LENGTH_CM,
    StateHistory,
    build_state_history,
)


Array1D = NDArray[np.float64]


@dataclass(frozen=True)
class TimeHistoryPipelineResult:
    """Saved outputs from the scan-by-scan state-history reduction."""

    state_history: StateHistory
    optical_path_length_cm: float


def _write_state_history_csv(path: Path, state_history: StateHistory) -> None:
    """Write the state-history arrays as a CSV table."""

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "scan_index",
                "scan_time_s",
                "temperature_k",
                "pressure_atm",
                "co_mole_fraction",
            ]
        )

        for row in zip(
            state_history.scan_index,
            state_history.scan_time_s,
            state_history.temperature_k,
            state_history.pressure_atm,
            state_history.co_mole_fraction,
        ):
            writer.writerow([f"{value:.8g}" for value in row])


def run_time_history_pipeline(
    voigt_fit_data: Path | str,
    output_dir: Path | str,
    *,
    figure_output_dir: Path | str | None = None,
    optical_path_length_cm: float = DEFAULT_OPTICAL_PATH_LENGTH_CM,
) -> TimeHistoryPipelineResult:
    """Convert the per-scan Voigt fits into state-history plots and tables."""

    voigt_fit_data = Path(voigt_fit_data)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_output_dir = output_dir if figure_output_dir is None else Path(figure_output_dir)
    figure_output_dir.mkdir(parents=True, exist_ok=True)

    with np.load(voigt_fit_data) as data:
        temperature_k = np.asarray(data["temperature_k"], dtype=float)
        mean_apparent_pressure_atm = np.asarray(data["mean_apparent_pressure_atm"], dtype=float)
        strongest_line_area_cm_inv = np.asarray(data["line_areas"][:, 0], dtype=float)

    state_history = build_state_history(
        temperature_k,
        mean_apparent_pressure_atm,
        strongest_line_area_cm_inv,
        sweep_frequency_hz=DEFAULT_REFERENCE_FREQUENCY_HZ,
        optical_path_length_cm=optical_path_length_cm,
    )

    np.savez(
        output_dir / "state_history.npz",
        scan_index=state_history.scan_index,
        scan_time_s=state_history.scan_time_s,
        temperature_k=state_history.temperature_k,
        pressure_atm=state_history.pressure_atm,
        co_mole_fraction=state_history.co_mole_fraction,
        optical_path_length_cm=optical_path_length_cm,
    )
    _write_state_history_csv(output_dir / "state_history.csv", state_history)

    figure = plot_state_history(
        state_history.scan_index,
        state_history.temperature_k,
        state_history.pressure_atm,
        state_history.co_mole_fraction,
    )
    save_figure(figure, figure_output_dir / "state_history.png")
    plt.close(figure)

    return TimeHistoryPipelineResult(
        state_history=state_history,
        optical_path_length_cm=optical_path_length_cm,
    )
