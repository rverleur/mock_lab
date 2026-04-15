"""State-history pipeline for scan-by-scan temperature, pressure, and CO."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from mock_lab.io.matlab_loader import DEFAULT_PHASE_START_S, DEFAULT_REFERENCE_FREQUENCY_HZ
from mock_lab.plotting.figures import plot_state_history, save_figure
from mock_lab.plotting.mpl import plt
from mock_lab.spectroscopy.state_estimation import (
    DEFAULT_OPTICAL_PATH_LENGTH_CM,
    StateHistory,
    build_state_history,
    evaluate_state_from_fit_parameters,
)
from mock_lab.spectroscopy.voigt import (
    DEFAULT_CO_TRANSITIONS,
    expand_constrained_parameters,
    finite_difference_jacobian,
)


Array1D = NDArray[np.float64]


@dataclass(frozen=True)
class TimeHistoryPipelineResult:
    """Saved outputs from the scan-by-scan state-history reduction."""

    state_history: StateHistory
    optical_path_length_cm: float
    confidence_level: float
    temperature_ci95_lower: Array1D
    temperature_ci95_upper: Array1D
    pressure_ci95_lower: Array1D
    pressure_ci95_upper: Array1D
    co_mole_fraction_ci95_lower: Array1D
    co_mole_fraction_ci95_upper: Array1D


def _write_state_history_csv(
    path: Path,
    state_history: StateHistory,
    *,
    temperature_ci95_lower: Array1D,
    temperature_ci95_upper: Array1D,
    pressure_ci95_lower: Array1D,
    pressure_ci95_upper: Array1D,
    co_mole_fraction_ci95_lower: Array1D,
    co_mole_fraction_ci95_upper: Array1D,
) -> None:
    """Write the state-history arrays as a CSV table."""

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "scan_index",
                "scan_time_s",
                "temperature_k",
                "temperature_ci95_lower",
                "temperature_ci95_upper",
                "pressure_atm",
                "pressure_ci95_lower",
                "pressure_ci95_upper",
                "co_mole_fraction",
                "co_mole_fraction_ci95_lower",
                "co_mole_fraction_ci95_upper",
            ]
        )

        for row in zip(
            state_history.scan_index,
            state_history.scan_time_s,
            state_history.temperature_k,
            temperature_ci95_lower,
            temperature_ci95_upper,
            state_history.pressure_atm,
            pressure_ci95_lower,
            pressure_ci95_upper,
            state_history.co_mole_fraction,
            co_mole_fraction_ci95_lower,
            co_mole_fraction_ci95_upper,
        ):
            writer.writerow([f"{value:.8g}" for value in row])


def _state_confidence_intervals(
    reduced_parameter_vectors: NDArray[np.float64],
    reduced_parameter_covariance: NDArray[np.float64],
    success: NDArray[np.bool_],
    state_history: StateHistory,
    *,
    confidence_scale: float,
    optical_path_length_cm: float,
) -> tuple[Array1D, Array1D, Array1D, Array1D, Array1D, Array1D]:
    """Propagate fit-parameter covariance into state-history confidence bands."""

    scan_count = reduced_parameter_vectors.shape[0]
    temperature_lower = np.full(scan_count, np.nan, dtype=float)
    temperature_upper = np.full(scan_count, np.nan, dtype=float)
    pressure_lower = np.full(scan_count, np.nan, dtype=float)
    pressure_upper = np.full(scan_count, np.nan, dtype=float)
    co_mole_fraction_lower = np.full(scan_count, np.nan, dtype=float)
    co_mole_fraction_upper = np.full(scan_count, np.nan, dtype=float)

    if not np.isfinite(confidence_scale):
        return (
            temperature_lower,
            temperature_upper,
            pressure_lower,
            pressure_upper,
            co_mole_fraction_lower,
            co_mole_fraction_upper,
        )

    for scan_index in range(scan_count):
        if not success[scan_index]:
            continue

        reduced_vector = np.asarray(reduced_parameter_vectors[scan_index], dtype=float)
        covariance = np.asarray(reduced_parameter_covariance[scan_index], dtype=float)

        if not np.all(np.isfinite(reduced_vector)) or not np.all(np.isfinite(covariance)):
            continue

        def state_vector(parameter_vector: Array1D) -> Array1D:
            parameters = expand_constrained_parameters(
                parameter_vector,
                transitions=DEFAULT_CO_TRANSITIONS,
            )
            return evaluate_state_from_fit_parameters(
                parameters,
                optical_path_length_cm=optical_path_length_cm,
                transitions=DEFAULT_CO_TRANSITIONS,
            )

        state_jacobian = finite_difference_jacobian(state_vector, reduced_vector)
        state_covariance = state_jacobian @ covariance @ state_jacobian.T
        state_standard_error = np.sqrt(
            np.clip(np.diag(np.asarray(state_covariance, dtype=float)), a_min=0.0, a_max=None)
        )
        fit_half_width = confidence_scale * state_standard_error
        half_width = np.asarray(
            np.clip(fit_half_width, a_min=0.0, a_max=None),
            dtype=float,
        )
        nominal_state = np.array(
            [
                state_history.temperature_k[scan_index],
                state_history.pressure_atm[scan_index],
                state_history.co_mole_fraction[scan_index],
            ],
            dtype=float,
        )

        temperature_lower[scan_index] = nominal_state[0] - half_width[0]
        temperature_upper[scan_index] = nominal_state[0] + half_width[0]
        pressure_lower[scan_index] = nominal_state[1] - half_width[1]
        pressure_upper[scan_index] = nominal_state[1] + half_width[1]
        co_mole_fraction_lower[scan_index] = nominal_state[2] - half_width[2]
        co_mole_fraction_upper[scan_index] = nominal_state[2] + half_width[2]

    return (
        temperature_lower,
        temperature_upper,
        pressure_lower,
        pressure_upper,
        co_mole_fraction_lower,
        co_mole_fraction_upper,
    )


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
        collisional_hwhm_cm_inv = np.asarray(data["collisional_hwhm_cm_inv"], dtype=float)
        strongest_line_area_cm_inv = np.asarray(data["line_areas"][:, 0], dtype=float)
        success = np.asarray(data["success"], dtype=bool)
        reduced_parameter_vectors = np.asarray(data["reduced_parameter_vectors"], dtype=float)
        reduced_parameter_covariance = np.asarray(data["reduced_parameter_covariance"], dtype=float)
        confidence_level = float(np.asarray(data["confidence_level"]).item())
        confidence_scale = float(np.asarray(data["confidence_scale"]).item())

    state_history = build_state_history(
        temperature_k,
        collisional_hwhm_cm_inv,
        strongest_line_area_cm_inv,
        sweep_frequency_hz=DEFAULT_REFERENCE_FREQUENCY_HZ,
        optical_path_length_cm=optical_path_length_cm,
    )
    (
        temperature_ci95_lower,
        temperature_ci95_upper,
        pressure_ci95_lower,
        pressure_ci95_upper,
        co_mole_fraction_ci95_lower,
        co_mole_fraction_ci95_upper,
    ) = _state_confidence_intervals(
        reduced_parameter_vectors,
        reduced_parameter_covariance,
        success,
        state_history,
        confidence_scale=confidence_scale,
        optical_path_length_cm=optical_path_length_cm,
    )

    np.savez(
        output_dir / "state_history.npz",
        scan_index=state_history.scan_index,
        scan_time_s=state_history.scan_time_s,
        temperature_k=state_history.temperature_k,
        temperature_ci95_lower=temperature_ci95_lower,
        temperature_ci95_upper=temperature_ci95_upper,
        pressure_atm=state_history.pressure_atm,
        pressure_ci95_lower=pressure_ci95_lower,
        pressure_ci95_upper=pressure_ci95_upper,
        co_mole_fraction=state_history.co_mole_fraction,
        co_mole_fraction_ci95_lower=co_mole_fraction_ci95_lower,
        co_mole_fraction_ci95_upper=co_mole_fraction_ci95_upper,
        confidence_level=confidence_level,
        optical_path_length_cm=optical_path_length_cm,
    )
    _write_state_history_csv(
        output_dir / "state_history.csv",
        state_history,
        temperature_ci95_lower=temperature_ci95_lower,
        temperature_ci95_upper=temperature_ci95_upper,
        pressure_ci95_lower=pressure_ci95_lower,
        pressure_ci95_upper=pressure_ci95_upper,
        co_mole_fraction_ci95_lower=co_mole_fraction_ci95_lower,
        co_mole_fraction_ci95_upper=co_mole_fraction_ci95_upper,
    )

    figure = plot_state_history(
        1.0e6 * (state_history.scan_time_s + DEFAULT_PHASE_START_S),
        state_history.temperature_k,
        state_history.pressure_atm,
        state_history.co_mole_fraction,
        temperature_lower=temperature_ci95_lower,
        temperature_upper=temperature_ci95_upper,
        pressure_lower=pressure_ci95_lower,
        pressure_upper=pressure_ci95_upper,
        co_mole_fraction_lower=co_mole_fraction_ci95_lower,
        co_mole_fraction_upper=co_mole_fraction_ci95_upper,
        uncertainty_label=f"{int(round(100.0 * confidence_level))}% CI",
        xlabel=r"Time [$\mu$s]",
        temperature_ylim=(1000.0, 5000.0),
        pressure_ylim=(-0.5, 4.5),
        co_mole_fraction_percent_ylim=(0.0, 6.5),
    )
    save_figure(figure, figure_output_dir / "state_history.png")
    plt.close(figure)

    return TimeHistoryPipelineResult(
        state_history=state_history,
        optical_path_length_cm=optical_path_length_cm,
        confidence_level=confidence_level,
        temperature_ci95_lower=temperature_ci95_lower,
        temperature_ci95_upper=temperature_ci95_upper,
        pressure_ci95_lower=pressure_ci95_lower,
        pressure_ci95_upper=pressure_ci95_upper,
        co_mole_fraction_ci95_lower=co_mole_fraction_ci95_lower,
        co_mole_fraction_ci95_upper=co_mole_fraction_ci95_upper,
    )
