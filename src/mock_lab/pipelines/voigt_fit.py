"""Pipeline for fitting the shock absorbance sweeps with three Voigt profiles."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from mock_lab.plotting.figures import plot_voigt_fit, save_figure
from mock_lab.plotting.mpl import plt
from mock_lab.spectroscopy.voigt import (
    DEFAULT_CO_TRANSITIONS,
    VoigtFitResult,
    fit_voigt_spectra,
)


Array1D = NDArray[np.float64]
Array2D = NDArray[np.float64]
Bool1D = NDArray[np.bool_]


@dataclass(frozen=True)
class VoigtFitPipelineResult:
    """Saved outputs from the three-transition Voigt-fitting pass."""

    frequency_cm_inv: Array1D
    fitted_absorbance_sweeps: Array2D
    success: Bool1D
    temperature_k: Array1D
    mean_apparent_pressure_atm: Array1D
    temperature_ci95_lower: Array1D
    temperature_ci95_upper: Array1D
    mean_apparent_pressure_ci95_lower: Array1D
    mean_apparent_pressure_ci95_upper: Array1D
    line_centers_relative_cm_inv: Array2D
    collisional_hwhm_cm_inv: Array2D
    line_areas: Array2D
    line_centers_ci95_lower: Array2D
    line_centers_ci95_upper: Array2D
    collisional_hwhm_ci95_lower: Array2D
    collisional_hwhm_ci95_upper: Array2D
    line_areas_ci95_lower: Array2D
    line_areas_ci95_upper: Array2D
    rmse_absorbance: Array1D
    plot_sweep_index: int


def _write_summary_table(
    path: Path,
    fit_results: tuple[VoigtFitResult | None, ...],
) -> None:
    """Write one CSV row per fitted sweep."""

    path.parent.mkdir(parents=True, exist_ok=True)
    transition_labels = [transition.label for transition in DEFAULT_CO_TRANSITIONS]

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        header = [
            "sweep_index",
            "success",
            "temperature_k",
            "temperature_ci95_half_width",
            "mean_apparent_pressure_atm",
            "mean_apparent_pressure_ci95_half_width",
            *[f"line_center_{label}" for label in transition_labels],
            *[f"line_center_ci95_half_width_{label}" for label in transition_labels],
            *[f"collisional_hwhm_{label}" for label in transition_labels],
            *[f"collisional_hwhm_ci95_half_width_{label}" for label in transition_labels],
            *[f"line_area_{label}" for label in transition_labels],
            *[f"line_area_ci95_half_width_{label}" for label in transition_labels],
            "baseline_offset",
            "baseline_slope",
            "rmse_absorbance",
            "nfev",
            "message",
        ]
        writer.writerow(header)

        for sweep_index, result in enumerate(fit_results):
            if result is None:
                writer.writerow([sweep_index, False, *[""] * (len(header) - 2)])
                continue

            writer.writerow(
                [
                    sweep_index,
                    result.success,
                    f"{result.parameters.temperature_k:.8g}",
                    f"{result.temperature_ci_half_width:.8g}",
                    f"{np.mean(result.apparent_pressure_atm):.8g}",
                    f"{result.mean_apparent_pressure_ci_half_width:.8g}",
                    *[f"{value:.8g}" for value in result.parameters.line_centers_relative_cm_inv],
                    *[f"{value:.8g}" for value in result.line_centers_ci_half_width],
                    *[f"{value:.8g}" for value in result.parameters.collisional_hwhm_cm_inv],
                    *[f"{value:.8g}" for value in result.collisional_hwhm_ci_half_width],
                    *[f"{value:.8g}" for value in result.parameters.line_areas],
                    *[f"{value:.8g}" for value in result.line_areas_ci_half_width],
                    f"{result.parameters.baseline_offset:.8g}",
                    f"{result.parameters.baseline_slope:.8g}",
                    f"{result.rmse_absorbance:.8g}",
                    result.nfev,
                    result.message,
                ]
            )


def _select_plot_result(
    fit_results: tuple[VoigtFitResult | None, ...],
    requested_index: int,
) -> tuple[int, VoigtFitResult]:
    """Return a valid fit result for plotting."""

    clamped_index = int(np.clip(requested_index, 0, len(fit_results) - 1))

    if fit_results[clamped_index] is not None:
        return clamped_index, fit_results[clamped_index]

    for sweep_index, result in enumerate(fit_results):
        if result is not None:
            return sweep_index, result

    raise RuntimeError("No successful Voigt-fit result is available for plotting.")


def run_voigt_fit_pipeline(
    shock_frequency_data: Path | str,
    output_dir: Path | str,
    *,
    figure_output_dir: Path | str | None = None,
    table_output_dir: Path | str | None = None,
    plot_sweep_index: int = 0,
    minimum_peak_absorbance: float = 0.02,
) -> VoigtFitPipelineResult:
    """Fit every usable shock absorbance sweep with the three-line Voigt model."""

    shock_frequency_data = Path(shock_frequency_data)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_output_dir = output_dir if figure_output_dir is None else Path(figure_output_dir)
    table_output_dir = output_dir if table_output_dir is None else Path(table_output_dir)
    figure_output_dir.mkdir(parents=True, exist_ok=True)
    table_output_dir.mkdir(parents=True, exist_ok=True)

    with np.load(shock_frequency_data) as data:
        frequency_cm_inv = np.asarray(data["relative_wavenumber_cm_inv"], dtype=float)
        absorbance_sweeps = np.asarray(data["absorbance_sweeps"], dtype=float)

    fit_results = fit_voigt_spectra(
        frequency_cm_inv,
        absorbance_sweeps,
        minimum_peak_absorbance=minimum_peak_absorbance,
    )

    fitted_absorbance_sweeps = np.full_like(absorbance_sweeps, np.nan)
    success = np.zeros(absorbance_sweeps.shape[0], dtype=bool)
    temperature_k = np.full(absorbance_sweeps.shape[0], np.nan, dtype=float)
    mean_apparent_pressure_atm = np.full(absorbance_sweeps.shape[0], np.nan, dtype=float)
    temperature_ci95_half_width = np.full(absorbance_sweeps.shape[0], np.nan, dtype=float)
    mean_apparent_pressure_ci95_half_width = np.full(absorbance_sweeps.shape[0], np.nan, dtype=float)
    line_centers_relative_cm_inv = np.full(
        (absorbance_sweeps.shape[0], len(DEFAULT_CO_TRANSITIONS)),
        np.nan,
        dtype=float,
    )
    collisional_hwhm_cm_inv = np.full(
        (absorbance_sweeps.shape[0], len(DEFAULT_CO_TRANSITIONS)),
        np.nan,
        dtype=float,
    )
    line_areas = np.full((absorbance_sweeps.shape[0], len(DEFAULT_CO_TRANSITIONS)), np.nan, dtype=float)
    line_centers_ci95_half_width = np.full_like(line_centers_relative_cm_inv, np.nan)
    collisional_hwhm_ci95_half_width = np.full_like(collisional_hwhm_cm_inv, np.nan)
    line_areas_ci95_half_width = np.full_like(line_areas, np.nan)
    reduced_parameter_vectors = np.full((absorbance_sweeps.shape[0], 8), np.nan, dtype=float)
    reduced_parameter_covariance = np.full((absorbance_sweeps.shape[0], 8, 8), np.nan, dtype=float)
    rmse_absorbance = np.full(absorbance_sweeps.shape[0], np.nan, dtype=float)
    baseline_offset = np.full(absorbance_sweeps.shape[0], np.nan, dtype=float)
    baseline_slope = np.full(absorbance_sweeps.shape[0], np.nan, dtype=float)
    confidence_level = float("nan")
    confidence_scale = float("nan")

    for sweep_index, result in enumerate(fit_results):
        if result is None:
            continue

        mask = np.isfinite(absorbance_sweeps[sweep_index])
        fitted_absorbance_sweeps[sweep_index, mask] = result.fitted_absorbance
        success[sweep_index] = result.success
        temperature_k[sweep_index] = result.parameters.temperature_k
        mean_apparent_pressure_atm[sweep_index] = float(np.mean(result.apparent_pressure_atm))
        temperature_ci95_half_width[sweep_index] = result.temperature_ci_half_width
        mean_apparent_pressure_ci95_half_width[sweep_index] = result.mean_apparent_pressure_ci_half_width
        line_centers_relative_cm_inv[sweep_index] = result.parameters.line_centers_relative_cm_inv
        collisional_hwhm_cm_inv[sweep_index] = result.parameters.collisional_hwhm_cm_inv
        line_areas[sweep_index] = result.parameters.line_areas
        line_centers_ci95_half_width[sweep_index] = result.line_centers_ci_half_width
        collisional_hwhm_ci95_half_width[sweep_index] = result.collisional_hwhm_ci_half_width
        line_areas_ci95_half_width[sweep_index] = result.line_areas_ci_half_width
        reduced_parameter_vectors[sweep_index] = result.reduced_parameter_vector
        reduced_parameter_covariance[sweep_index] = result.reduced_parameter_covariance
        rmse_absorbance[sweep_index] = result.rmse_absorbance
        baseline_offset[sweep_index] = result.parameters.baseline_offset
        baseline_slope[sweep_index] = result.parameters.baseline_slope
        confidence_level = result.confidence_level
        confidence_scale = result.confidence_scale

    temperature_ci95_lower = temperature_k - temperature_ci95_half_width
    temperature_ci95_upper = temperature_k + temperature_ci95_half_width
    mean_apparent_pressure_ci95_lower = (
        mean_apparent_pressure_atm - mean_apparent_pressure_ci95_half_width
    )
    mean_apparent_pressure_ci95_upper = (
        mean_apparent_pressure_atm + mean_apparent_pressure_ci95_half_width
    )
    line_centers_ci95_lower = line_centers_relative_cm_inv - line_centers_ci95_half_width
    line_centers_ci95_upper = line_centers_relative_cm_inv + line_centers_ci95_half_width
    collisional_hwhm_ci95_lower = collisional_hwhm_cm_inv - collisional_hwhm_ci95_half_width
    collisional_hwhm_ci95_upper = collisional_hwhm_cm_inv + collisional_hwhm_ci95_half_width
    line_areas_ci95_lower = line_areas - line_areas_ci95_half_width
    line_areas_ci95_upper = line_areas + line_areas_ci95_half_width

    selected_index, selected_result = _select_plot_result(fit_results, plot_sweep_index)

    np.savez(
        output_dir / "voigt_fit_results.npz",
        frequency_cm_inv=frequency_cm_inv,
        absorbance_sweeps=absorbance_sweeps,
        fitted_absorbance_sweeps=fitted_absorbance_sweeps,
        success=success,
        temperature_k=temperature_k,
        temperature_ci95_lower=temperature_ci95_lower,
        temperature_ci95_upper=temperature_ci95_upper,
        mean_apparent_pressure_atm=mean_apparent_pressure_atm,
        mean_apparent_pressure_ci95_lower=mean_apparent_pressure_ci95_lower,
        mean_apparent_pressure_ci95_upper=mean_apparent_pressure_ci95_upper,
        line_centers_relative_cm_inv=line_centers_relative_cm_inv,
        line_centers_ci95_lower=line_centers_ci95_lower,
        line_centers_ci95_upper=line_centers_ci95_upper,
        collisional_hwhm_cm_inv=collisional_hwhm_cm_inv,
        collisional_hwhm_ci95_lower=collisional_hwhm_ci95_lower,
        collisional_hwhm_ci95_upper=collisional_hwhm_ci95_upper,
        line_areas=line_areas,
        line_areas_ci95_lower=line_areas_ci95_lower,
        line_areas_ci95_upper=line_areas_ci95_upper,
        reduced_parameter_vectors=reduced_parameter_vectors,
        reduced_parameter_covariance=reduced_parameter_covariance,
        confidence_level=confidence_level,
        confidence_scale=confidence_scale,
        baseline_offset=baseline_offset,
        baseline_slope=baseline_slope,
        rmse_absorbance=rmse_absorbance,
        plot_sweep_index=selected_index,
        selected_component_absorbance=selected_result.component_absorbance,
        selected_component_labels=np.asarray(selected_result.labels, dtype="U16"),
        selected_line_centers_relative_cm_inv=selected_result.parameters.line_centers_relative_cm_inv,
    )

    _write_summary_table(table_output_dir / "voigt_fit_summary.csv", fit_results)

    fit_figure = plot_voigt_fit(
        selected_result.frequency_cm_inv,
        selected_result.absorbance,
        selected_result.fitted_absorbance,
        selected_result.component_absorbance,
        component_labels=selected_result.labels,
        component_centers_cm_inv=selected_result.parameters.line_centers_relative_cm_inv,
        report_style=True,
    )
    save_figure(fit_figure, figure_output_dir / "shock_voigt_fit.png")
    plt.close(fit_figure)

    return VoigtFitPipelineResult(
        frequency_cm_inv=frequency_cm_inv,
        fitted_absorbance_sweeps=fitted_absorbance_sweeps,
        success=success,
        temperature_k=temperature_k,
        mean_apparent_pressure_atm=mean_apparent_pressure_atm,
        temperature_ci95_lower=temperature_ci95_lower,
        temperature_ci95_upper=temperature_ci95_upper,
        mean_apparent_pressure_ci95_lower=mean_apparent_pressure_ci95_lower,
        mean_apparent_pressure_ci95_upper=mean_apparent_pressure_ci95_upper,
        line_centers_relative_cm_inv=line_centers_relative_cm_inv,
        collisional_hwhm_cm_inv=collisional_hwhm_cm_inv,
        line_areas=line_areas,
        line_centers_ci95_lower=line_centers_ci95_lower,
        line_centers_ci95_upper=line_centers_ci95_upper,
        collisional_hwhm_ci95_lower=collisional_hwhm_ci95_lower,
        collisional_hwhm_ci95_upper=collisional_hwhm_ci95_upper,
        line_areas_ci95_lower=line_areas_ci95_lower,
        line_areas_ci95_upper=line_areas_ci95_upper,
        rmse_absorbance=rmse_absorbance,
        plot_sweep_index=selected_index,
    )
