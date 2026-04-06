"""Voigt-profile modeling and fitting utilities for the CO mock lab."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares
from scipy.special import voigt_profile


Array1D = NDArray[np.float64]
Array2D = NDArray[np.float64]

T_REF_K = 296.0
BOLTZMANN_CONSTANT_J_K = 1.380649e-23
AVOGADRO_NUMBER = 6.02214076e23
LIGHT_SPEED_M_S = 299792458.0
CO_MOLAR_MASS_KG_MOL = 28.0101e-3
CO_MOLECULAR_MASS_KG = CO_MOLAR_MASS_KG_MOL / AVOGADRO_NUMBER


@dataclass(frozen=True)
class Transition:
    """Spectroscopic parameters for one CO transition."""

    label: str
    center_cm_inv: float
    line_strength_ref: float
    lower_state_energy_cm_inv: float
    gamma_n2_cm_inv_atm: float
    n_n2: float
    gamma_co_cm_inv_atm: float
    n_co: float


DEFAULT_CO_TRANSITIONS: tuple[Transition, ...] = (
    Transition(
        label="P(0,31)",
        center_cm_inv=2008.525,
        line_strength_ref=2.669e-22,
        lower_state_energy_cm_inv=1901.131,
        gamma_n2_cm_inv_atm=0.0412,
        n_n2=0.47,
        gamma_co_cm_inv_atm=0.0430,
        n_co=0.50,
    ),
    Transition(
        label="P(2,20)",
        center_cm_inv=2008.422,
        line_strength_ref=1.149e-28,
        lower_state_energy_cm_inv=5051.740,
        gamma_n2_cm_inv_atm=0.0526,
        n_n2=0.57,
        gamma_co_cm_inv_atm=0.0550,
        n_co=0.50,
    ),
    Transition(
        label="P(3,14)",
        center_cm_inv=2008.552,
        line_strength_ref=2.877e-32,
        lower_state_energy_cm_inv=6742.874,
        gamma_n2_cm_inv_atm=0.0607,
        n_n2=0.65,
        gamma_co_cm_inv_atm=0.0610,
        n_co=0.50,
    ),
)


@dataclass(frozen=True)
class VoigtFitParameters:
    """Physical and nuisance parameters for one fitted spectrum.

    This matches the MATLAB demo structure more closely:
    each line center, collisional width, and integrated area is a free
    parameter, while all transitions share one Doppler temperature.
    """

    temperature_k: float
    line_centers_relative_cm_inv: Array1D
    collisional_hwhm_cm_inv: Array1D
    line_areas: Array1D
    baseline_offset: float
    baseline_slope: float


@dataclass(frozen=True)
class VoigtFitResult:
    """Best-fit model and diagnostics for one shock absorbance spectrum."""

    frequency_cm_inv: Array1D
    absorbance: Array1D
    fitted_absorbance: Array1D
    component_absorbance: Array2D
    residual_absorbance: Array1D
    parameters: VoigtFitParameters
    gaussian_sigma_cm_inv: Array1D
    apparent_pressure_atm: Array1D
    labels: tuple[str, ...]
    success: bool
    status: int
    message: str
    nfev: int
    cost: float
    rmse_absorbance: float


def transition_relative_offsets(
    transitions: tuple[Transition, ...] = DEFAULT_CO_TRANSITIONS,
    anchor_index: int = 0,
) -> Array1D:
    """Return fixed line-center offsets relative to one anchor transition."""

    centers = np.asarray([transition.center_cm_inv for transition in transitions], dtype=float)
    return centers - centers[anchor_index]


def doppler_sigma_cm_inv(center_cm_inv: float, temperature_k: float) -> float:
    """Return the Gaussian standard deviation for Doppler broadening."""

    doppler_fwhm_cm_inv = center_cm_inv * np.sqrt(
        8.0
        * BOLTZMANN_CONSTANT_J_K
        * temperature_k
        * np.log(2.0)
        / (CO_MOLECULAR_MASS_KG * LIGHT_SPEED_M_S**2)
    )
    return float(doppler_fwhm_cm_inv / (2.0 * np.sqrt(2.0 * np.log(2.0))))


def integrated_area_guess(
    doppler_sigma_cm_inv_value: float,
    collisional_hwhm_cm_inv_value: float,
    peak_absorbance_guess: float,
) -> float:
    """Convert a peak absorbance guess into an integrated-area guess."""

    linecenter_profile_value = float(
        voigt_profile(
            np.array([0.0], dtype=float),
            doppler_sigma_cm_inv_value,
            collisional_hwhm_cm_inv_value,
        )[0]
    )
    return float(max(peak_absorbance_guess / linecenter_profile_value, 1.0e-8))


def apparent_pressure_atm(
    collisional_hwhm_cm_inv: Array1D,
    temperature_k: float,
    transitions: tuple[Transition, ...] = DEFAULT_CO_TRANSITIONS,
) -> Array1D:
    """Convert fitted collisional widths into per-line apparent pressures."""

    gamma_ref = np.asarray([transition.gamma_n2_cm_inv_atm for transition in transitions], dtype=float)
    n_values = np.asarray([transition.n_n2 for transition in transitions], dtype=float)
    scaling = gamma_ref * (T_REF_K / temperature_k) ** n_values
    return np.asarray(collisional_hwhm_cm_inv, dtype=float) / scaling


def evaluate_voigt_spectrum(
    frequency_cm_inv: Array1D,
    parameters: VoigtFitParameters,
    *,
    transitions: tuple[Transition, ...] = DEFAULT_CO_TRANSITIONS,
) -> tuple[Array1D, Array2D, Array1D, Array1D]:
    """Evaluate the sum of Voigt profiles on a frequency axis."""

    gaussian_sigma_cm_inv = np.asarray(
        [
            doppler_sigma_cm_inv(transition.center_cm_inv, parameters.temperature_k)
            for transition in transitions
        ],
        dtype=float,
    )
    centered_frequency = frequency_cm_inv - np.mean(frequency_cm_inv)
    component_absorbance = np.empty((len(transitions), frequency_cm_inv.size), dtype=float)

    for line_index, (line_area, center, sigma, gamma) in enumerate(
        zip(
            parameters.line_areas,
            parameters.line_centers_relative_cm_inv,
            gaussian_sigma_cm_inv,
            parameters.collisional_hwhm_cm_inv,
        )
    ):
        component_absorbance[line_index] = line_area * voigt_profile(
            frequency_cm_inv - center,
            sigma,
            gamma,
        )

    fitted_absorbance = np.sum(component_absorbance, axis=0)
    fitted_absorbance += parameters.baseline_offset + parameters.baseline_slope * centered_frequency

    return (
        fitted_absorbance,
        component_absorbance,
        gaussian_sigma_cm_inv,
        apparent_pressure_atm(
            parameters.collisional_hwhm_cm_inv,
            parameters.temperature_k,
            transitions=transitions,
        ),
    )


def estimate_initial_parameters(
    frequency_cm_inv: Array1D,
    absorbance: Array1D,
    *,
    transitions: tuple[Transition, ...] = DEFAULT_CO_TRANSITIONS,
    anchor_index: int = 0,
    temperature_guess_k: float = 1400.0,
    collisional_width_guess_cm_inv: float = 0.04,
    peak_search_half_width_cm_inv: float = 0.03,
) -> VoigtFitParameters:
    """Estimate a MATLAB-style initial parameter set for one spectrum."""

    mask = np.isfinite(frequency_cm_inv) & np.isfinite(absorbance)

    if np.count_nonzero(mask) < 16:
        raise ValueError("At least 16 finite samples are required to seed the Voigt fit.")

    frequency_fit = np.asarray(frequency_cm_inv[mask], dtype=float)
    absorbance_fit = np.asarray(absorbance[mask], dtype=float)
    edge_count = min(max(16, frequency_fit.size // 12), frequency_fit.size // 2)
    edge_indices = np.concatenate(
        [
            np.arange(edge_count, dtype=np.int64),
            np.arange(frequency_fit.size - edge_count, frequency_fit.size, dtype=np.int64),
        ]
    )
    baseline_coefficients = np.polyfit(
        frequency_fit[edge_indices],
        absorbance_fit[edge_indices],
        1,
    )
    frequency_mean = float(np.mean(frequency_fit))
    baseline_slope = float(baseline_coefficients[0])
    baseline_offset = float(np.polyval(baseline_coefficients, frequency_mean))
    baseline_fit = baseline_offset + baseline_slope * (frequency_fit - frequency_mean)
    baseline_corrected_absorbance = absorbance_fit - baseline_fit

    anchor_center_guess = float(frequency_fit[int(np.nanargmax(baseline_corrected_absorbance))])
    initial_center_guesses = anchor_center_guess + transition_relative_offsets(
        transitions,
        anchor_index=anchor_index,
    )
    collisional_width_guesses = np.full(len(transitions), collisional_width_guess_cm_inv, dtype=float)
    line_area_guesses = np.empty(len(transitions), dtype=float)

    for line_index, (center_guess, transition, width_guess) in enumerate(
        zip(initial_center_guesses, transitions, collisional_width_guesses)
    ):
        local_mask = np.abs(frequency_fit - center_guess) <= peak_search_half_width_cm_inv

        if np.any(local_mask):
            local_frequency = frequency_fit[local_mask]
            local_absorbance = baseline_corrected_absorbance[local_mask]
            local_peak_index = int(np.nanargmax(local_absorbance))
            refined_center = float(local_frequency[local_peak_index])
            peak_absorbance_guess = float(max(local_absorbance[local_peak_index], 1.0e-4))
            initial_center_guesses[line_index] = refined_center
        else:
            peak_absorbance_guess = float(max(np.nanmax(baseline_corrected_absorbance), 1.0e-4))

        sigma_guess = doppler_sigma_cm_inv(transition.center_cm_inv, temperature_guess_k)
        line_area_guesses[line_index] = integrated_area_guess(
            sigma_guess,
            width_guess,
            peak_absorbance_guess,
        )

    return VoigtFitParameters(
        temperature_k=float(temperature_guess_k),
        line_centers_relative_cm_inv=np.asarray(initial_center_guesses, dtype=float),
        collisional_hwhm_cm_inv=np.asarray(collisional_width_guesses, dtype=float),
        line_areas=np.asarray(line_area_guesses, dtype=float),
        baseline_offset=baseline_offset,
        baseline_slope=baseline_slope,
    )


def _parameters_to_vector(parameters: VoigtFitParameters) -> Array1D:
    """Pack fit parameters into the optimizer vector."""

    return np.concatenate(
        [
            np.array([parameters.temperature_k], dtype=float),
            np.asarray(parameters.line_centers_relative_cm_inv, dtype=float),
            np.asarray(parameters.collisional_hwhm_cm_inv, dtype=float),
            np.asarray(parameters.line_areas, dtype=float),
            np.array([parameters.baseline_offset, parameters.baseline_slope], dtype=float),
        ]
    )


def _vector_to_parameters(
    parameter_vector: Array1D,
    transition_count: int,
) -> VoigtFitParameters:
    """Unpack the optimizer vector into a dataclass."""

    centers_start = 1
    widths_start = centers_start + transition_count
    areas_start = widths_start + transition_count

    return VoigtFitParameters(
        temperature_k=float(parameter_vector[0]),
        line_centers_relative_cm_inv=np.asarray(
            parameter_vector[centers_start:widths_start],
            dtype=float,
        ),
        collisional_hwhm_cm_inv=np.asarray(
            parameter_vector[widths_start:areas_start],
            dtype=float,
        ),
        line_areas=np.asarray(
            parameter_vector[areas_start : areas_start + transition_count],
            dtype=float,
        ),
        baseline_offset=float(parameter_vector[-2]),
        baseline_slope=float(parameter_vector[-1]),
    )


def _parameter_bounds(
    frequency_cm_inv: Array1D,
    initial_parameters: VoigtFitParameters,
    transition_count: int,
    center_half_window_cm_inv: float = 0.06,
) -> tuple[Array1D, Array1D]:
    """Return bounded search limits around the MATLAB-style initial guesses."""

    center_lower = np.maximum(
        initial_parameters.line_centers_relative_cm_inv - center_half_window_cm_inv,
        np.min(frequency_cm_inv) - 0.01,
    )
    center_upper = np.minimum(
        initial_parameters.line_centers_relative_cm_inv + center_half_window_cm_inv,
        np.max(frequency_cm_inv) + 0.01,
    )
    lower_bounds = np.concatenate(
        [
            np.array([500.0], dtype=float),
            center_lower,
            np.full(transition_count, 1.0e-4, dtype=float),
            np.full(transition_count, 1.0e-8, dtype=float),
            np.array([-0.2, -5.0], dtype=float),
        ]
    )
    upper_bounds = np.concatenate(
        [
            np.array([6000.0], dtype=float),
            center_upper,
            np.full(transition_count, 0.5, dtype=float),
            np.full(transition_count, 10.0, dtype=float),
            np.array([0.2, 5.0], dtype=float),
        ]
    )
    return lower_bounds, upper_bounds


def fit_voigt_spectrum(
    frequency_cm_inv: Array1D,
    absorbance: Array1D,
    *,
    initial_parameters: VoigtFitParameters | None = None,
    transitions: tuple[Transition, ...] = DEFAULT_CO_TRANSITIONS,
    anchor_index: int = 0,
    max_nfev: int = 1500,
) -> VoigtFitResult:
    """Fit one absorbance spectrum with a sum of three Voigt profiles.

    This keeps a shared Doppler temperature, but it frees every line center,
    collisional width, and integrated area to match the MATLAB demo structure
    more closely than the earlier constrained-offset implementation.
    """

    if frequency_cm_inv.shape != absorbance.shape:
        raise ValueError("Frequency and absorbance inputs must have matching shapes.")

    mask = np.isfinite(frequency_cm_inv) & np.isfinite(absorbance)

    if np.count_nonzero(mask) < 16:
        raise ValueError("At least 16 finite samples are required for Voigt fitting.")

    frequency_fit = np.asarray(frequency_cm_inv[mask], dtype=float)
    absorbance_fit = np.asarray(absorbance[mask], dtype=float)

    if initial_parameters is None:
        initial_parameters = estimate_initial_parameters(
            frequency_fit,
            absorbance_fit,
            transitions=transitions,
            anchor_index=anchor_index,
        )

    lower_bounds, upper_bounds = _parameter_bounds(
        frequency_fit,
        initial_parameters,
        len(transitions),
    )
    initial_vector = np.clip(_parameters_to_vector(initial_parameters), lower_bounds, upper_bounds)

    def residual_function(parameter_vector: Array1D) -> Array1D:
        parameters = _vector_to_parameters(parameter_vector, len(transitions))
        fitted_absorbance, _, _, _ = evaluate_voigt_spectrum(
            frequency_fit,
            parameters,
            transitions=transitions,
        )
        return fitted_absorbance - absorbance_fit

    optimization_result = least_squares(
        residual_function,
        initial_vector,
        bounds=(lower_bounds, upper_bounds),
        method="trf",
        loss="soft_l1",
        f_scale=0.01,
        max_nfev=max_nfev,
    )
    best_fit_parameters = _vector_to_parameters(
        np.asarray(optimization_result.x, dtype=float),
        len(transitions),
    )
    (
        fitted_absorbance,
        component_absorbance,
        gaussian_sigma_cm_inv,
        apparent_pressure_values_atm,
    ) = evaluate_voigt_spectrum(
        frequency_fit,
        best_fit_parameters,
        transitions=transitions,
    )
    residual_absorbance = absorbance_fit - fitted_absorbance

    return VoigtFitResult(
        frequency_cm_inv=frequency_fit,
        absorbance=absorbance_fit,
        fitted_absorbance=fitted_absorbance,
        component_absorbance=component_absorbance,
        residual_absorbance=residual_absorbance,
        parameters=best_fit_parameters,
        gaussian_sigma_cm_inv=np.asarray(gaussian_sigma_cm_inv, dtype=float),
        apparent_pressure_atm=np.asarray(apparent_pressure_values_atm, dtype=float),
        labels=tuple(transition.label for transition in transitions),
        success=bool(optimization_result.success),
        status=int(optimization_result.status),
        message=str(optimization_result.message),
        nfev=int(optimization_result.nfev),
        cost=float(optimization_result.cost),
        rmse_absorbance=float(np.sqrt(np.mean(residual_absorbance**2))),
    )


def fit_voigt_spectra(
    frequency_cm_inv: Array1D,
    absorbance_sweeps: Array2D,
    *,
    transitions: tuple[Transition, ...] = DEFAULT_CO_TRANSITIONS,
    anchor_index: int = 0,
    minimum_peak_absorbance: float = 0.02,
) -> tuple[VoigtFitResult | None, ...]:
    """Fit a stack of absorbance spectra sweep-by-sweep."""

    if absorbance_sweeps.ndim != 2:
        raise ValueError("Batch Voigt fitting expects a 2D array of absorbance sweeps.")

    results: list[VoigtFitResult | None] = []
    initial_parameters: VoigtFitParameters | None = None

    for sweep in absorbance_sweeps:
        if np.count_nonzero(np.isfinite(sweep)) < 16 or float(np.nanmax(sweep)) < minimum_peak_absorbance:
            results.append(None)
            continue

        fit_result = fit_voigt_spectrum(
            frequency_cm_inv,
            sweep,
            initial_parameters=initial_parameters,
            transitions=transitions,
            anchor_index=anchor_index,
        )
        results.append(fit_result)

        if fit_result.success:
            initial_parameters = fit_result.parameters

    return tuple(results)
