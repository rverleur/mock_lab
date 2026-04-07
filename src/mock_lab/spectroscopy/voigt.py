"""Voigt-profile modeling and fitting utilities for the CO mock lab."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares
from scipy.special import voigt_profile
from scipy.stats import t as student_t

from mock_lab.spectroscopy.tips import get_co_partition_sum


Array1D = NDArray[np.float64]
Array2D = NDArray[np.float64]

T_REF_K = 296.0
BOLTZMANN_CONSTANT_J_K = 1.380649e-23
AVOGADRO_NUMBER = 6.02214076e23
LIGHT_SPEED_M_S = 299792458.0
CO_MOLAR_MASS_KG_MOL = 28.0101e-3
CO_MOLECULAR_MASS_KG = CO_MOLAR_MASS_KG_MOL / AVOGADRO_NUMBER
SECOND_RADIATION_CONSTANT_CM_K = 1.438776877


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


# These three transitions are the handout-provided spectroscopic constants used
# throughout the current fitting and state-reduction workflow.
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

    The stored result is always expanded into explicit per-line centers,
    collisional widths, and integrated areas even though the optimizer now
    uses a constrained internal parameterization:

    - `P(0,31)` is the anchor line and is forced to remain the strongest line
    - `P(3,14)` stays at a fixed center offset from `P(0,31)`
    - `P(3,14)` shares its collisional width with `P(2,20)`
    - the three integrated areas are tied together by temperature-dependent
      line-strength ratios, so only the strongest-line area is fitted freely
    - all three transitions still share one Doppler temperature
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
    reduced_parameter_vector: Array1D
    reduced_parameter_covariance: Array2D
    confidence_level: float
    confidence_scale: float
    temperature_ci_half_width: float
    line_centers_ci_half_width: Array1D
    collisional_hwhm_ci_half_width: Array1D
    line_areas_ci_half_width: Array1D
    mean_apparent_pressure_ci_half_width: float


def transition_relative_offsets(
    transitions: tuple[Transition, ...] = DEFAULT_CO_TRANSITIONS,
    anchor_index: int = 0,
) -> Array1D:
    """Return fixed line-center offsets relative to one anchor transition."""

    centers = np.asarray([transition.center_cm_inv for transition in transitions], dtype=float)
    return centers - centers[anchor_index]


def co_partition_function_ratio(temperature_k: float) -> float:
    """Return the CO partition-function ratio Q(T)/Q(T_ref)."""

    return float(get_co_partition_sum(float(temperature_k)) / get_co_partition_sum(float(T_REF_K)))


def line_strength_at_temperature(
    temperature_k: float,
    transition: Transition,
) -> float:
    """Return the temperature-scaled integrated line strength for one transition."""

    partition_ratio = co_partition_function_ratio(temperature_k)
    stimulated_emission_ratio = (
        1.0
        - np.exp(-SECOND_RADIATION_CONSTANT_CM_K * transition.center_cm_inv / temperature_k)
    ) / (
        1.0
        - np.exp(-SECOND_RADIATION_CONSTANT_CM_K * transition.center_cm_inv / T_REF_K)
    )
    lower_state_ratio = np.exp(
        -SECOND_RADIATION_CONSTANT_CM_K
        * transition.lower_state_energy_cm_inv
        * (1.0 / temperature_k - 1.0 / T_REF_K)
    )
    return float(
        transition.line_strength_ref
        * (1.0 / partition_ratio)
        * lower_state_ratio
        * stimulated_emission_ratio
    )


def transition_strength_ratios(
    temperature_k: float,
    transitions: tuple[Transition, ...] = DEFAULT_CO_TRANSITIONS,
    anchor_index: int = 0,
) -> Array1D:
    """Return temperature-dependent integrated-area ratios relative to the anchor line."""

    strengths = np.asarray(
        [line_strength_at_temperature(float(temperature_k), transition) for transition in transitions],
        dtype=float,
    )
    anchor_strength = float(strengths[anchor_index])

    if not np.isfinite(anchor_strength) or anchor_strength <= 0.0:
        raise ValueError("Anchor transition line strength must remain positive.")

    return strengths / anchor_strength


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


def _confidence_scale(confidence_level: float, dof: int) -> float:
    """Return the two-sided confidence multiplier for a Student-t interval."""

    if dof <= 0:
        return float("nan")

    return float(student_t.ppf(0.5 * (1.0 + confidence_level), dof))


def _finite_difference_jacobian(
    function: Callable[[Array1D], Array1D],
    parameter_vector: Array1D,
    relative_step: float = 1.0e-6,
) -> Array2D:
    """Estimate the Jacobian of a vector-valued function by central differences."""

    parameter_vector = np.asarray(parameter_vector, dtype=float)
    base_value = np.atleast_1d(np.asarray(function(parameter_vector), dtype=float))
    jacobian = np.empty((base_value.size, parameter_vector.size), dtype=float)

    for index in range(parameter_vector.size):
        step = relative_step * max(abs(float(parameter_vector[index])), 1.0)
        plus_vector = np.array(parameter_vector, copy=True)
        minus_vector = np.array(parameter_vector, copy=True)
        plus_vector[index] += step
        minus_vector[index] -= step
        plus_value = np.atleast_1d(np.asarray(function(plus_vector), dtype=float))
        minus_value = np.atleast_1d(np.asarray(function(minus_vector), dtype=float))
        jacobian[:, index] = (plus_value - minus_value) / (2.0 * step)

    return jacobian


def _reduced_parameter_covariance(
    jacobian: Array2D,
    residual_vector: Array1D,
) -> tuple[Array2D, float]:
    """Estimate the reduced-parameter covariance from the least-squares Jacobian."""

    jacobian = np.asarray(jacobian, dtype=float)
    residual_vector = np.asarray(residual_vector, dtype=float)
    degrees_of_freedom = residual_vector.size - jacobian.shape[1]

    if degrees_of_freedom <= 0:
        covariance = np.full((jacobian.shape[1], jacobian.shape[1]), np.nan, dtype=float)
        return covariance, float("nan")

    residual_variance = float(np.sum(residual_vector**2) / degrees_of_freedom)
    information_matrix = jacobian.T @ jacobian
    covariance = residual_variance * np.linalg.pinv(information_matrix)
    return np.asarray(covariance, dtype=float), residual_variance


def _fit_summary_vector(
    parameters: VoigtFitParameters,
    mean_apparent_pressure_atm: float,
) -> Array1D:
    """Flatten the main reported fit quantities into one vector."""

    return np.concatenate(
        [
            np.array([parameters.temperature_k], dtype=float),
            np.asarray(parameters.line_centers_relative_cm_inv, dtype=float),
            np.asarray(parameters.collisional_hwhm_cm_inv, dtype=float),
            np.asarray(parameters.line_areas, dtype=float),
            np.array([mean_apparent_pressure_atm], dtype=float),
        ]
    )


def finite_difference_jacobian(
    function: Callable[[Array1D], Array1D],
    parameter_vector: Array1D,
    relative_step: float = 1.0e-6,
) -> Array2D:
    """Public wrapper around the local finite-difference Jacobian helper."""

    return _finite_difference_jacobian(function, parameter_vector, relative_step=relative_step)


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
    """Estimate a constrained initial parameter set for one spectrum."""

    if anchor_index != 0:
        raise ValueError("The constrained three-line fit currently requires anchor_index=0.")

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
    nominal_offsets = transition_relative_offsets(
        transitions,
        anchor_index=anchor_index,
    )
    initial_center_guesses = np.empty(len(transitions), dtype=float)
    initial_center_guesses[0] = anchor_center_guess

    if len(transitions) > 1:
        secondary_center_guess = anchor_center_guess + nominal_offsets[1]
        local_mask = np.abs(frequency_fit - secondary_center_guess) <= peak_search_half_width_cm_inv

        if np.any(local_mask):
            local_frequency = frequency_fit[local_mask]
            local_absorbance = baseline_corrected_absorbance[local_mask]
            local_peak_index = int(np.nanargmax(local_absorbance))
            initial_center_guesses[1] = float(local_frequency[local_peak_index])
        else:
            initial_center_guesses[1] = secondary_center_guess

    if len(transitions) > 2:
        initial_center_guesses[2:] = anchor_center_guess + nominal_offsets[2:]

    collisional_width_guesses = np.full(len(transitions), collisional_width_guess_cm_inv, dtype=float)
    if len(transitions) > 2:
        collisional_width_guesses[2:] = collisional_width_guesses[1]

    peak_absorbance_guess = float(max(np.nanmax(baseline_corrected_absorbance), 1.0e-4))
    sigma_guess = doppler_sigma_cm_inv(transitions[0].center_cm_inv, temperature_guess_k)
    strongest_line_area_guess = integrated_area_guess(
        sigma_guess,
        collisional_width_guess_cm_inv,
        peak_absorbance_guess,
    )
    strength_ratios = transition_strength_ratios(
        temperature_guess_k,
        transitions=transitions,
        anchor_index=anchor_index,
    )
    line_area_guesses = strongest_line_area_guess * strength_ratios

    return VoigtFitParameters(
        temperature_k=float(temperature_guess_k),
        line_centers_relative_cm_inv=np.asarray(initial_center_guesses, dtype=float),
        collisional_hwhm_cm_inv=np.asarray(collisional_width_guesses, dtype=float),
        line_areas=np.asarray(line_area_guesses, dtype=float),
        baseline_offset=baseline_offset,
        baseline_slope=baseline_slope,
    )


def _parameters_to_vector(
    parameters: VoigtFitParameters,
    transitions: tuple[Transition, ...] = DEFAULT_CO_TRANSITIONS,
) -> Array1D:
    """Pack explicit fit parameters into the constrained optimizer vector."""

    nominal_offsets = transition_relative_offsets(transitions, anchor_index=0)
    secondary_center_delta = 0.0

    if parameters.line_centers_relative_cm_inv.size > 1:
        secondary_center_delta = float(
            parameters.line_centers_relative_cm_inv[1]
            - (parameters.line_centers_relative_cm_inv[0] + nominal_offsets[1])
        )

    primary_width = float(parameters.collisional_hwhm_cm_inv[0])
    shared_weak_width = float(
        parameters.collisional_hwhm_cm_inv[1]
        if parameters.collisional_hwhm_cm_inv.size > 1
        else primary_width
    )
    strongest_line_area = float(parameters.line_areas[0])

    return np.concatenate(
        [
            np.array([parameters.temperature_k], dtype=float),
            np.array([parameters.line_centers_relative_cm_inv[0]], dtype=float),
            np.array([secondary_center_delta], dtype=float),
            np.array([primary_width, shared_weak_width], dtype=float),
            np.array([strongest_line_area], dtype=float),
            np.array([parameters.baseline_offset, parameters.baseline_slope], dtype=float),
        ]
    )


def _vector_to_parameters(
    parameter_vector: Array1D,
    transition_count: int,
    transitions: tuple[Transition, ...] = DEFAULT_CO_TRANSITIONS,
) -> VoigtFitParameters:
    """Expand the constrained optimizer vector into explicit line parameters."""

    if transition_count != len(transitions):
        raise ValueError("Transition count must match the constrained transition set.")

    nominal_offsets = transition_relative_offsets(transitions, anchor_index=0)
    anchor_center = float(parameter_vector[1])
    secondary_center_delta = float(parameter_vector[2])
    temperature_k = float(parameter_vector[0])
    line_centers = np.asarray(
        [
            anchor_center,
            anchor_center + nominal_offsets[1] + secondary_center_delta,
            anchor_center + nominal_offsets[2],
        ],
        dtype=float,
    )
    primary_width = float(parameter_vector[3])
    shared_weak_width = float(parameter_vector[4])
    collisional_widths = np.asarray(
        [primary_width, shared_weak_width, shared_weak_width],
        dtype=float,
    )
    strongest_line_area = float(parameter_vector[5])
    line_areas = strongest_line_area * transition_strength_ratios(
        temperature_k,
        transitions=transitions,
        anchor_index=0,
    )

    return VoigtFitParameters(
        temperature_k=temperature_k,
        line_centers_relative_cm_inv=line_centers,
        collisional_hwhm_cm_inv=collisional_widths,
        line_areas=np.asarray(line_areas, dtype=float),
        baseline_offset=float(parameter_vector[-2]),
        baseline_slope=float(parameter_vector[-1]),
    )


def pack_constrained_parameters(
    parameters: VoigtFitParameters,
    transitions: tuple[Transition, ...] = DEFAULT_CO_TRANSITIONS,
) -> Array1D:
    """Return the constrained optimizer vector for one explicit fit result."""

    return _parameters_to_vector(parameters, transitions=transitions)


def expand_constrained_parameters(
    parameter_vector: Array1D,
    transitions: tuple[Transition, ...] = DEFAULT_CO_TRANSITIONS,
) -> VoigtFitParameters:
    """Expand one constrained optimizer vector into explicit line parameters."""

    return _vector_to_parameters(
        np.asarray(parameter_vector, dtype=float),
        len(transitions),
        transitions=transitions,
    )


def _parameter_bounds(
    frequency_cm_inv: Array1D,
    initial_parameters: VoigtFitParameters,
    transition_count: int,
    transitions: tuple[Transition, ...] = DEFAULT_CO_TRANSITIONS,
    center_half_window_cm_inv: float = 0.05,
    secondary_center_delta_limit_cm_inv: float = 0.03,
) -> tuple[Array1D, Array1D]:
    """Return bounded search limits for the constrained optimizer vector."""

    if transition_count != len(transitions):
        raise ValueError("Transition count must match the constrained transition set.")

    nominal_offsets = transition_relative_offsets(transitions, anchor_index=0)
    anchor_center = float(initial_parameters.line_centers_relative_cm_inv[0])
    secondary_center_delta = float(
        initial_parameters.line_centers_relative_cm_inv[1]
        - (anchor_center + nominal_offsets[1])
    )
    center_lower = max(anchor_center - center_half_window_cm_inv, float(np.min(frequency_cm_inv) - 0.01))
    center_upper = min(anchor_center + center_half_window_cm_inv, float(np.max(frequency_cm_inv) + 0.01))
    lower_bounds = np.concatenate(
        [
            np.array([500.0], dtype=float),
            np.array([center_lower], dtype=float),
            np.array([secondary_center_delta - secondary_center_delta_limit_cm_inv], dtype=float),
            np.array([1.0e-4, 1.0e-4], dtype=float),
            np.array([1.0e-8], dtype=float),
            np.array([-0.2, -5.0], dtype=float),
        ]
    )
    upper_bounds = np.concatenate(
        [
            np.array([6000.0], dtype=float),
            np.array([center_upper], dtype=float),
            np.array([secondary_center_delta + secondary_center_delta_limit_cm_inv], dtype=float),
            np.array([0.5, 0.5], dtype=float),
            np.array([10.0], dtype=float),
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
    confidence_level: float = 0.95,
) -> VoigtFitResult:
    """Fit one absorbance spectrum with a sum of three Voigt profiles.

    The current constrained model is tuned to keep line identities stable
    sweep-to-sweep:

    - `P(0,31)` is the anchor line and remains the dominant transition
    - `P(2,20)` keeps a small free center adjustment near its nominal offset
    - `P(3,14)` is held at a fixed center offset from `P(0,31)`
    - `P(2,20)` and `P(3,14)` share a collisional half-width
    - all line areas are derived from one strongest-line area and
      temperature-dependent line-strength ratios
    """

    if anchor_index != 0:
        raise ValueError("The constrained three-line fit currently requires anchor_index=0.")

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
        transitions=transitions,
    )
    initial_vector = np.clip(
        _parameters_to_vector(initial_parameters, transitions=transitions),
        lower_bounds,
        upper_bounds,
    )

    def residual_function(parameter_vector: Array1D) -> Array1D:
        parameters = _vector_to_parameters(
            parameter_vector,
            len(transitions),
            transitions=transitions,
        )
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
    best_fit_vector = np.asarray(optimization_result.x, dtype=float)
    best_fit_parameters = _vector_to_parameters(
        best_fit_vector,
        len(transitions),
        transitions=transitions,
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
    mean_apparent_pressure_atm = float(np.mean(apparent_pressure_values_atm))
    reduced_parameter_covariance, _ = _reduced_parameter_covariance(
        np.asarray(optimization_result.jac, dtype=float),
        np.asarray(optimization_result.fun, dtype=float),
    )
    degrees_of_freedom = frequency_fit.size - best_fit_vector.size
    confidence_scale = _confidence_scale(confidence_level, degrees_of_freedom)

    temperature_ci_half_width = float("nan")
    line_centers_ci_half_width = np.full(len(transitions), np.nan, dtype=float)
    collisional_hwhm_ci_half_width = np.full(len(transitions), np.nan, dtype=float)
    line_areas_ci_half_width = np.full(len(transitions), np.nan, dtype=float)
    mean_apparent_pressure_ci_half_width = float("nan")

    if np.all(np.isfinite(reduced_parameter_covariance)) and np.isfinite(confidence_scale):
        def summary_function(parameter_vector: Array1D) -> Array1D:
            parameters = expand_constrained_parameters(parameter_vector, transitions=transitions)
            mean_pressure = float(
                np.mean(
                    apparent_pressure_atm(
                        parameters.collisional_hwhm_cm_inv,
                        parameters.temperature_k,
                        transitions=transitions,
                    )
                )
            )
            return _fit_summary_vector(parameters, mean_pressure)

        summary_jacobian = _finite_difference_jacobian(summary_function, best_fit_vector)
        summary_covariance = summary_jacobian @ reduced_parameter_covariance @ summary_jacobian.T
        summary_standard_error = np.sqrt(
            np.clip(np.diag(np.asarray(summary_covariance, dtype=float)), a_min=0.0, a_max=None)
        )
        summary_half_width = confidence_scale * summary_standard_error
        transition_count = len(transitions)
        centers_start = 1
        widths_start = centers_start + transition_count
        areas_start = widths_start + transition_count
        pressure_index = areas_start + transition_count
        temperature_ci_half_width = float(summary_half_width[0])
        line_centers_ci_half_width = np.asarray(
            summary_half_width[centers_start:widths_start],
            dtype=float,
        )
        collisional_hwhm_ci_half_width = np.asarray(
            summary_half_width[widths_start:areas_start],
            dtype=float,
        )
        line_areas_ci_half_width = np.asarray(
            summary_half_width[areas_start:pressure_index],
            dtype=float,
        )
        mean_apparent_pressure_ci_half_width = float(summary_half_width[pressure_index])

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
        reduced_parameter_vector=best_fit_vector,
        reduced_parameter_covariance=np.asarray(reduced_parameter_covariance, dtype=float),
        confidence_level=float(confidence_level),
        confidence_scale=float(confidence_scale),
        temperature_ci_half_width=temperature_ci_half_width,
        line_centers_ci_half_width=np.asarray(line_centers_ci_half_width, dtype=float),
        collisional_hwhm_ci_half_width=np.asarray(collisional_hwhm_ci_half_width, dtype=float),
        line_areas_ci_half_width=np.asarray(line_areas_ci_half_width, dtype=float),
        mean_apparent_pressure_ci_half_width=mean_apparent_pressure_ci_half_width,
    )


def fit_voigt_spectra(
    frequency_cm_inv: Array1D,
    absorbance_sweeps: Array2D,
    *,
    transitions: tuple[Transition, ...] = DEFAULT_CO_TRANSITIONS,
    anchor_index: int = 0,
    minimum_peak_absorbance: float = 0.02,
    confidence_level: float = 0.95,
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
            confidence_level=confidence_level,
        )
        results.append(fit_result)

        if fit_result.success:
            initial_parameters = fit_result.parameters

    return tuple(results)
