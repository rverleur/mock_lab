"""State-estimation utilities derived from the fitted absorbance spectra."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares

from mock_lab.spectroscopy.collisional_broadening import (
    BATH_GAS_REFERENCE_SPECIES,
    bath_gas_model_half_widths,
    effective_bath_gas_broadening_fwhm_coefficient_cm_inv_atm,
)
from mock_lab.spectroscopy.voigt import (
    DEFAULT_CO_TRANSITIONS,
    Transition,
    VoigtFitParameters,
    line_strength_at_temperature,
)


Array1D = NDArray[np.float64]
Array2D = NDArray[np.float64]

DEFAULT_OPTICAL_PATH_LENGTH_CM = 10.32
DEFAULT_PRESSURE_BROADENING_SCALE = 0.84
DEFAULT_STATE_SOLVER_ITERATIONS = 10
PRESSURE_SOLVE_TRANSITION_INDEX = 0


@dataclass(frozen=True)
class StateHistory:
    """Scan-by-scan thermochemical estimates derived from the Voigt fits."""

    scan_index: Array1D
    scan_time_s: Array1D
    temperature_k: Array1D
    pressure_atm: Array1D
    co_mole_fraction: Array1D


@dataclass(frozen=True)
class StateEstimate:
    """One self-consistent state estimate from fitted linewidth and line area."""

    temperature_k: float
    pressure_atm: float
    co_mole_fraction: float
    per_line_pressure_atm: Array1D


def _pressure_transition_index(transition_count: int) -> int | None:
    """Return the transition index used for pressure / mole-fraction recovery."""

    if 0 <= PRESSURE_SOLVE_TRANSITION_INDEX < transition_count:
        return PRESSURE_SOLVE_TRANSITION_INDEX
    return None


def _co_mole_fraction_from_pressure(
    temperature_k: float,
    pressure_atm: float,
    integrated_area_cm_inv: float,
    *,
    optical_path_length_cm: float,
    transition: Transition,
) -> float:
    """Return the physically bounded CO mole fraction implied by one pressure.

    This helper is used during uncertainty propagation. When a perturbed
    pressure approaches zero, the raw algebraic value from `Area / (SPL)` can
    become arbitrarily large even though a mole fraction cannot exceed one.
    Clipping here prevents meaningless super-physical values from dominating
    the uncertainty combination.
    """

    value = estimate_co_mole_fraction(
        np.array([temperature_k], dtype=float),
        np.array([pressure_atm], dtype=float),
        np.array([integrated_area_cm_inv], dtype=float),
        optical_path_length_cm=optical_path_length_cm,
        transition=transition,
    )[0]
    if np.isfinite(value):
        value = float(np.clip(value, 0.0, 0.999))
    return float(value)


def _root_sum_square(values: Array2D) -> Array1D:
    """Return an overflow-safe RSS combination along axis 0.

    The uncertainty propagation can occasionally produce very large but finite
    intermediate values on weak or inconsistent early-time sweeps. Computing
    RSS with a direct `values**2` can overflow even when the final result only
    needs to be treated as "very large". This helper rescales each column
    before squaring, which is numerically stable.
    """

    matrix = np.asarray(values, dtype=float)

    if matrix.ndim == 1:
        matrix = matrix[np.newaxis, :]

    finite_abs = np.where(np.isfinite(matrix), np.abs(matrix), np.nan)
    scale = np.nanmax(finite_abs, axis=0)
    combined = np.zeros(matrix.shape[1], dtype=float)

    valid = np.isfinite(scale) & (scale > 0.0)
    if np.any(valid):
        normalized = matrix[:, valid] / scale[valid]
        normalized[~np.isfinite(normalized)] = 0.0
        combined[valid] = scale[valid] * np.sqrt(np.sum(normalized**2, axis=0))

    no_finite = ~np.isfinite(scale)
    combined[no_finite] = np.nan
    return np.asarray(combined, dtype=float)


def _effective_broadening_fwhm_vector(
    temperature_k: float,
    co_mole_fraction: float,
    transitions: tuple[Transition, ...],
    *,
    broadening_scale: float = DEFAULT_PRESSURE_BROADENING_SCALE,
) -> Array1D:
    """Return the per-transition collisional FWHM coefficients in `cm^-1/atm`."""

    return np.asarray(
        [
            broadening_scale
            * effective_bath_gas_broadening_fwhm_coefficient_cm_inv_atm(
                transition_value.broadening_by_species,
                float(temperature_k),
                float(co_mole_fraction),
            )
            for transition_value in transitions
        ],
        dtype=float,
    )


def _state_residual_vector(
    state_vector: Array1D,
    *,
    temperature_k: float,
    collisional_hwhm_cm_inv: Array1D,
    integrated_area_cm_inv: float,
    optical_path_length_cm: float,
    transitions: tuple[Transition, ...],
    transition: Transition,
    pressure_transition_index: int,
    broadening_scale: float,
) -> Array1D:
    """Return dimensionless residuals for the coupled `P` / `X_CO` solve.

    The state solve uses:

    - one strongest-line integrated-area equation
    - one strongest-line collisional FWHM equation

    The fit itself still optimizes the Voigt Lorentz HWHM because
    `scipy.special.voigt_profile` requires that parameterization. The state
    reduction converts that fitted width to FWHM before solving the pressure
    relation.
    """

    pressure_atm = float(state_vector[0])
    co_mole_fraction = float(state_vector[1])

    effective_broadening_fwhm = _effective_broadening_fwhm_vector(
        temperature_k,
        co_mole_fraction,
        transitions,
        broadening_scale=broadening_scale,
    )
    measured_fwhm = float(2.0 * collisional_hwhm_cm_inv[pressure_transition_index])
    modeled_fwhm = float(
        pressure_atm * effective_broadening_fwhm[pressure_transition_index]
    )
    fwhm_scale = max(abs(measured_fwhm), 1.0e-10)
    width_residual = np.array(
        [(modeled_fwhm - measured_fwhm) / fwhm_scale],
        dtype=float,
    )

    line_strength = line_strength_at_temperature(float(temperature_k), transition)
    modeled_area = line_strength * pressure_atm * co_mole_fraction * optical_path_length_cm
    area_scale = max(abs(float(integrated_area_cm_inv)), 1.0e-10)
    area_residual = np.array(
        [(modeled_area - float(integrated_area_cm_inv)) / area_scale],
        dtype=float,
    )

    return np.concatenate([width_residual, area_residual])


def _state_vector_from_estimate(state: StateEstimate) -> Array1D:
    """Pack one state estimate into `[temperature, pressure, X_CO]`."""

    return np.array(
        [
            float(state.temperature_k),
            float(state.pressure_atm),
            float(state.co_mole_fraction),
        ],
        dtype=float,
    )


def pressure_line_consistency_half_width(
    per_line_pressure_atm: Array1D,
    nominal_pressure_atm: float,
) -> float:
    """Return the largest informative line-to-line pressure deviation.

    Any `NaN` entries are ignored. With the current strongest-line-only state
    reduction this usually returns `NaN`, which the caller converts to zero.
    """

    finite_pressure = np.asarray(per_line_pressure_atm, dtype=float)
    finite_pressure = finite_pressure[np.isfinite(finite_pressure)]

    if finite_pressure.size < 2 or not np.isfinite(nominal_pressure_atm):
        return float("nan")

    return float(np.max(np.abs(finite_pressure - float(nominal_pressure_atm))))


def estimate_co_mole_fraction(
    temperature_k: Array1D,
    pressure_atm: Array1D,
    integrated_area_cm_inv: Array1D,
    *,
    optical_path_length_cm: float = DEFAULT_OPTICAL_PATH_LENGTH_CM,
    transition: Transition = DEFAULT_CO_TRANSITIONS[0],
) -> Array1D:
    """Estimate the CO mole fraction from integrated absorbance area."""

    co_mole_fraction = np.full_like(temperature_k, np.nan, dtype=float)

    for index, (temperature_value, pressure_value, area_value) in enumerate(
        zip(temperature_k, pressure_atm, integrated_area_cm_inv)
    ):
        if not (
            np.isfinite(temperature_value)
            and np.isfinite(pressure_value)
            and np.isfinite(area_value)
            and temperature_value > 0.0
            and pressure_value > 0.0
            and area_value > 0.0
        ):
            continue

        line_strength = line_strength_at_temperature(float(temperature_value), transition)
        denominator = float(pressure_value) * optical_path_length_cm * line_strength

        if denominator > 0.0:
            co_mole_fraction[index] = float(area_value / denominator)

    return co_mole_fraction


def solve_pressure_and_co_mole_fraction(
    temperature_k: float,
    collisional_hwhm_cm_inv: Array1D,
    integrated_area_cm_inv: float,
    *,
    optical_path_length_cm: float = DEFAULT_OPTICAL_PATH_LENGTH_CM,
    broadening_scale: float = DEFAULT_PRESSURE_BROADENING_SCALE,
    transitions: tuple[Transition, ...] = DEFAULT_CO_TRANSITIONS,
    transition: Transition = DEFAULT_CO_TRANSITIONS[0],
    max_iterations: int = DEFAULT_STATE_SOLVER_ITERATIONS,
) -> StateEstimate:
    """Solve the coupled pressure / CO-mole-fraction problem for one sweep.

    The current default pressure model follows the handout simplification that
    all non-CO collision partners may be treated as one N2-equivalent bath gas.
    This reduces the broadening model to a two-partner mixture:

    - `X_CO * gamma_CO(T)`
    - `(1 - X_CO) * gamma_bath(T)` with `gamma_bath == gamma_N2`

    Pressure and CO mole fraction are solved simultaneously from:

    - the strongest-transition collisional FWHM equation
    - the strongest-transition integrated-area equation

    The fit returns a Lorentz HWHM for each line because that is the parameter
    expected by `scipy.special.voigt_profile`, but the state reduction converts
    the fitted strongest-line width to FWHM and uses

    `Delta nu_C = 2 P * 0.84 * sum X_i gamma_i(T)`

    in the strongest-transition equation.

    No legacy background-composition model is retained in the active solver.
    """

    collisional_hwhm_cm_inv = np.asarray(collisional_hwhm_cm_inv, dtype=float)

    if not (
        np.isfinite(temperature_k)
        and temperature_k > 0.0
        and np.all(np.isfinite(collisional_hwhm_cm_inv))
        and np.all(collisional_hwhm_cm_inv > 0.0)
        and np.isfinite(integrated_area_cm_inv)
        and integrated_area_cm_inv > 0.0
    ):
        nan_vector = np.full(len(transitions), np.nan, dtype=float)
        return StateEstimate(
            temperature_k=float(temperature_k),
            pressure_atm=float("nan"),
            co_mole_fraction=float("nan"),
            per_line_pressure_atm=nan_vector,
        )

    pressure_transition_index = _pressure_transition_index(len(transitions))

    if pressure_transition_index is None:
        nan_vector = np.full(len(transitions), np.nan, dtype=float)
        return StateEstimate(
            temperature_k=float(temperature_k),
            pressure_atm=float("nan"),
            co_mole_fraction=float("nan"),
            per_line_pressure_atm=nan_vector,
        )

    co_mole_fraction_guess = float(
        np.clip(0.10, 1.0e-8, 0.999)
    )
    effective_broadening_fwhm_guess = _effective_broadening_fwhm_vector(
        float(temperature_k),
        co_mole_fraction_guess,
        transitions,
        broadening_scale=broadening_scale,
    )
    strongest_fwhm = float(2.0 * collisional_hwhm_cm_inv[pressure_transition_index])
    strongest_broadening_guess = float(
        effective_broadening_fwhm_guess[pressure_transition_index]
    )
    pressure_guess = float(max(strongest_fwhm / strongest_broadening_guess, 1.0e-8))
    line_strength = line_strength_at_temperature(float(temperature_k), transition)
    if np.isfinite(pressure_guess) and pressure_guess > 0.0:
        co_mole_fraction_guess = float(
            np.clip(
                integrated_area_cm_inv
                / (pressure_guess * optical_path_length_cm * line_strength),
                1.0e-8,
                0.999,
            )
        )
    pressure_guess = float(max(pressure_guess, 1.0e-8))

    result = least_squares(
        _state_residual_vector,
        x0=np.array([pressure_guess, co_mole_fraction_guess], dtype=float),
        bounds=(
            np.array([1.0e-8, 1.0e-8], dtype=float),
            np.array([1.0e3, 0.999], dtype=float),
        ),
        kwargs={
            "temperature_k": float(temperature_k),
            "collisional_hwhm_cm_inv": np.asarray(collisional_hwhm_cm_inv, dtype=float),
            "integrated_area_cm_inv": float(integrated_area_cm_inv),
            "optical_path_length_cm": float(optical_path_length_cm),
            "transitions": transitions,
            "transition": transition,
            "pressure_transition_index": pressure_transition_index,
            "broadening_scale": float(broadening_scale),
        },
        max_nfev=max(25, 5 * int(max_iterations)),
    )

    pressure_atm = float(result.x[0]) if result.success else float("nan")
    co_mole_fraction = float(result.x[1]) if result.success else float("nan")

    effective_broadening_fwhm = _effective_broadening_fwhm_vector(
        float(temperature_k),
        co_mole_fraction,
        transitions,
        broadening_scale=broadening_scale,
    )
    per_line_pressure = np.full(len(transitions), np.nan, dtype=float)
    if (
        np.isfinite(effective_broadening_fwhm[pressure_transition_index])
        and effective_broadening_fwhm[pressure_transition_index] > 0.0
    ):
        per_line_pressure[pressure_transition_index] = (
            2.0 * collisional_hwhm_cm_inv[pressure_transition_index]
            / effective_broadening_fwhm[pressure_transition_index]
        )

    return StateEstimate(
        temperature_k=float(temperature_k),
        pressure_atm=float(pressure_atm),
        co_mole_fraction=float(co_mole_fraction),
        per_line_pressure_atm=np.asarray(per_line_pressure, dtype=float),
    )


def evaluate_state_from_fit_parameters(
    parameters: VoigtFitParameters,
    *,
    optical_path_length_cm: float = DEFAULT_OPTICAL_PATH_LENGTH_CM,
    broadening_scale: float = DEFAULT_PRESSURE_BROADENING_SCALE,
    transitions: tuple[Transition, ...] = DEFAULT_CO_TRANSITIONS,
    transition: Transition = DEFAULT_CO_TRANSITIONS[0],
) -> Array1D:
    """Return `[temperature, pressure, co_mole_fraction]` for one fit."""

    state = solve_pressure_and_co_mole_fraction(
        parameters.temperature_k,
        parameters.collisional_hwhm_cm_inv,
        parameters.line_areas[0],
        optical_path_length_cm=optical_path_length_cm,
        broadening_scale=broadening_scale,
        transitions=transitions,
        transition=transition,
    )
    return np.array([parameters.temperature_k, state.pressure_atm, state.co_mole_fraction], dtype=float)


def build_state_history(
    temperature_k: Array1D,
    collisional_hwhm_cm_inv: Array2D,
    strongest_line_area_cm_inv: Array1D,
    *,
    sweep_frequency_hz: float,
    optical_path_length_cm: float = DEFAULT_OPTICAL_PATH_LENGTH_CM,
    broadening_scale: float = DEFAULT_PRESSURE_BROADENING_SCALE,
    transition: Transition = DEFAULT_CO_TRANSITIONS[0],
    transitions: tuple[Transition, ...] = DEFAULT_CO_TRANSITIONS,
) -> StateHistory:
    """Build scan-by-scan temperature, pressure, and CO mole-fraction arrays."""

    scan_index = np.arange(temperature_k.size, dtype=float)
    scan_time_s = scan_index / sweep_frequency_hz
    pressure_atm = np.full_like(temperature_k, np.nan, dtype=float)
    co_mole_fraction = np.full_like(temperature_k, np.nan, dtype=float)

    for index, (temperature_value, width_value, area_value) in enumerate(
        zip(temperature_k, collisional_hwhm_cm_inv, strongest_line_area_cm_inv)
    ):
        state = solve_pressure_and_co_mole_fraction(
            float(temperature_value),
            np.asarray(width_value, dtype=float),
            float(area_value),
            optical_path_length_cm=optical_path_length_cm,
            broadening_scale=broadening_scale,
            transitions=transitions,
            transition=transition,
        )
        pressure_atm[index] = state.pressure_atm
        co_mole_fraction[index] = state.co_mole_fraction

    return StateHistory(
        scan_index=scan_index,
        scan_time_s=scan_time_s,
        temperature_k=np.asarray(temperature_k, dtype=float),
        pressure_atm=pressure_atm,
        co_mole_fraction=co_mole_fraction,
    )


def _with_perturbed_broadening(
    transitions: tuple[Transition, ...],
    species: str,
    *,
    gamma_scale: float = 1.0,
    exponent_scale: float = 1.0,
) -> tuple[Transition, ...]:
    """Return transitions with one collision partner perturbed uniformly."""

    perturbed_transitions: list[Transition] = []

    for transition in transitions:
        partner = transition.broadening_by_species[species]
        updated_partner = replace(
            partner,
            gamma_ref_cm_inv_atm=max(
                partner.gamma_ref_cm_inv_atm * gamma_scale,
                np.finfo(float).tiny,
            ),
            temperature_exponent=max(
                partner.temperature_exponent * exponent_scale,
                np.finfo(float).tiny,
            ),
        )
        updated_map = dict(transition.broadening_by_species)
        updated_map[species] = updated_partner
        perturbed_transitions.append(
            replace(
                transition,
                broadening_by_species=updated_map,
            )
        )

    return tuple(perturbed_transitions)


def _with_bath_gas_model_offsets(
    transitions: tuple[Transition, ...],
    *,
    gamma_sign: float = 0.0,
    exponent_sign: float = 0.0,
) -> tuple[Transition, ...]:
    """Return transitions with per-line bath-gas model offsets applied."""

    updated_transitions: list[Transition] = []

    for transition in transitions:
        gamma_half_width, exponent_half_width = bath_gas_model_half_widths(
            transition.broadening_by_species
        )
        partner = transition.broadening_by_species[BATH_GAS_REFERENCE_SPECIES]
        updated_partner = replace(
            partner,
            gamma_ref_cm_inv_atm=max(
                partner.gamma_ref_cm_inv_atm + gamma_sign * gamma_half_width,
                np.finfo(float).tiny,
            ),
            temperature_exponent=max(
                partner.temperature_exponent + exponent_sign * exponent_half_width,
                np.finfo(float).tiny,
            ),
        )
        updated_map = dict(transition.broadening_by_species)
        updated_map[BATH_GAS_REFERENCE_SPECIES] = updated_partner
        updated_transitions.append(replace(transition, broadening_by_species=updated_map))

    return tuple(updated_transitions)


def state_estimate_from_fit_parameters(
    parameters: VoigtFitParameters,
    *,
    optical_path_length_cm: float = DEFAULT_OPTICAL_PATH_LENGTH_CM,
    transitions: tuple[Transition, ...] = DEFAULT_CO_TRANSITIONS,
    transition: Transition = DEFAULT_CO_TRANSITIONS[0],
) -> StateEstimate:
    """Return the full state estimate for one fitted spectrum."""

    return solve_pressure_and_co_mole_fraction(
        parameters.temperature_k,
        parameters.collisional_hwhm_cm_inv,
        parameters.line_areas[0],
        optical_path_length_cm=optical_path_length_cm,
        transitions=transitions,
        transition=transition,
    )


def estimate_broadening_parameter_uncertainty(
    parameters: VoigtFitParameters,
    *,
    optical_path_length_cm: float = DEFAULT_OPTICAL_PATH_LENGTH_CM,
    transitions: tuple[Transition, ...] = DEFAULT_CO_TRANSITIONS,
    transition: Transition = DEFAULT_CO_TRANSITIONS[0],
) -> Array1D:
    """Estimate state uncertainty from the CO and bath-gas broadening values."""

    nominal_state = evaluate_state_from_fit_parameters(
        parameters,
        optical_path_length_cm=optical_path_length_cm,
        transitions=transitions,
        transition=transition,
    )

    if not np.all(np.isfinite(nominal_state)):
        return np.full(3, np.nan, dtype=float)

    contributions: list[Array1D] = []

    for species in ("CO", BATH_GAS_REFERENCE_SPECIES):
        gamma_scale = 1.0 + transitions[0].broadening_by_species[species].gamma_relative_uncertainty
        perturbed_gamma_transitions = _with_perturbed_broadening(
            transitions,
            species,
            gamma_scale=gamma_scale,
        )
        perturbed_gamma_state = evaluate_state_from_fit_parameters(
            parameters,
            optical_path_length_cm=optical_path_length_cm,
            transitions=perturbed_gamma_transitions,
            transition=perturbed_gamma_transitions[0],
        )
        contributions.append(np.abs(perturbed_gamma_state - nominal_state))

        exponent_scale = (
            1.0 + transitions[0].broadening_by_species[species].exponent_relative_uncertainty
        )
        perturbed_exponent_transitions = _with_perturbed_broadening(
            transitions,
            species,
            exponent_scale=exponent_scale,
        )
        perturbed_exponent_state = evaluate_state_from_fit_parameters(
            parameters,
            optical_path_length_cm=optical_path_length_cm,
            transitions=perturbed_exponent_transitions,
            transition=perturbed_exponent_transitions[0],
        )
        contributions.append(np.abs(perturbed_exponent_state - nominal_state))

    contribution_matrix = np.asarray(contributions, dtype=float)
    half_width = _root_sum_square(contribution_matrix)
    half_width[0] = 0.0
    return np.asarray(half_width, dtype=float)


def estimate_bath_gas_model_uncertainty(
    parameters: VoigtFitParameters,
    *,
    optical_path_length_cm: float = DEFAULT_OPTICAL_PATH_LENGTH_CM,
    transitions: tuple[Transition, ...] = DEFAULT_CO_TRANSITIONS,
    transition: Transition = DEFAULT_CO_TRANSITIONS[0],
) -> Array1D:
    """Estimate state uncertainty from the N2-equivalent bath-gas assumption."""

    nominal_state = state_estimate_from_fit_parameters(
        parameters,
        optical_path_length_cm=optical_path_length_cm,
        transitions=transitions,
        transition=transition,
    )
    nominal_vector = _state_vector_from_estimate(nominal_state)

    if not np.all(np.isfinite(nominal_vector)):
        return np.full(3, np.nan, dtype=float)

    deviations: list[Array1D] = []

    for gamma_sign, exponent_sign in (
        (+1.0, 0.0),
        (-1.0, 0.0),
        (0.0, +1.0),
        (0.0, -1.0),
    ):
        perturbed_transitions = _with_bath_gas_model_offsets(
            transitions,
            gamma_sign=gamma_sign,
            exponent_sign=exponent_sign,
        )
        perturbed_state = state_estimate_from_fit_parameters(
            parameters,
            optical_path_length_cm=optical_path_length_cm,
            transitions=perturbed_transitions,
            transition=perturbed_transitions[0],
        )
        deviations.append(np.abs(_state_vector_from_estimate(perturbed_state) - nominal_vector))

    half_width = np.nanmax(np.asarray(deviations, dtype=float), axis=0)
    half_width[0] = 0.0
    return np.asarray(half_width, dtype=float)


def estimate_line_consistency_uncertainty(
    parameters: VoigtFitParameters,
    *,
    optical_path_length_cm: float = DEFAULT_OPTICAL_PATH_LENGTH_CM,
    transitions: tuple[Transition, ...] = DEFAULT_CO_TRANSITIONS,
    transition: Transition = DEFAULT_CO_TRANSITIONS[0],
) -> Array1D:
    """Estimate uncertainty from disagreement among the line-by-line pressures."""

    state = state_estimate_from_fit_parameters(
        parameters,
        optical_path_length_cm=optical_path_length_cm,
        transitions=transitions,
        transition=transition,
    )

    if not np.isfinite(state.pressure_atm):
        return np.full(3, np.nan, dtype=float)

    pressure_half_width = pressure_line_consistency_half_width(
        state.per_line_pressure_atm,
        state.pressure_atm,
    )

    if not np.isfinite(pressure_half_width):
        return np.zeros(3, dtype=float)

    lower_pressure = max(state.pressure_atm - pressure_half_width, np.finfo(float).tiny)
    upper_pressure = state.pressure_atm + pressure_half_width
    lower_xco = _co_mole_fraction_from_pressure(
        parameters.temperature_k,
        lower_pressure,
        parameters.line_areas[0],
        optical_path_length_cm=optical_path_length_cm,
        transition=transition,
    )
    upper_xco = _co_mole_fraction_from_pressure(
        parameters.temperature_k,
        upper_pressure,
        parameters.line_areas[0],
        optical_path_length_cm=optical_path_length_cm,
        transition=transition,
    )
    xco_half_width = float(
        np.nanmax(
            np.abs(
                np.array([lower_xco, upper_xco], dtype=float)
                - float(state.co_mole_fraction)
            )
        )
    )
    return np.array([0.0, pressure_half_width, xco_half_width], dtype=float)


def estimate_state_model_uncertainty(
    parameters: VoigtFitParameters,
    *,
    optical_path_length_cm: float = DEFAULT_OPTICAL_PATH_LENGTH_CM,
    transitions: tuple[Transition, ...] = DEFAULT_CO_TRANSITIONS,
    transition: Transition = DEFAULT_CO_TRANSITIONS[0],
    include_broadening_parameters: bool = True,
    include_bath_gas_model: bool = False,
    include_line_consistency: bool = False,
) -> Array1D:
    """Estimate non-fit state uncertainty from model parameters and mismatch."""

    contributions: list[Array1D] = []

    if include_broadening_parameters:
        contributions.append(
            estimate_broadening_parameter_uncertainty(
                parameters,
                optical_path_length_cm=optical_path_length_cm,
                transitions=transitions,
                transition=transition,
            )
        )
    if include_bath_gas_model:
        contributions.append(
            estimate_bath_gas_model_uncertainty(
                parameters,
                optical_path_length_cm=optical_path_length_cm,
                transitions=transitions,
                transition=transition,
            )
        )
    if include_line_consistency:
        contributions.append(
            estimate_line_consistency_uncertainty(
                parameters,
                optical_path_length_cm=optical_path_length_cm,
                transitions=transitions,
                transition=transition,
            )
        )

    if not contributions:
        return np.zeros(3, dtype=float)

    contribution_matrix = np.asarray(contributions, dtype=float)
    return _root_sum_square(contribution_matrix)
