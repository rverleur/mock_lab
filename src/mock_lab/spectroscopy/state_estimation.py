"""State-estimation utilities derived from the fitted absorbance spectra."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import NDArray

from mock_lab.spectroscopy.collisional_broadening import (
    BATH_GAS_REFERENCE_SPECIES,
    DEFAULT_MIXTURE_COMPOSITION_MODEL,
    MixtureCompositionModel,
    bath_gas_model_half_widths,
    effective_bath_gas_broadening_coefficient_cm_inv_atm,
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
DEFAULT_PRESSURE_BROADENING_SCALE = 1.0
DEFAULT_STATE_SOLVER_ITERATIONS = 10


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
    effective_broadening_cm_inv_atm: Array1D
    collision_partner_mole_fractions: dict[str, float]


def _co_mole_fraction_from_pressure(
    temperature_k: float,
    pressure_atm: float,
    integrated_area_cm_inv: float,
    *,
    optical_path_length_cm: float,
    transition: Transition,
) -> float:
    """Return the CO mole fraction implied by one pressure estimate."""

    value = estimate_co_mole_fraction(
        np.array([temperature_k], dtype=float),
        np.array([pressure_atm], dtype=float),
        np.array([integrated_area_cm_inv], dtype=float),
        optical_path_length_cm=optical_path_length_cm,
        transition=transition,
    )[0]
    return float(value)


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
    """Return the largest line-to-line pressure deviation from the nominal value."""

    finite_pressure = np.asarray(per_line_pressure_atm, dtype=float)
    finite_pressure = finite_pressure[np.isfinite(finite_pressure)]

    if finite_pressure.size < 2 or not np.isfinite(nominal_pressure_atm):
        return float("nan")

    return float(np.max(np.abs(finite_pressure - float(nominal_pressure_atm))))


def corrected_pressure_from_broadening(
    apparent_pressure_atm: Array1D,
    broadening_scale: float = DEFAULT_PRESSURE_BROADENING_SCALE,
) -> Array1D:
    """Return the reported pressure from the composition-based broadening model."""

    pressure_atm = np.full_like(apparent_pressure_atm, np.nan, dtype=float)
    valid = np.isfinite(apparent_pressure_atm) & (apparent_pressure_atm > 0.0)
    pressure_atm[valid] = apparent_pressure_atm[valid] / broadening_scale
    return pressure_atm


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
    transitions: tuple[Transition, ...] = DEFAULT_CO_TRANSITIONS,
    transition: Transition = DEFAULT_CO_TRANSITIONS[0],
    composition_model: MixtureCompositionModel = DEFAULT_MIXTURE_COMPOSITION_MODEL,
    max_iterations: int = DEFAULT_STATE_SOLVER_ITERATIONS,
) -> StateEstimate:
    """Solve the coupled pressure / CO-mole-fraction problem for one sweep.

    The current default pressure model follows the handout simplification that
    all non-CO collision partners may be treated as one N2-equivalent bath gas.
    This reduces the broadening model to a two-partner mixture:

    - `X_CO * gamma_CO(T)`
    - `(1 - X_CO) * gamma_bath(T)` with `gamma_bath == gamma_N2`

    The legacy background-composition arguments are retained for API
    compatibility but do not alter the active pressure solve.
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
            effective_broadening_cm_inv_atm=nan_vector,
            collision_partner_mole_fractions={"CO": float("nan"), BATH_GAS_REFERENCE_SPECIES: float("nan")},
        )

    co_mole_fraction = float(np.clip(composition_model.default_co_mole_fraction, 0.0, 0.999))

    for _ in range(max_iterations):
        effective_broadening = np.asarray(
            [
                effective_bath_gas_broadening_coefficient_cm_inv_atm(
                    transition_value.broadening_by_species,
                    float(temperature_k),
                    co_mole_fraction,
                )
                for transition_value in transitions
            ],
            dtype=float,
        )
        per_line_pressure = np.full(len(transitions), np.nan, dtype=float)
        valid = np.isfinite(effective_broadening) & (effective_broadening > 0.0)
        per_line_pressure[valid] = collisional_hwhm_cm_inv[valid] / effective_broadening[valid]
        pressure_atm = float(np.nanmean(per_line_pressure))

        if not np.isfinite(pressure_atm) or pressure_atm <= 0.0:
            break

        line_strength = line_strength_at_temperature(float(temperature_k), transition)
        updated_co_mole_fraction = float(
            np.clip(
                integrated_area_cm_inv / (pressure_atm * optical_path_length_cm * line_strength),
                0.0,
                0.999,
            )
        )

        if abs(updated_co_mole_fraction - co_mole_fraction) < 1.0e-10:
            co_mole_fraction = updated_co_mole_fraction
            break

        co_mole_fraction = updated_co_mole_fraction

    collision_partner_mole_fractions = {
        BATH_GAS_REFERENCE_SPECIES: max(1.0 - co_mole_fraction, 0.0),
        "CO": co_mole_fraction,
    }
    effective_broadening = np.asarray(
        [
            effective_bath_gas_broadening_coefficient_cm_inv_atm(
                transition_value.broadening_by_species,
                float(temperature_k),
                co_mole_fraction,
            )
            for transition_value in transitions
        ],
        dtype=float,
    )
    per_line_pressure = np.full(len(transitions), np.nan, dtype=float)
    valid = np.isfinite(effective_broadening) & (effective_broadening > 0.0)
    per_line_pressure[valid] = collisional_hwhm_cm_inv[valid] / effective_broadening[valid]
    pressure_atm = float(np.nanmean(per_line_pressure))
    co_mole_fraction = _co_mole_fraction_from_pressure(
        float(temperature_k),
        float(pressure_atm),
        float(integrated_area_cm_inv),
        optical_path_length_cm=optical_path_length_cm,
        transition=transition,
    )

    return StateEstimate(
        temperature_k=float(temperature_k),
        pressure_atm=float(pressure_atm),
        co_mole_fraction=float(co_mole_fraction),
        per_line_pressure_atm=np.asarray(per_line_pressure, dtype=float),
        effective_broadening_cm_inv_atm=np.asarray(effective_broadening, dtype=float),
        collision_partner_mole_fractions=collision_partner_mole_fractions,
    )


def evaluate_state_from_fit_parameters(
    parameters: VoigtFitParameters,
    *,
    optical_path_length_cm: float = DEFAULT_OPTICAL_PATH_LENGTH_CM,
    broadening_scale: float = DEFAULT_PRESSURE_BROADENING_SCALE,
    transitions: tuple[Transition, ...] = DEFAULT_CO_TRANSITIONS,
    transition: Transition = DEFAULT_CO_TRANSITIONS[0],
    composition_model: MixtureCompositionModel = DEFAULT_MIXTURE_COMPOSITION_MODEL,
) -> Array1D:
    """Return `[temperature, pressure, co_mole_fraction]` for one fit."""

    state = solve_pressure_and_co_mole_fraction(
        parameters.temperature_k,
        parameters.collisional_hwhm_cm_inv,
        parameters.line_areas[0],
        optical_path_length_cm=optical_path_length_cm,
        transitions=transitions,
        transition=transition,
        composition_model=composition_model,
    )
    reported_pressure = corrected_pressure_from_broadening(
        np.array([state.pressure_atm], dtype=float),
        broadening_scale=broadening_scale,
    )[0]
    return np.array([parameters.temperature_k, reported_pressure, state.co_mole_fraction], dtype=float)


def build_state_history(
    temperature_k: Array1D,
    collisional_hwhm_cm_inv: Array2D,
    strongest_line_area_cm_inv: Array1D,
    *,
    sweep_frequency_hz: float,
    optical_path_length_cm: float = DEFAULT_OPTICAL_PATH_LENGTH_CM,
    transition: Transition = DEFAULT_CO_TRANSITIONS[0],
    transitions: tuple[Transition, ...] = DEFAULT_CO_TRANSITIONS,
    composition_model: MixtureCompositionModel = DEFAULT_MIXTURE_COMPOSITION_MODEL,
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
            transitions=transitions,
            transition=transition,
            composition_model=composition_model,
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


def _with_absolute_broadening_offset(
    transitions: tuple[Transition, ...],
    species: str,
    *,
    gamma_offset_cm_inv_atm: float = 0.0,
    exponent_offset: float = 0.0,
) -> tuple[Transition, ...]:
    """Return transitions with one collision partner shifted by absolute offsets."""

    updated_transitions: list[Transition] = []

    for transition in transitions:
        partner = transition.broadening_by_species[species]
        updated_partner = replace(
            partner,
            gamma_ref_cm_inv_atm=max(
                partner.gamma_ref_cm_inv_atm + gamma_offset_cm_inv_atm,
                np.finfo(float).tiny,
            ),
            temperature_exponent=max(
                partner.temperature_exponent + exponent_offset,
                np.finfo(float).tiny,
            ),
        )
        updated_map = dict(transition.broadening_by_species)
        updated_map[species] = updated_partner
        updated_transitions.append(replace(transition, broadening_by_species=updated_map))

    return tuple(updated_transitions)


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
    composition_model: MixtureCompositionModel = DEFAULT_MIXTURE_COMPOSITION_MODEL,
) -> StateEstimate:
    """Return the full state estimate for one fitted spectrum."""

    return solve_pressure_and_co_mole_fraction(
        parameters.temperature_k,
        parameters.collisional_hwhm_cm_inv,
        parameters.line_areas[0],
        optical_path_length_cm=optical_path_length_cm,
        transitions=transitions,
        transition=transition,
        composition_model=composition_model,
    )


def estimate_broadening_parameter_uncertainty(
    parameters: VoigtFitParameters,
    *,
    optical_path_length_cm: float = DEFAULT_OPTICAL_PATH_LENGTH_CM,
    transitions: tuple[Transition, ...] = DEFAULT_CO_TRANSITIONS,
    transition: Transition = DEFAULT_CO_TRANSITIONS[0],
    composition_model: MixtureCompositionModel = DEFAULT_MIXTURE_COMPOSITION_MODEL,
) -> Array1D:
    """Estimate state uncertainty from the CO and bath-gas broadening values."""

    nominal_state = evaluate_state_from_fit_parameters(
        parameters,
        optical_path_length_cm=optical_path_length_cm,
        transitions=transitions,
        transition=transition,
        composition_model=composition_model,
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
            composition_model=composition_model,
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
            composition_model=composition_model,
        )
        contributions.append(np.abs(perturbed_exponent_state - nominal_state))

    contribution_matrix = np.asarray(contributions, dtype=float)
    half_width = np.sqrt(np.nansum(contribution_matrix**2, axis=0))
    half_width[0] = 0.0
    return np.asarray(half_width, dtype=float)


def estimate_bath_gas_model_uncertainty(
    parameters: VoigtFitParameters,
    *,
    optical_path_length_cm: float = DEFAULT_OPTICAL_PATH_LENGTH_CM,
    transitions: tuple[Transition, ...] = DEFAULT_CO_TRANSITIONS,
    transition: Transition = DEFAULT_CO_TRANSITIONS[0],
    composition_model: MixtureCompositionModel = DEFAULT_MIXTURE_COMPOSITION_MODEL,
) -> Array1D:
    """Estimate state uncertainty from the N2-equivalent bath-gas assumption."""

    nominal_state = state_estimate_from_fit_parameters(
        parameters,
        optical_path_length_cm=optical_path_length_cm,
        transitions=transitions,
        transition=transition,
        composition_model=composition_model,
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
            composition_model=composition_model,
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
    composition_model: MixtureCompositionModel = DEFAULT_MIXTURE_COMPOSITION_MODEL,
) -> Array1D:
    """Estimate uncertainty from disagreement among the line-by-line pressures."""

    state = state_estimate_from_fit_parameters(
        parameters,
        optical_path_length_cm=optical_path_length_cm,
        transitions=transitions,
        transition=transition,
        composition_model=composition_model,
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
    composition_model: MixtureCompositionModel = DEFAULT_MIXTURE_COMPOSITION_MODEL,
    include_broadening_parameters: bool = True,
    include_bath_gas_model: bool = False,
    include_line_consistency: bool = True,
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
                composition_model=composition_model,
            )
        )
    if include_bath_gas_model:
        contributions.append(
            estimate_bath_gas_model_uncertainty(
                parameters,
                optical_path_length_cm=optical_path_length_cm,
                transitions=transitions,
                transition=transition,
                composition_model=composition_model,
            )
        )
    if include_line_consistency:
        contributions.append(
            estimate_line_consistency_uncertainty(
                parameters,
                optical_path_length_cm=optical_path_length_cm,
                transitions=transitions,
                transition=transition,
                composition_model=composition_model,
            )
        )

    if not contributions:
        return np.zeros(3, dtype=float)

    contribution_matrix = np.asarray(contributions, dtype=float)
    return np.sqrt(np.nansum(contribution_matrix**2, axis=0))


def estimate_broadening_model_uncertainty(
    parameters: VoigtFitParameters,
    *,
    optical_path_length_cm: float = DEFAULT_OPTICAL_PATH_LENGTH_CM,
    transitions: tuple[Transition, ...] = DEFAULT_CO_TRANSITIONS,
    transition: Transition = DEFAULT_CO_TRANSITIONS[0],
    composition_model: MixtureCompositionModel = DEFAULT_MIXTURE_COMPOSITION_MODEL,
) -> Array1D:
    """Backward-compatible wrapper for the non-fit state uncertainty estimate."""

    return estimate_state_model_uncertainty(
        parameters,
        optical_path_length_cm=optical_path_length_cm,
        transitions=transitions,
        transition=transition,
        composition_model=composition_model,
        include_broadening_parameters=True,
        include_bath_gas_model=False,
        include_line_consistency=True,
    )
