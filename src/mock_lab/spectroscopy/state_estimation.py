"""State-estimation utilities derived from the fitted absorbance spectra."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from mock_lab.spectroscopy.tips import get_co_partition_sum
from mock_lab.spectroscopy.voigt import DEFAULT_CO_TRANSITIONS, T_REF_K, Transition


Array1D = NDArray[np.float64]

SECOND_RADIATION_CONSTANT_CM_K = 1.438776877
IDEAL_GAS_NUMBER_DENSITY_COEFF_CM3 = 101325.0 / 1.380649e-23 / 1.0e6
DEFAULT_OPTICAL_PATH_LENGTH_CM = 10.32
DEFAULT_PRESSURE_BROADENING_SCALE = 0.84


@dataclass(frozen=True)
class StateHistory:
    """Scan-by-scan thermochemical estimates derived from the Voigt fits."""

    scan_index: Array1D
    scan_time_s: Array1D
    temperature_k: Array1D
    pressure_atm: Array1D
    co_mole_fraction: Array1D


def co_partition_function_ratio(temperature_k: float) -> float:
    """Return the CO partition-function ratio Q(T)/Q(T_ref) from local TIPS data."""

    return float(get_co_partition_sum(float(temperature_k)) / get_co_partition_sum(float(T_REF_K)))


def line_strength_at_temperature(
    temperature_k: float,
    transition: Transition,
) -> float:
    """Approximate the transition line strength at temperature `T`."""

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


def corrected_pressure_from_broadening(
    mean_apparent_pressure_atm: Array1D,
    broadening_scale: float = DEFAULT_PRESSURE_BROADENING_SCALE,
) -> Array1D:
    """Convert the fitted apparent pressure into the handout-corrected pressure."""

    pressure_atm = np.full_like(mean_apparent_pressure_atm, np.nan, dtype=float)
    valid = np.isfinite(mean_apparent_pressure_atm) & (mean_apparent_pressure_atm > 0.0)
    pressure_atm[valid] = mean_apparent_pressure_atm[valid] / broadening_scale
    return pressure_atm


def estimate_co_mole_fraction(
    temperature_k: Array1D,
    pressure_atm: Array1D,
    integrated_area_cm_inv: Array1D,
    *,
    optical_path_length_cm: float = DEFAULT_OPTICAL_PATH_LENGTH_CM,
    transition: Transition = DEFAULT_CO_TRANSITIONS[0],
) -> Array1D:
    """Estimate the CO mole fraction from integrated absorbance area.

    The strongest transition is used by default because it is the most stable
    fit target across the current sweep stack.
    """

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
        total_number_density_cm3 = (
            IDEAL_GAS_NUMBER_DENSITY_COEFF_CM3 * float(pressure_value) / float(temperature_value)
        )
        denominator = total_number_density_cm3 * optical_path_length_cm * line_strength

        if denominator > 0.0:
            co_mole_fraction[index] = float(area_value / denominator)

    return co_mole_fraction


def build_state_history(
    temperature_k: Array1D,
    mean_apparent_pressure_atm: Array1D,
    strongest_line_area_cm_inv: Array1D,
    *,
    sweep_frequency_hz: float,
    optical_path_length_cm: float = DEFAULT_OPTICAL_PATH_LENGTH_CM,
) -> StateHistory:
    """Build scan-by-scan temperature, pressure, and CO mole-fraction arrays."""

    scan_index = np.arange(temperature_k.size, dtype=float)
    scan_time_s = scan_index / sweep_frequency_hz
    pressure_atm = corrected_pressure_from_broadening(mean_apparent_pressure_atm)
    co_mole_fraction = estimate_co_mole_fraction(
        np.asarray(temperature_k, dtype=float),
        pressure_atm,
        np.asarray(strongest_line_area_cm_inv, dtype=float),
        optical_path_length_cm=optical_path_length_cm,
    )

    return StateHistory(
        scan_index=scan_index,
        scan_time_s=scan_time_s,
        temperature_k=np.asarray(temperature_k, dtype=float),
        pressure_atm=pressure_atm,
        co_mole_fraction=co_mole_fraction,
    )
