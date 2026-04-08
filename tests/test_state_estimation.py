import numpy as np

from mock_lab.spectroscopy.state_estimation import (
    DEFAULT_OPTICAL_PATH_LENGTH_CM,
    build_state_history,
    corrected_pressure_from_broadening,
    estimate_co_mole_fraction,
)
from mock_lab.spectroscopy.voigt import DEFAULT_CO_TRANSITIONS, line_strength_at_temperature


def test_corrected_pressure_applies_handout_scaling() -> None:
    apparent_pressure = np.array([0.84, 1.68, np.nan])
    corrected_pressure = corrected_pressure_from_broadening(apparent_pressure)

    assert np.allclose(corrected_pressure[:2], np.array([1.0, 2.0]))
    assert np.isnan(corrected_pressure[2])


def test_estimate_co_mole_fraction_matches_pressure_normalized_area_relation() -> None:
    temperature_k = np.array([2000.0, 2200.0])
    pressure_atm = np.array([1.0, 1.2])
    expected_co_mole_fraction = np.array([0.18, 0.22])
    integrated_area_cm_inv = np.array(
        [
            line_strength_at_temperature(float(temperature_k[0]), DEFAULT_CO_TRANSITIONS[0])
            * pressure_atm[0]
            * expected_co_mole_fraction[0]
            * DEFAULT_OPTICAL_PATH_LENGTH_CM,
            line_strength_at_temperature(float(temperature_k[1]), DEFAULT_CO_TRANSITIONS[0])
            * pressure_atm[1]
            * expected_co_mole_fraction[1]
            * DEFAULT_OPTICAL_PATH_LENGTH_CM,
        ],
        dtype=float,
    )

    co_mole_fraction = estimate_co_mole_fraction(
        temperature_k,
        pressure_atm,
        integrated_area_cm_inv,
        optical_path_length_cm=DEFAULT_OPTICAL_PATH_LENGTH_CM,
    )

    assert np.allclose(co_mole_fraction, expected_co_mole_fraction)


def test_build_state_history_constructs_scan_axis() -> None:
    state_history = build_state_history(
        np.array([1800.0, 1900.0, np.nan]),
        np.array([0.84, 1.68, np.nan]),
        np.array([0.004, 0.005, np.nan]),
        sweep_frequency_hz=300000.0,
    )

    assert np.allclose(state_history.scan_index, np.array([0.0, 1.0, 2.0]))
    assert np.isclose(state_history.scan_time_s[1], 1.0 / 300000.0)
    assert np.allclose(state_history.pressure_atm[:2], np.array([1.0, 2.0]))
