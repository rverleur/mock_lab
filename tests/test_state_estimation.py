import numpy as np

from mock_lab.spectroscopy.collisional_broadening import (
    effective_bath_gas_broadening_coefficient_cm_inv_atm,
)
from mock_lab.spectroscopy.state_estimation import (
    DEFAULT_OPTICAL_PATH_LENGTH_CM,
    DEFAULT_PRESSURE_BROADENING_SCALE,
    build_state_history,
    corrected_pressure_from_broadening,
    estimate_co_mole_fraction,
    estimate_line_consistency_uncertainty,
    estimate_state_model_uncertainty,
    solve_pressure_and_co_mole_fraction,
)
from mock_lab.spectroscopy.voigt import (
    DEFAULT_CO_TRANSITIONS,
    VoigtFitParameters,
    line_strength_at_temperature,
)


def test_corrected_pressure_applies_handout_scale() -> None:
    apparent_pressure = np.array([1.0, 2.0, np.nan])
    corrected_pressure = corrected_pressure_from_broadening(apparent_pressure)

    assert np.allclose(
        corrected_pressure[:2],
        apparent_pressure[:2] / DEFAULT_PRESSURE_BROADENING_SCALE,
    )
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
    expected_temperature = np.array([1800.0, 1900.0, np.nan])
    expected_pressure = np.array([1.0, 2.0, np.nan])
    expected_co_mole_fraction = np.array([0.12, 0.18, np.nan])
    collisional_hwhm = np.full((3, len(DEFAULT_CO_TRANSITIONS)), np.nan, dtype=float)

    for index in range(2):
        collisional_hwhm[index] = np.asarray(
            [
                expected_pressure[index]
                * DEFAULT_PRESSURE_BROADENING_SCALE
                * effective_bath_gas_broadening_coefficient_cm_inv_atm(
                    transition.broadening_by_species,
                    float(expected_temperature[index]),
                    float(expected_co_mole_fraction[index]),
                )
                for transition in DEFAULT_CO_TRANSITIONS
            ],
            dtype=float,
        )
    # The strongest transition now drives the pressure / X_CO solve directly,
    # so weaker-line widths should not affect the recovered state.
    collisional_hwhm[0, 1] *= 0.15
    collisional_hwhm[1, 1] *= 3.5
    collisional_hwhm[0, 2] *= 3.0
    collisional_hwhm[1, 2] *= 0.25
    strongest_line_area = np.array(
        [
            line_strength_at_temperature(float(expected_temperature[0]), DEFAULT_CO_TRANSITIONS[0])
            * expected_pressure[0]
            * expected_co_mole_fraction[0]
            * DEFAULT_OPTICAL_PATH_LENGTH_CM,
            line_strength_at_temperature(float(expected_temperature[1]), DEFAULT_CO_TRANSITIONS[0])
            * expected_pressure[1]
            * expected_co_mole_fraction[1]
            * DEFAULT_OPTICAL_PATH_LENGTH_CM,
            np.nan,
        ],
        dtype=float,
    )
    state_history = build_state_history(
        expected_temperature,
        collisional_hwhm,
        strongest_line_area,
        sweep_frequency_hz=300000.0,
    )

    assert np.allclose(state_history.scan_index, np.array([0.0, 1.0, 2.0]))
    assert np.isclose(state_history.scan_time_s[1], 1.0 / 300000.0)
    assert np.allclose(state_history.pressure_atm[:2], expected_pressure[:2], rtol=1.0e-6)
    assert np.allclose(
        state_history.co_mole_fraction[:2],
        expected_co_mole_fraction[:2],
        rtol=1.0e-6,
    )


def test_solve_pressure_and_co_mole_fraction_recovers_synthetic_state() -> None:
    expected_temperature = 1950.0
    expected_pressure = 1.85
    expected_co_mole_fraction = 0.21
    collisional_hwhm = np.asarray(
        [
            expected_pressure
            * DEFAULT_PRESSURE_BROADENING_SCALE
            * effective_bath_gas_broadening_coefficient_cm_inv_atm(
                transition.broadening_by_species,
                expected_temperature,
                expected_co_mole_fraction,
            )
            for transition in DEFAULT_CO_TRANSITIONS
        ],
        dtype=float,
    )
    strongest_line_area = (
        line_strength_at_temperature(expected_temperature, DEFAULT_CO_TRANSITIONS[0])
        * expected_pressure
        * expected_co_mole_fraction
        * DEFAULT_OPTICAL_PATH_LENGTH_CM
    )

    state = solve_pressure_and_co_mole_fraction(
        expected_temperature,
        collisional_hwhm,
        strongest_line_area,
    )

    assert np.isclose(state.pressure_atm, expected_pressure, rtol=1.0e-6)
    assert np.isclose(state.co_mole_fraction, expected_co_mole_fraction, rtol=1.0e-6)


def test_line_consistency_uncertainty_is_zero_with_strongest_line_only_pressure() -> None:
    parameters = VoigtFitParameters(
        temperature_k=1800.0,
        line_centers_relative_cm_inv=np.array([0.0, -0.10, 0.03], dtype=float),
        collisional_hwhm_cm_inv=np.array([0.050, 0.020, 0.020], dtype=float),
        line_areas=np.array([0.010, 0.002, 0.001], dtype=float),
        baseline_offset=0.0,
        baseline_slope=0.0,
    )

    half_width = estimate_line_consistency_uncertainty(parameters)

    assert np.allclose(half_width, np.zeros(3))


def test_line_consistency_uncertainty_ignores_weaker_line_widths() -> None:
    expected_temperature = 1800.0
    expected_pressure = 1.4
    expected_co_mole_fraction = 0.16
    line_widths = np.asarray(
        [
            expected_pressure
            * DEFAULT_PRESSURE_BROADENING_SCALE
            * effective_bath_gas_broadening_coefficient_cm_inv_atm(
                transition.broadening_by_species,
                expected_temperature,
                expected_co_mole_fraction,
            )
            for transition in DEFAULT_CO_TRANSITIONS
        ],
        dtype=float,
    )
    line_widths[1] *= 0.1
    line_widths[2] *= 4.0
    line_area = (
        line_strength_at_temperature(expected_temperature, DEFAULT_CO_TRANSITIONS[0])
        * expected_pressure
        * expected_co_mole_fraction
        * DEFAULT_OPTICAL_PATH_LENGTH_CM
    )
    parameters = VoigtFitParameters(
        temperature_k=expected_temperature,
        line_centers_relative_cm_inv=np.array([0.0, -0.10, 0.03], dtype=float),
        collisional_hwhm_cm_inv=line_widths,
        line_areas=np.array([line_area, line_area * 0.05, line_area * 0.02], dtype=float),
        baseline_offset=0.0,
        baseline_slope=0.0,
    )

    half_width = estimate_line_consistency_uncertainty(parameters)

    assert np.allclose(half_width, np.zeros(3))


def test_state_model_uncertainty_remains_finite_for_large_pressure_mismatch() -> None:
    parameters = VoigtFitParameters(
        temperature_k=1750.0,
        line_centers_relative_cm_inv=np.array([0.0, -0.10, 0.03], dtype=float),
        collisional_hwhm_cm_inv=np.array([0.15, 1.0e-6, 1.0e-6], dtype=float),
        line_areas=np.array([0.01, 5.0e-4, 1.0e-4], dtype=float),
        baseline_offset=0.0,
        baseline_slope=0.0,
    )

    half_width = estimate_state_model_uncertainty(parameters)

    assert np.all(np.isfinite(half_width))
