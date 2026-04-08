import numpy as np

from mock_lab.spectroscopy.tips import get_co_partition_sum
from mock_lab.spectroscopy.voigt import (
    DEFAULT_CO_TRANSITIONS,
    SECOND_RADIATION_CONSTANT_CM_K,
    T_REF_K,
    hitran_line_strength_to_cm2_atm,
    line_strength_at_temperature,
)


def test_hitran_line_strength_conversion_matches_standard_296k_factor() -> None:
    conversion = hitran_line_strength_to_cm2_atm(1.0, reference_temperature_k=T_REF_K)

    assert np.isclose(conversion, 2.4794e19, rtol=5.0e-5)


def test_line_strength_at_reference_temperature_returns_stored_reference_value() -> None:
    transition = DEFAULT_CO_TRANSITIONS[0]

    assert np.isclose(
        line_strength_at_temperature(T_REF_K, transition),
        transition.line_strength_ref,
        rtol=1.0e-12,
    )


def test_line_strength_temperature_scaling_includes_temperature_ratio() -> None:
    transition = DEFAULT_CO_TRANSITIONS[0]
    temperature_k = 1800.0
    expected_strength = transition.line_strength_ref * (
        get_co_partition_sum(T_REF_K) / get_co_partition_sum(temperature_k)
    )
    expected_strength *= T_REF_K / temperature_k
    expected_strength *= np.exp(
        -SECOND_RADIATION_CONSTANT_CM_K
        * transition.lower_state_energy_cm_inv
        * (1.0 / temperature_k - 1.0 / T_REF_K)
    )
    expected_strength *= (
        1.0 - np.exp(-SECOND_RADIATION_CONSTANT_CM_K * transition.center_cm_inv / temperature_k)
    ) / (
        1.0 - np.exp(-SECOND_RADIATION_CONSTANT_CM_K * transition.center_cm_inv / T_REF_K)
    )

    assert np.isclose(
        line_strength_at_temperature(temperature_k, transition),
        expected_strength,
        rtol=1.0e-12,
    )
