import numpy as np

from mock_lab.pipelines.monte_carlo_state_history import sample_transitions_uniform
from mock_lab.spectroscopy.collisional_broadening import (
    BATH_GAS_REFERENCE_SPECIES,
    NON_CO_COLLISION_PARTNERS,
    bath_gas_model_half_widths,
)
from mock_lab.spectroscopy.voigt import DEFAULT_CO_TRANSITIONS


def test_sample_transitions_uniform_preserves_shape_and_changes_supported_fields() -> None:
    sampled_transitions = sample_transitions_uniform(
        np.random.default_rng(123),
        DEFAULT_CO_TRANSITIONS,
    )

    assert len(sampled_transitions) == len(DEFAULT_CO_TRANSITIONS)

    for nominal, sampled in zip(DEFAULT_CO_TRANSITIONS, sampled_transitions):
        assert sampled.label == nominal.label
        assert sampled.source_csv_row == nominal.source_csv_row
        assert sampled.center_cm_inv != nominal.center_cm_inv
        assert sampled.line_strength_ref > 0.0
        assert sampled.broadening_by_species["CO"].gamma_ref_cm_inv_atm > 0.0
        assert sampled.broadening_by_species[BATH_GAS_REFERENCE_SPECIES].gamma_ref_cm_inv_atm > 0.0
        for species in NON_CO_COLLISION_PARTNERS:
            assert (
                sampled.broadening_by_species[species].gamma_ref_cm_inv_atm
                == sampled.broadening_by_species[BATH_GAS_REFERENCE_SPECIES].gamma_ref_cm_inv_atm
            )


def test_sample_transitions_uniform_respects_documented_uncertainty_bounds() -> None:
    sampled_transitions = sample_transitions_uniform(
        np.random.default_rng(456),
        DEFAULT_CO_TRANSITIONS,
    )

    for nominal, sampled in zip(DEFAULT_CO_TRANSITIONS, sampled_transitions):
        assert abs(sampled.center_cm_inv - nominal.center_cm_inv) <= (
            nominal.uncertainties["wavenumber_cm_inv"].upper_bound_absolute + 1.0e-12
        )
        assert abs(sampled.line_strength_ref - nominal.line_strength_ref) <= (
            nominal.uncertainties["line_strength"].upper_bound_absolute + 1.0e-18
        )
        co_partner = nominal.broadening_by_species["CO"]
        sampled_co = sampled.broadening_by_species["CO"]
        assert abs(
            sampled_co.gamma_ref_cm_inv_atm - co_partner.gamma_ref_cm_inv_atm
        ) <= co_partner.gamma_ref_cm_inv_atm * co_partner.gamma_relative_uncertainty + 1.0e-12
        assert abs(
            sampled_co.temperature_exponent - co_partner.temperature_exponent
        ) <= co_partner.temperature_exponent * co_partner.exponent_relative_uncertainty + 1.0e-12

        bath_partner = nominal.broadening_by_species[BATH_GAS_REFERENCE_SPECIES]
        sampled_bath = sampled.broadening_by_species[BATH_GAS_REFERENCE_SPECIES]
        bath_gamma_half_width, bath_exponent_half_width = bath_gas_model_half_widths(
            nominal.broadening_by_species
        )
        assert abs(
            sampled_bath.gamma_ref_cm_inv_atm - bath_partner.gamma_ref_cm_inv_atm
        ) <= np.sqrt(
            (bath_partner.gamma_ref_cm_inv_atm * bath_partner.gamma_relative_uncertainty) ** 2
            + bath_gamma_half_width**2
        ) + 1.0e-12
        assert abs(
            sampled_bath.temperature_exponent - bath_partner.temperature_exponent
        ) <= np.sqrt(
            (bath_partner.temperature_exponent * bath_partner.exponent_relative_uncertainty) ** 2
            + bath_exponent_half_width**2
        ) + 1.0e-12
