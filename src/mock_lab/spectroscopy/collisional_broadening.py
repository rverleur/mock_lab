"""Species-resolved collisional-broadening helpers for CO pressure reduction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


BATH_GAS_REFERENCE_SPECIES = "N2"
COLLISION_PARTNERS: tuple[str, ...] = (
    "N2",
    "CO",
    "CO2",
    "H2O",
    "OH",
    "H2",
    "O2",
    "O",
    "H",
)
NON_CO_COLLISION_PARTNERS: tuple[str, ...] = tuple(
    species for species in COLLISION_PARTNERS if species != "CO"
)

DEFAULT_BROADENING_PARAMETER_UNCERTAINTY_FRACTION = 0.05


@dataclass(frozen=True)
class CollisionPartnerBroadening:
    """Reference broadening data for one CO-collision partner pair."""

    gamma_ref_cm_inv_atm: float
    temperature_exponent: float
    gamma_relative_uncertainty: float = DEFAULT_BROADENING_PARAMETER_UNCERTAINTY_FRACTION
    exponent_relative_uncertainty: float = DEFAULT_BROADENING_PARAMETER_UNCERTAINTY_FRACTION


def effective_bath_gas_broadening_coefficient_cm_inv_atm(
    broadening_by_species: dict[str, CollisionPartnerBroadening],
    temperature_k: float,
    co_mole_fraction: float,
) -> float:
    """Return the two-partner effective HWHM coefficient.

    The handout permits the non-CO collision partners to be treated as one
    bath gas with N2-equivalent broadening parameters. This helper therefore
    collapses the mixture to:

    - `X_CO * gamma_CO(T)`
    - `(1 - X_CO) * gamma_bath(T)` with `gamma_bath == gamma_N2`
    """

    if not np.isfinite(temperature_k) or temperature_k <= 0.0:
        return float("nan")

    co_value = float(np.clip(co_mole_fraction, 0.0, 0.999))
    bath_value = max(1.0 - co_value, 0.0)
    co_partner = broadening_by_species["CO"]
    bath_partner = broadening_by_species[BATH_GAS_REFERENCE_SPECIES]
    co_term = (
        co_value
        * co_partner.gamma_ref_cm_inv_atm
        * (296.0 / float(temperature_k)) ** co_partner.temperature_exponent
    )
    bath_term = (
        bath_value
        * bath_partner.gamma_ref_cm_inv_atm
        * (296.0 / float(temperature_k)) ** bath_partner.temperature_exponent
    )
    return float(co_term + bath_term)


def effective_bath_gas_broadening_fwhm_coefficient_cm_inv_atm(
    broadening_by_species: dict[str, CollisionPartnerBroadening],
    temperature_k: float,
    co_mole_fraction: float,
) -> float:
    """Return the two-partner effective collisional FWHM coefficient.

    The table values are treated as Lorentz HWHM coefficients. The pressure
    reduction now works in the full-width-at-half-maximum convention, so this
    helper simply returns `2 * gamma_eff`.
    """

    return float(
        2.0
        * effective_bath_gas_broadening_coefficient_cm_inv_atm(
            broadening_by_species,
            temperature_k,
            co_mole_fraction,
        )
    )


def bath_gas_model_half_widths(
    broadening_by_species: dict[str, CollisionPartnerBroadening],
) -> tuple[float, float]:
    """Return conservative bath-gas half-widths from the non-CO partner spread.

    The current pressure model treats every non-CO collision partner as
    N2-equivalent. The uncertainty from that simplification is bounded here by
    the full spread of the tabulated non-CO coefficients and exponents around
    the N2 reference values.
    """

    bath_partner = broadening_by_species[BATH_GAS_REFERENCE_SPECIES]
    gamma_half_width = max(
        abs(broadening_by_species[species].gamma_ref_cm_inv_atm - bath_partner.gamma_ref_cm_inv_atm)
        for species in NON_CO_COLLISION_PARTNERS
    )
    exponent_half_width = max(
        abs(broadening_by_species[species].temperature_exponent - bath_partner.temperature_exponent)
        for species in NON_CO_COLLISION_PARTNERS
    )
    return float(gamma_half_width), float(exponent_half_width)
