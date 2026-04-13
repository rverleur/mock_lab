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


@dataclass(frozen=True)
class MixtureCompositionModel:
    """Legacy endpoint non-CO background compositions.

    The active pressure-reduction code now uses the simpler handout-supported
    two-partner model implemented by
    `effective_bath_gas_broadening_coefficient_cm_inv_atm()`. These endpoint
    compositions are retained only as a legacy reference for the earlier
    species-resolved broadening experiments.
    """

    nominal_non_co_background: dict[str, float]
    alternate_non_co_background: dict[str, float]
    default_co_mole_fraction: float = 0.10


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


def normalize_species_fractions(
    species_fractions: dict[str, float],
    *,
    supported_species: tuple[str, ...] = COLLISION_PARTNERS,
) -> dict[str, float]:
    """Return a normalized species-fraction dictionary on the supported keys."""

    normalized = {
        species: float(species_fractions.get(species, 0.0))
        for species in supported_species
    }
    total = float(sum(normalized.values()))

    if total <= 0.0:
        raise ValueError("At least one species fraction must be positive.")

    return {species: value / total for species, value in normalized.items()}


def build_collision_partner_mole_fractions(
    non_co_background: dict[str, float],
    *,
    co_mole_fraction: float | None = None,
    default_co_mole_fraction: float = 0.10,
) -> dict[str, float]:
    """Return a full collision-partner mixture including CO.

    The non-CO background shares are normalized, then scaled to occupy the
    remaining mixture fraction after the fitted CO mole fraction is inserted.

    This helper is retained for legacy mixture-model experiments and is not
    used by the active bath-gas pressure reduction.
    """

    co_value = default_co_mole_fraction if co_mole_fraction is None else float(co_mole_fraction)
    co_value = float(np.clip(co_value, 0.0, 0.999))

    background_species = {
        species: value
        for species, value in non_co_background.items()
        if species != "CO"
    }
    background = normalize_species_fractions(background_species)
    remaining_fraction = max(1.0 - co_value, 0.0)
    composition = {
        species: remaining_fraction * background.get(species, 0.0)
        for species in COLLISION_PARTNERS
    }
    composition["CO"] = co_value
    return composition


def interpolate_background_composition(
    composition_model: MixtureCompositionModel,
    interpolation_fraction: float,
) -> dict[str, float]:
    """Interpolate between nominal and alternate non-CO background mixtures.

    This helper is retained for legacy mixture-model experiments and is not
    used by the active bath-gas pressure reduction.
    """

    nominal = normalize_species_fractions(
        composition_model.nominal_non_co_background,
        supported_species=tuple(species for species in COLLISION_PARTNERS if species != "CO"),
    )
    alternate = normalize_species_fractions(
        composition_model.alternate_non_co_background,
        supported_species=tuple(species for species in COLLISION_PARTNERS if species != "CO"),
    )
    alpha = float(np.clip(interpolation_fraction, 0.0, 1.0))
    interpolated = {
        species: (1.0 - alpha) * nominal[species] + alpha * alternate[species]
        for species in nominal
    }
    return normalize_species_fractions(
        interpolated,
        supported_species=tuple(species for species in COLLISION_PARTNERS if species != "CO"),
    )


# These editable defaults are retained only for legacy experiments with the
# earlier mixture-model pressure reduction. They are not used by the active
# bath-gas state solver.
DEFAULT_MIXTURE_COMPOSITION_MODEL = MixtureCompositionModel(
    nominal_non_co_background={
        "N2": 0.68,
        "CO2": 0.08,
        "H2O": 0.16,
        "OH": 0.02,
        "H2": 0.03,
        "O2": 0.015,
        "O": 0.01,
        "H": 0.005,
    },
    alternate_non_co_background={
        "N2": 0.71,
        "CO2": 0.095,
        "H2O": 0.17,
        "OH": 0.005,
        "H2": 0.01,
        "O2": 0.005,
        "O": 0.003,
        "H": 0.002,
    },
    default_co_mole_fraction=0.10,
)
