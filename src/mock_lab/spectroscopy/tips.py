"""Helpers for reading vendored HITRAN TIPS partition sums."""

from __future__ import annotations

from functools import lru_cache
import pickle
from pathlib import Path

import numpy as np


def _tips_root() -> Path:
    """Return the repository-local vendored TIPS directory."""

    return Path(__file__).resolve().parents[3] / "third_party" / "hitran_tips"


@lru_cache(maxsize=1)
def _molecular_data() -> dict:
    """Load the local TIPS molecular metadata file."""

    path = _tips_root() / "data" / "molecular_data.QTmol"

    with path.open("rb") as handle:
        return pickle.loads(handle.read())


@lru_cache(maxsize=32)
def _qt_table(molecule: str) -> tuple[dict, dict]:
    """Load one molecule's local TIPS partition-sum table and metadata."""

    molecular_data = _molecular_data()
    molecule_data = molecular_data[molecule]
    path = _tips_root() / "data" / f"{molecule_data['mol_id']}.QTpy"

    with path.open("rb") as handle:
        table = pickle.loads(handle.read())

    return molecule_data, table


def get_partition_sum(
    molecule: str,
    isotopologue: str,
    temperature_k: float,
) -> float:
    """Return the local TIPS partition sum Q(T) for one molecule/isotopologue."""

    molecule_data, table = _qt_table(molecule)

    if isotopologue not in molecule_data["list_iso"]:
        raise KeyError(f"Unknown isotopologue '{isotopologue}' for molecule '{molecule}'.")

    if not np.isfinite(temperature_k):
        raise ValueError("Partition-sum evaluation requires a finite temperature.")

    minimum_temperature_k = 1.0
    maximum_temperature_k = float(molecule_data[isotopologue]["Tmax"])

    if not minimum_temperature_k <= temperature_k <= maximum_temperature_k:
        raise ValueError(
            f"Temperature {temperature_k} K is outside the local TIPS range "
            f"[{minimum_temperature_k}, {maximum_temperature_k}] for {molecule} iso {isotopologue}."
        )

    lower_temperature_k = float(np.floor(temperature_k))
    upper_temperature_k = float(np.ceil(temperature_k))
    lower_value = float(table[isotopologue][lower_temperature_k])

    if upper_temperature_k == lower_temperature_k:
        return lower_value

    upper_value = float(table[isotopologue][upper_temperature_k])
    interpolation_fraction = (temperature_k - lower_temperature_k) / (
        upper_temperature_k - lower_temperature_k
    )
    return lower_value + (upper_value - lower_value) * interpolation_fraction


def get_co_partition_sum(temperature_k: float, isotopologue: str = "1") -> float:
    """Return the local TIPS partition sum for the main CO isotopologue."""

    return get_partition_sum("CO", isotopologue, temperature_k)
