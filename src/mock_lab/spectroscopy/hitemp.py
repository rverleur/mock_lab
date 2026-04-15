"""Utilities for extracting fixed-width HiTEMP/HITRAN `.par` line lists."""

from __future__ import annotations

import csv
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from math import nan
from pathlib import Path
from typing import Any


PAR_RECORD_LENGTH = 160
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_HITEMP_CO_PAR_PATH = REPO_ROOT / "third_party" / "HiTEMP" / "05_HITEMP2019.par"
DEFAULT_HITEMP_SELECTED_TRANSITIONS_CSV_PATH = (
    REPO_ROOT / "third_party" / "HiTEMP" / "mock_lab_co_transitions.csv"
)

ABSOLUTE_UNCERTAINTY_UPPER_BOUND: dict[int, float] = {
    0: nan,
    1: 1.0,
    2: 1.0e-1,
    3: 1.0e-2,
    4: 1.0e-3,
    5: 1.0e-4,
    6: 1.0e-5,
    7: 1.0e-6,
    8: 1.0e-7,
    9: 1.0e-8,
}
RELATIVE_UNCERTAINTY_UPPER_BOUND: dict[int, float] = {
    0: nan,
    1: nan,
    2: nan,
    3: nan,
    4: 0.20,
    5: 0.10,
    6: 0.05,
    7: 0.02,
    8: 0.01,
    9: 0.001,
}
UNCERTAINTY_PARAMETER_NAMES: tuple[str, ...] = (
    "wavenumber_cm_inv",
    "line_strength",
    "air_broadened_hwhm_cm_inv_atm",
    "self_broadened_hwhm_cm_inv_atm",
    "temperature_dependence_air",
    "pressure_shift_air_cm_inv_atm",
)
ABSOLUTE_UNCERTAINTY_PARAMETERS = {
    "wavenumber_cm_inv",
    "pressure_shift_air_cm_inv_atm",
}


@dataclass(frozen=True)
class ParField:
    """One fixed-width field in a HiTEMP/HITRAN 160-character record."""

    name: str
    start: int
    stop: int
    parser: Callable[[str], Any]


@dataclass(frozen=True)
class UncertaintyEstimate:
    """Upper-bound uncertainty estimate decoded from one HiTEMP/HITRAN code."""

    code: int
    uncertainty_type: str
    absolute_upper_bound: float
    relative_upper_bound_fraction: float
    relative_upper_bound_absolute: float
    upper_bound_absolute: float


def _parse_int(value: str) -> int | str:
    """Parse an integer field while preserving blanks as empty strings."""

    stripped = value.strip()
    return int(stripped) if stripped else ""


def _parse_float(value: str) -> float | str:
    """Parse a floating-point field while preserving blanks as empty strings."""

    stripped = value.strip()
    return float(stripped) if stripped else ""


def _parse_string(value: str) -> str:
    """Parse a text field by removing fixed-width padding."""

    return value.strip()


PAR_FIELDS: tuple[ParField, ...] = (
    ParField("molecule_id", 0, 2, _parse_int),
    ParField("isotopologue_id", 2, 3, _parse_int),
    ParField("wavenumber_cm_inv", 3, 15, _parse_float),
    ParField("line_strength_hitran", 15, 25, _parse_float),
    ParField("einstein_a_s_inv", 25, 35, _parse_float),
    ParField("air_broadened_hwhm_cm_inv_atm", 35, 40, _parse_float),
    ParField("self_broadened_hwhm_cm_inv_atm", 40, 45, _parse_float),
    ParField("lower_state_energy_cm_inv", 45, 55, _parse_float),
    ParField("temperature_dependence_air", 55, 59, _parse_float),
    ParField("pressure_shift_air_cm_inv_atm", 59, 67, _parse_float),
    ParField("upper_global_quanta", 67, 82, _parse_string),
    ParField("lower_global_quanta", 82, 97, _parse_string),
    ParField("upper_local_quanta", 97, 112, _parse_string),
    ParField("lower_local_quanta", 112, 127, _parse_string),
    ParField("uncertainty_indices", 127, 133, _parse_string),
    ParField("reference_indices", 133, 145, _parse_string),
    ParField("line_mixing_flag", 145, 146, _parse_string),
    ParField("upper_statistical_weight", 146, 153, _parse_float),
    ParField("lower_statistical_weight", 153, 160, _parse_float),
)
CSV_HEADER = tuple(field.name for field in PAR_FIELDS)


def normalize_par_record(raw_line: str, *, line_number: int | None = None) -> str:
    """Return one 160-character `.par` record with newline characters removed."""

    record = raw_line.rstrip("\r\n")
    if len(record) < PAR_RECORD_LENGTH:
        location = f" on line {line_number}" if line_number is not None else ""
        raise ValueError(
            f"Expected at least {PAR_RECORD_LENGTH} characters{location}, got {len(record)}."
        )
    return record[:PAR_RECORD_LENGTH]


def parse_hitemp_par_line(raw_line: str, *, line_number: int | None = None) -> dict[str, Any]:
    """Parse one HiTEMP/HITRAN 160-character `.par` record into named fields."""

    record = normalize_par_record(raw_line, line_number=line_number)
    return {
        field.name: field.parser(record[field.start : field.stop])
        for field in PAR_FIELDS
    }


def iter_hitemp_par_records(par_path: Path) -> Iterator[dict[str, Any]]:
    """Yield parsed records from a HiTEMP/HITRAN `.par` file one line at a time."""

    with par_path.open("r", encoding="ascii", errors="replace", newline="") as par_file:
        for line_number, raw_line in enumerate(par_file, start=1):
            if not raw_line.strip():
                continue
            yield parse_hitemp_par_line(raw_line, line_number=line_number)


def write_hitemp_par_csv(par_path: Path, csv_path: Path) -> int:
    """Convert a HiTEMP/HITRAN `.par` file to CSV and return the row count."""

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    row_count = 0

    with csv_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_HEADER)
        writer.writeheader()
        for record in iter_hitemp_par_records(par_path):
            writer.writerow(record)
            row_count += 1

    return row_count


def read_hitemp_par_rows(par_path: Path, par_line_numbers: set[int]) -> dict[int, dict[str, Any]]:
    """Return selected one-indexed `.par` lines without loading the full file."""

    if not par_line_numbers:
        return {}

    requested_line_numbers = set(par_line_numbers)
    selected_records: dict[int, dict[str, Any]] = {}
    final_line_number = max(requested_line_numbers)

    with par_path.open("r", encoding="ascii", errors="replace", newline="") as par_file:
        for line_number, raw_line in enumerate(par_file, start=1):
            if line_number > final_line_number:
                break
            if line_number in requested_line_numbers:
                selected_records[line_number] = parse_hitemp_par_line(
                    raw_line,
                    line_number=line_number,
                )

    missing_lines = requested_line_numbers - set(selected_records)
    if missing_lines:
        missing = ", ".join(str(line_number) for line_number in sorted(missing_lines))
        raise ValueError(f"Could not find requested HiTEMP .par line(s): {missing}.")

    return selected_records


def par_line_number_from_csv_row(csv_row_number: int) -> int:
    """Convert a one-indexed CSV row number into the underlying `.par` line number.

    The generated CSV includes one header row, while the `.par` file does not.
    """

    if csv_row_number <= 1:
        raise ValueError("CSV row numbers must account for the header and be greater than 1.")

    return csv_row_number - 1


def read_hitemp_records_by_csv_row(
    csv_row_numbers: set[int],
    *,
    par_path: Path = DEFAULT_HITEMP_CO_PAR_PATH,
) -> dict[int, dict[str, Any]]:
    """Return parsed HiTEMP records keyed by the generated CSV row number."""

    par_rows_by_csv_row = {
        csv_row_number: par_line_number_from_csv_row(csv_row_number)
        for csv_row_number in csv_row_numbers
    }
    records_by_par_line = read_hitemp_par_rows(par_path, set(par_rows_by_csv_row.values()))
    return {
        csv_row_number: records_by_par_line[par_line_number]
        for csv_row_number, par_line_number in par_rows_by_csv_row.items()
    }


def read_selected_transition_records(
    csv_row_numbers: set[int],
    *,
    csv_path: Path = DEFAULT_HITEMP_SELECTED_TRANSITIONS_CSV_PATH,
) -> dict[int, dict[str, Any]]:
    """Return curated transition rows keyed by their original HiTEMP CSV row number."""

    if not csv_row_numbers:
        return {}

    requested_rows = {int(row_number) for row_number in csv_row_numbers}
    selected_records: dict[int, dict[str, Any]] = {}

    with csv_path.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            source_csv_row = int(row["source_csv_row"])
            if source_csv_row not in requested_rows:
                continue
            selected_records[source_csv_row] = {
                "molecule_id": int(row["molecule_id"]),
                "isotopologue_id": int(row["isotopologue_id"]),
                "wavenumber_cm_inv": float(row["wavenumber_cm_inv"]),
                "line_strength_hitran": float(row["line_strength_hitran_ref"]),
                "einstein_a_s_inv": float(row["einstein_a_s_inv"]),
                "air_broadened_hwhm_cm_inv_atm": float(
                    row["air_broadened_hwhm_cm_inv_atm"]
                ),
                "self_broadened_hwhm_cm_inv_atm": float(
                    row["self_broadened_hwhm_cm_inv_atm"]
                ),
                "lower_state_energy_cm_inv": float(row["lower_state_energy_cm_inv"]),
                "temperature_dependence_air": float(row["temperature_dependence_air"]),
                "pressure_shift_air_cm_inv_atm": float(
                    row["pressure_shift_air_cm_inv_atm"]
                ),
                "upper_global_quanta": row["upper_global_quanta"],
                "lower_global_quanta": row["lower_global_quanta"],
                "upper_local_quanta": row["upper_local_quanta"],
                "lower_local_quanta": row["lower_local_quanta"],
                "uncertainty_indices": row["uncertainty_indices"],
                "reference_indices": row["reference_indices"],
                "line_mixing_flag": row["line_mixing_flag"],
                "upper_statistical_weight": float(row["upper_statistical_weight"]),
                "lower_statistical_weight": float(row["lower_statistical_weight"]),
            }

    missing_rows = requested_rows - set(selected_records)
    if missing_rows:
        missing = ", ".join(str(row_number) for row_number in sorted(missing_rows))
        raise ValueError(
            "Could not find requested curated HiTEMP transition row(s): "
            f"{missing}."
        )

    return selected_records


def split_uncertainty_indices(uncertainty_indices: str) -> tuple[int, ...]:
    """Split the six-character HiTEMP/HITRAN uncertainty-code string."""

    codes = tuple(int(character) for character in uncertainty_indices.strip())
    if len(codes) != len(UNCERTAINTY_PARAMETER_NAMES):
        raise ValueError(
            "Expected six HiTEMP/HITRAN uncertainty indices, "
            f"got {len(codes)} from {uncertainty_indices!r}."
        )
    return codes


def split_reference_indices(reference_indices: str) -> tuple[int, ...]:
    """Split the space-delimited reference-index field when possible."""

    stripped = reference_indices.strip()
    if not stripped:
        return ()

    return tuple(int(value) for value in stripped.split())


def uncertainty_estimate_from_code(
    code: int,
    value: float,
    *,
    uncertainty_type: str,
) -> UncertaintyEstimate:
    """Return the uncertainty implied by one HiTEMP/HITRAN code.

    HiTEMP/HITRAN uses absolute uncertainty codes for line position and air
    pressure-induced line shift, and relative uncertainty codes for line
    intensity and broadening-related parameters.
    """

    if uncertainty_type == "absolute":
        absolute_upper_bound = ABSOLUTE_UNCERTAINTY_UPPER_BOUND.get(code, nan)
        relative_upper_bound_fraction = nan
        relative_upper_bound_absolute = nan
        upper_bound_absolute = absolute_upper_bound
    elif uncertainty_type == "relative":
        absolute_upper_bound = nan
        relative_upper_bound_fraction = RELATIVE_UNCERTAINTY_UPPER_BOUND.get(code, nan)
        relative_upper_bound_absolute = abs(value) * relative_upper_bound_fraction
        upper_bound_absolute = relative_upper_bound_absolute
    else:
        raise ValueError(f"Unknown uncertainty type: {uncertainty_type!r}.")

    return UncertaintyEstimate(
        code=code,
        uncertainty_type=uncertainty_type,
        absolute_upper_bound=absolute_upper_bound,
        relative_upper_bound_fraction=relative_upper_bound_fraction,
        relative_upper_bound_absolute=relative_upper_bound_absolute,
        upper_bound_absolute=upper_bound_absolute,
    )


def uncertainty_estimates_for_record(
    record: dict[str, Any],
    *,
    line_strength_value: float | None = None,
) -> dict[str, UncertaintyEstimate]:
    """Return uncertainty estimates for the six coded parameters in one record."""

    codes = split_uncertainty_indices(str(record["uncertainty_indices"]))
    values = {
        "wavenumber_cm_inv": float(record["wavenumber_cm_inv"]),
        "line_strength": (
            float(line_strength_value)
            if line_strength_value is not None
            else float(record["line_strength_hitran"])
        ),
        "air_broadened_hwhm_cm_inv_atm": float(record["air_broadened_hwhm_cm_inv_atm"]),
        "self_broadened_hwhm_cm_inv_atm": float(record["self_broadened_hwhm_cm_inv_atm"]),
        "temperature_dependence_air": float(record["temperature_dependence_air"]),
        "pressure_shift_air_cm_inv_atm": float(record["pressure_shift_air_cm_inv_atm"]),
    }

    return {
        parameter_name: uncertainty_estimate_from_code(
            code,
            values[parameter_name],
            uncertainty_type=(
                "absolute"
                if parameter_name in ABSOLUTE_UNCERTAINTY_PARAMETERS
                else "relative"
            ),
        )
        for parameter_name, code in zip(UNCERTAINTY_PARAMETER_NAMES, codes)
    }
