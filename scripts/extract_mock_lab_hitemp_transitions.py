"""Export the three HiTEMP CO transitions used by the mock-lab fit."""

from __future__ import annotations

import csv
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mock_lab.spectroscopy.hitemp import UNCERTAINTY_PARAMETER_NAMES
from mock_lab.spectroscopy.voigt import DEFAULT_CO_TRANSITIONS


DEFAULT_CSV_FILE = REPO_ROOT / "third_party" / "HiTEMP" / "mock_lab_co_transitions.csv"


def transition_to_row(transition) -> dict[str, object]:
    """Flatten one transition plus its uncertainty metadata into a CSV row."""

    row: dict[str, object] = {
        "label": transition.label,
        "source_csv_row": transition.source_csv_row,
        "source_par_line": transition.source_par_line,
        "molecule_id": transition.molecule_id,
        "isotopologue_id": transition.isotopologue_id,
        "upper_global_quanta": transition.upper_global_quanta,
        "lower_global_quanta": transition.lower_global_quanta,
        "upper_local_quanta": transition.upper_local_quanta,
        "lower_local_quanta": transition.lower_local_quanta,
        "wavenumber_cm_inv": transition.center_cm_inv,
        "line_strength_hitran_ref": transition.line_strength_hitran_ref,
        "line_strength_ref_cm2_atm": transition.line_strength_ref,
        "einstein_a_s_inv": transition.einstein_a_s_inv,
        "air_broadened_hwhm_cm_inv_atm": transition.air_broadened_hwhm_cm_inv_atm,
        "self_broadened_hwhm_cm_inv_atm": transition.self_broadened_hwhm_cm_inv_atm,
        "lower_state_energy_cm_inv": transition.lower_state_energy_cm_inv,
        "temperature_dependence_air": transition.temperature_dependence_air,
        "pressure_shift_air_cm_inv_atm": transition.pressure_shift_air_cm_inv_atm,
        "uncertainty_indices": transition.uncertainty_indices,
        "reference_indices": transition.reference_indices,
        "line_mixing_flag": transition.line_mixing_flag,
        "upper_statistical_weight": transition.upper_statistical_weight,
        "lower_statistical_weight": transition.lower_statistical_weight,
    }

    for parameter_index, parameter_name in enumerate(UNCERTAINTY_PARAMETER_NAMES):
        estimate = transition.uncertainties[parameter_name]
        row[f"{parameter_name}_uncertainty_code"] = estimate.code
        row[f"{parameter_name}_uncertainty_type"] = estimate.uncertainty_type
        row[f"{parameter_name}_absolute_upper_bound"] = estimate.absolute_upper_bound
        row[f"{parameter_name}_relative_upper_bound_fraction"] = (
            estimate.relative_upper_bound_fraction
        )
        row[f"{parameter_name}_relative_upper_bound_absolute"] = (
            estimate.relative_upper_bound_absolute
        )
        row[f"{parameter_name}_upper_bound_absolute_uncertainty"] = (
            estimate.upper_bound_absolute
        )
        row[f"{parameter_name}_reference_index"] = (
            transition.reference_index_values[parameter_index]
            if parameter_index < len(transition.reference_index_values)
            else ""
        )

    return row


def main(output_path: Path = DEFAULT_CSV_FILE) -> None:
    """Write the selected-transition metadata table."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [transition_to_row(transition) for transition in DEFAULT_CO_TRANSITIONS]

    with output_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=tuple(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} selected transition rows to {output_path}")


if __name__ == "__main__":
    output_path = DEFAULT_CSV_FILE

    main(output_path=output_path)
