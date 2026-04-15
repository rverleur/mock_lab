import csv
from pathlib import Path

import numpy as np

from mock_lab.spectroscopy.hitemp import (
    DEFAULT_HITEMP_SELECTED_TRANSITIONS_CSV_PATH,
    parse_hitemp_par_line,
    write_hitemp_par_csv,
)
from mock_lab.spectroscopy.hitemp import uncertainty_estimate_from_code
from mock_lab.spectroscopy.voigt import DEFAULT_CO_TRANSITIONS, load_default_co_transitions


SAMPLE_RECORD = (
    " 55    2.2487642.700E-164 8.704E-07.07970.08664763.01990.76-.000265"
    "             41             41                    R  0      457665 5 8 2 2 1 7     6.0    2.0"
)


def test_parse_hitemp_par_line_extracts_fixed_width_fields() -> None:
    assert len(SAMPLE_RECORD) == 160

    record = parse_hitemp_par_line(SAMPLE_RECORD)

    assert record["molecule_id"] == 5
    assert record["isotopologue_id"] == 5
    assert np.isclose(record["wavenumber_cm_inv"], 2.248764)
    assert np.isclose(record["line_strength_hitran"], 2.700e-164)
    assert np.isclose(record["einstein_a_s_inv"], 8.704e-7)
    assert np.isclose(record["air_broadened_hwhm_cm_inv_atm"], 0.0797)
    assert np.isclose(record["self_broadened_hwhm_cm_inv_atm"], 0.086)
    assert np.isclose(record["lower_state_energy_cm_inv"], 64763.0199)
    assert np.isclose(record["temperature_dependence_air"], 0.76)
    assert np.isclose(record["pressure_shift_air_cm_inv_atm"], -0.000265)
    assert record["upper_global_quanta"] == "41"
    assert record["lower_global_quanta"] == "41"
    assert record["upper_local_quanta"] == ""
    assert record["lower_local_quanta"] == "R  0"
    assert record["uncertainty_indices"] == "457665"
    assert record["reference_indices"] == "5 8 2 2 1 7"
    assert record["line_mixing_flag"] == ""
    assert np.isclose(record["upper_statistical_weight"], 6.0)
    assert np.isclose(record["lower_statistical_weight"], 2.0)


def test_write_hitemp_par_csv_includes_header(tmp_path) -> None:
    par_path = tmp_path / "sample.par"
    csv_path = tmp_path / "sample.csv"
    par_path.write_text(f"{SAMPLE_RECORD}\n{SAMPLE_RECORD}\n", encoding="ascii")

    row_count = write_hitemp_par_csv(par_path, csv_path)

    assert row_count == 2
    with csv_path.open("r", encoding="utf-8", newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))

    assert len(rows) == 2
    assert rows[0]["molecule_id"] == "5"
    assert rows[0]["wavenumber_cm_inv"] == "2.248764"


def test_uncertainty_estimate_uses_relative_bound_for_broadening() -> None:
    estimate = uncertainty_estimate_from_code(
        5,
        value=0.0436,
        uncertainty_type="relative",
    )

    assert np.isnan(estimate.absolute_upper_bound)
    assert np.isclose(estimate.relative_upper_bound_absolute, 0.00436)
    assert np.isclose(estimate.upper_bound_absolute, 0.00436)


def test_uncertainty_estimate_uses_absolute_bound_for_line_position() -> None:
    estimate = uncertainty_estimate_from_code(
        4,
        value=2008.525448,
        uncertainty_type="absolute",
    )

    assert np.isclose(estimate.absolute_upper_bound, 1.0e-3)
    assert np.isnan(estimate.relative_upper_bound_absolute)
    assert np.isclose(estimate.upper_bound_absolute, 1.0e-3)


def test_default_transitions_are_loaded_from_selected_hitemp_rows() -> None:
    rows_by_label = {
        transition.label: transition
        for transition in DEFAULT_CO_TRANSITIONS
    }

    assert rows_by_label["P(0,31)"].source_csv_row == 109056
    assert np.isclose(rows_by_label["P(0,31)"].center_cm_inv, 2008.525448)
    assert np.isclose(rows_by_label["P(0,31)"].air_broadened_hwhm_cm_inv_atm, 0.0436)
    assert np.isclose(rows_by_label["P(0,31)"].temperature_dependence_air, 0.67)
    assert rows_by_label["P(0,31)"].uncertainty_indices == "477665"
    assert rows_by_label["P(0,31)"].uncertainties["wavenumber_cm_inv"].uncertainty_type == "absolute"
    assert rows_by_label["P(0,31)"].uncertainties["line_strength"].uncertainty_type == "relative"
    assert np.isclose(
        rows_by_label["P(0,31)"].uncertainties["wavenumber_cm_inv"].upper_bound_absolute,
        1.0e-3,
    )
    assert np.isclose(
        rows_by_label["P(0,31)"].uncertainties["line_strength"].upper_bound_absolute,
        rows_by_label["P(0,31)"].line_strength_ref * 0.02,
    )

    assert rows_by_label["P(2,20)"].source_csv_row == 109053
    assert np.isclose(rows_by_label["P(2,20)"].center_cm_inv, 2008.421529)

    assert rows_by_label["P(3,14)"].source_csv_row == 109058
    assert np.isclose(rows_by_label["P(3,14)"].center_cm_inv, 2008.551943)


def test_load_default_transitions_works_with_curated_csv_only(monkeypatch) -> None:
    monkeypatch.setattr(
        "mock_lab.spectroscopy.voigt.DEFAULT_HITEMP_SELECTED_TRANSITIONS_CSV_PATH",
        DEFAULT_HITEMP_SELECTED_TRANSITIONS_CSV_PATH,
    )
    monkeypatch.setattr(
        "mock_lab.spectroscopy.voigt.DEFAULT_HITEMP_CO_PAR_PATH",
        Path("/tmp/does-not-exist.par"),
    )

    transitions = load_default_co_transitions()

    assert [transition.label for transition in transitions] == [
        "P(0,31)",
        "P(2,20)",
        "P(3,14)",
    ]
