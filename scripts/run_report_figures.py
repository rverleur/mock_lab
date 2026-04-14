"""Build the current etalon and shock figures for the report."""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mock_lab.pipelines.etalon import run_etalon_pipeline
from mock_lab.plotting.figures import (
    plot_etalon_report_figure,
    plot_shock_time_report_figure,
    plot_state_history,
    plot_voigt_fit,
    save_figure,
)
from mock_lab.pipelines.shock_snapshot import run_shock_snapshot_pipeline
from mock_lab.pipelines.voigt_fit import run_voigt_fit_pipeline


def main() -> None:
    """Run the current figure-generation workflow for the report."""

    etalon_data = REPO_ROOT / "data" / "raw" / "MockLabData_Etalon.mat"
    shock_data = REPO_ROOT / "data" / "raw" / "MockLabData_Shock.mat"
    baseline_dir = REPO_ROOT / "data" / "interim" / "baseline"
    etalon_output_dir = REPO_ROOT / "data" / "interim" / "etalon"
    shock_output_dir = REPO_ROOT / "data" / "processed" / "exports"
    figure_dir = REPO_ROOT / "results" / "figures"
    report_figure_dir = figure_dir / "report_figures"
    monte_carlo_summary = (
        REPO_ROOT
        / "results"
        / "monte_carlo"
        / "bath_gas_full_refit"
        / "state_history_monte_carlo_summary.npz"
    )

    etalon_result = run_etalon_pipeline(
        raw_data=etalon_data,
        output_dir=etalon_output_dir,
        figure_output_dir=figure_dir,
        plot_sweep_index = 60,
    )
    etalon_report_figure = plot_etalon_report_figure(
        etalon_result.fit.time_axis_s - etalon_result.fit.time_axis_s[0],
        etalon_result.representative_sweep,
        etalon_result.fit.peak_indices,
        etalon_result.fit.peak_wavenumber_cm_inv,
        etalon_result.fit.relative_wavenumber_cm_inv,
        etalon_result.fit.peak_residual_cm_inv,
    )
    save_figure(etalon_report_figure, report_figure_dir / "etalon.png")
    plt.close(etalon_report_figure)

    shock_result = run_shock_snapshot_pipeline(
        raw_data=shock_data,
        baseline_dir=baseline_dir,
        etalon_dir=etalon_output_dir,
        output_dir=shock_output_dir,
        figure_output_dir=figure_dir,
        plot_sweep_index = 60,
    )
    shock_report_figure = plot_shock_time_report_figure(
        shock_result.display_time_axis_s,
        shock_result.corrected_baseline_sweep,
        shock_result.scaled_signal_sweeps[shock_result.plot_sweep_index],
        shock_result.display_time_axis_s[shock_result.analysis_window],
        shock_result.absorbance_sweeps[shock_result.plot_sweep_index, shock_result.analysis_window],
    )
    save_figure(shock_report_figure, report_figure_dir / "shock_time.png")
    plt.close(shock_report_figure)

    run_voigt_fit_pipeline(
        shock_frequency_data=shock_output_dir / "shock_frequency_domain.npz",
        output_dir=shock_output_dir,
        figure_output_dir=figure_dir,
        table_output_dir=REPO_ROOT / "results" / "tables",
        plot_sweep_index = 60,
    )
    with np.load(shock_output_dir / "voigt_fit_results.npz") as voigt_data:
        selected_index = int(voigt_data["plot_sweep_index"])
        absorbance_sweeps = np.asarray(voigt_data["absorbance_sweeps"], dtype=float)
        fitted_absorbance_sweeps = np.asarray(voigt_data["fitted_absorbance_sweeps"], dtype=float)
        component_absorbance = np.asarray(voigt_data["selected_component_absorbance"], dtype=float)
        component_labels = tuple(np.asarray(voigt_data["selected_component_labels"], dtype=str).tolist())
        component_centers = np.asarray(
            voigt_data["selected_line_centers_relative_cm_inv"],
            dtype=float,
        )
        frequency_cm_inv = np.asarray(voigt_data["frequency_cm_inv"], dtype=float)

    mask = np.isfinite(absorbance_sweeps[selected_index])
    voigt_report_figure = plot_voigt_fit(
        frequency_cm_inv[mask],
        absorbance_sweeps[selected_index, mask],
        fitted_absorbance_sweeps[selected_index, mask],
        component_absorbance,
        component_labels=component_labels,
        component_centers_cm_inv=component_centers,
        report_style=True,
    )
    save_figure(voigt_report_figure, report_figure_dir / "voigt_fit.png")
    plt.close(voigt_report_figure)

    if monte_carlo_summary.is_file():
        with np.load(monte_carlo_summary) as mc_data:
            state_history_figure = plot_state_history(
                1.0e6 * np.asarray(mc_data["scan_time_s"], dtype=float),
                np.asarray(mc_data["mc_mean_temperature_k"], dtype=float),
                np.asarray(mc_data["mc_mean_pressure_atm"], dtype=float),
                np.asarray(mc_data["mc_mean_co_mole_fraction"], dtype=float),
                temperature_lower=np.asarray(mc_data["total_lower_temperature_k"], dtype=float),
                temperature_upper=np.asarray(mc_data["total_upper_temperature_k"], dtype=float),
                pressure_lower=np.asarray(mc_data["total_lower_pressure_atm"], dtype=float),
                pressure_upper=np.asarray(mc_data["total_upper_pressure_atm"], dtype=float),
                co_mole_fraction_lower=np.asarray(mc_data["total_lower_co_mole_fraction"], dtype=float),
                co_mole_fraction_upper=np.asarray(mc_data["total_upper_co_mole_fraction"], dtype=float),
                uncertainty_label=(
                    f"{int(round(100.0 * float(np.asarray(mc_data['confidence_level']).item())))}% total uncertainty"
                ),
                xlabel="time [µs]",
                temperature_ylim=(0.0, 5000.0),
                pressure_ylim=(0.0, 5.0),
                co_mole_fraction_ylim=(0.0, 0.06),
                co_mole_fraction_scale=1.0,
                co_mole_fraction_label="CO mole fraction [-]",
                xlim=(0.0, 1750.0),
                report_style=True,
            )
        save_figure(state_history_figure, report_figure_dir / "state_history.png")
        plt.close(state_history_figure)


if __name__ == "__main__":
    main()
