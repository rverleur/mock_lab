# Scripts

These files are IDE-friendly run targets. Each script resolves repo-local paths, imports the relevant pipeline function from `src/mock_lab/`, and keeps its editable settings in a `main()` function or the `if __name__ == "__main__":` block.

## Core Workflow

1. `run_baseline.py`: build `data/interim/baseline/baseline_average.npz`.
2. `run_etalon.py`: fit the relative wavenumber calibration under `data/interim/etalon/`.
3. `run_shock_snapshot.py`: export the calibrated shock absorbance bundle.
4. `run_voigt_fit.py`: refit the shock spectra and write the Voigt summary table.
5. `run_time_history.py`: build the state-history table and figure.
6. `run_full_pipeline.py`: run the full baseline-to-state-history chain and write the report summary.

## Extended Outputs

- `run_report_figures.py`: rebuild the report-facing figures.
- `run_voigt_fit_video.py`: encode the sweep-by-sweep Voigt-fit video.
- `run_state_history_monte_carlo.py`: run or resume the spectroscopy Monte Carlo.
- `plot_state_history_monte_carlo.py`: redraw the Monte Carlo summary figure from saved results.
- `extract_hitemp_par_to_csv.py`: convert a local full HiTEMP `.par` file to CSV.
- `extract_mock_lab_hitemp_transitions.py`: regenerate the curated three-transition CSV used by the runtime code.
