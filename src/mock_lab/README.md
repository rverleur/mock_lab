# `mock_lab` Package Guide

`src/mock_lab/` is the active reduction package. It turns the three provided MAT files into aligned sweeps, calibrated absorbance spectra, constrained Voigt-fit results, and scan-by-scan `T`, `P`, and `X_CO` histories.

## Package Map

- [`io/README.md`](io/README.md): raw MAT loading, TTL-reference decoding, time-axis reconstruction, and sweep splitting.
- [`spectroscopy/README.md`](spectroscopy/README.md): baseline removal, etalon calibration, spectroscopy assets, Voigt fitting, and state estimation.
- [`pipelines/README.md`](pipelines/README.md): stage-level orchestration and durable handoff files between stages.
- [`plotting/README.md`](plotting/README.md): figure helpers and the shared headless Matplotlib setup.
- [`reporting/README.md`](reporting/README.md): compact report-facing summaries built from saved outputs.

## End-To-End Flow

1. `mock_lab.io.matlab_loader` loads the MAT files, identifies the detector and reference channels, reconstructs time from the TTL reference, and cuts the continuous traces into phase-aligned sweeps.
2. `mock_lab.pipelines.baseline` averages the baseline sweeps, fits the edge line, and saves `data/interim/baseline/baseline_average.npz`.
3. `mock_lab.pipelines.etalon` builds a representative etalon sweep, picks peaks, fits the time-to-relative-wavenumber polynomial, and saves `data/interim/etalon/etalon_fit.npz`.
4. `mock_lab.pipelines.shock_snapshot` applies the baseline correction and etalon calibration to every shock sweep, computes absorbance on the calibrated axis, and saves `data/processed/exports/shock_frequency_domain.npz`.
5. `mock_lab.pipelines.voigt_fit` fits each usable absorbance sweep with the constrained three-transition CO model and saves `data/processed/exports/voigt_fit_results.npz`.
6. `mock_lab.pipelines.time_history` reduces the fitted temperature, linewidth, and strongest-line area to `results/tables/state_history.npz` and `results/tables/state_history.csv`.
7. `mock_lab.reporting.summary` writes `results/reports/analysis_summary.md` as a compact run summary.
8. `mock_lab.pipelines.monte_carlo_state_history` optionally reruns the fit under sampled spectroscopy perturbations to estimate total uncertainty.

## Durable Handoff Files

The code is organized so each stage writes a durable output that the next stage can load directly.

- `data/interim/baseline/baseline_average.npz`: averaged baseline sweep, fitted edge line, corrected baseline, and sweep metadata.
- `data/interim/etalon/etalon_fit.npz`: polynomial calibration, representative etalon sweep, and picked peaks.
- `data/processed/exports/shock_frequency_domain.npz`: calibrated shock absorbance sweeps and supporting preprocessing products.
- `data/processed/exports/voigt_fit_results.npz`: per-scan fitted parameters, covariance, success flags, and reconstructed absorbance.
- `results/tables/state_history.npz`: reduced `T`, `P`, and `X_CO` arrays with covariance-based confidence bounds.
- `results/monte_carlo/state_history_monte_carlo_summary.npz`: Monte Carlo summary arrays when the optional refit stage is run.

## Where To Change Behavior

- Edit `io/` when the raw MAT format, sweep reference frequency, or phase window changes.
- Edit `spectroscopy/` when the physical model changes: line data, line-shape assumptions, temperature scaling, or pressure/mole-fraction reduction.
- Edit `pipelines/` when stage boundaries, saved products, warm-start logic, or orchestration need to change.
- Edit `plotting/` when figure layout or backend behavior changes.
- Edit `reporting/` only for report-facing summaries; it should not contain primary analysis logic.

## Package-Wide Assumptions

- The reference channel is a TTL-like sweep marker at `300 kHz` unless a caller overrides it explicitly.
- Sweep extraction starts at a fixed local phase offset of `2.2 us`, so every downstream stage assumes the same baseline-ramp-baseline windowing.
- The etalon calibration is relative, not absolute, so the active analysis uses a relative wavenumber axis.
- The fitted spectrum is the sum of three CO transitions plus a linear baseline, with structural constraints that stabilize line identity.
- Pressure reduction treats all non-CO partners as one N2-equivalent bath gas and solves `P` and `X_CO` simultaneously from the strongest fitted line.
