# ME 617 Mock Lab

This repository is the working Python reduction for the scanned-wavelength direct-absorption mock lab. It loads the provided MATLAB baseline, etalon, and shock-tube traces; converts them into aligned sweeps and absorbance spectra; fits the shock scans with a constrained three-transition CO Voigt model; and reduces those fits to scan-by-scan temperature, pressure, and CO mole-fraction histories.

## Repository Map

- `src/mock_lab/io/`: MAT loading, reference-channel binarization, and sweep alignment.
- `src/mock_lab/spectroscopy/`: absorbance reduction, etalon calibration, HiTEMP/TIPS access, Voigt fitting, and state estimation.
- `src/mock_lab/pipelines/`: stage-level workflows for baseline, etalon, shock absorbance, Voigt fitting, time history, Monte Carlo, and the full pipeline.
- `src/mock_lab/plotting/`: figure builders used by the pipeline stages and report scripts.
- `src/mock_lab/reporting/`: concise report-facing run summaries.
- `scripts/`: IDE-friendly entry points that wire repo-local paths into the pipeline functions.
- `tests/`: unit tests plus pipeline smoke coverage.
- `data/raw/`: the three provided MAT files.
- `data/interim/`: reusable intermediate products such as the averaged baseline and etalon fit.
- `data/processed/exports/`: durable machine-readable exports such as the shock absorbance bundle and Voigt-fit results.
- `results/`: saved figures, tables, Monte Carlo outputs, videos, and report summaries.
- `report/`: the LaTeX report workspace and imported figures.
- `third_party/`: vendored spectroscopy assets. The runtime transition set comes from `third_party/HiTEMP/mock_lab_co_transitions.csv`; the full HiTEMP `.par` file is only needed if you want to regenerate that curated file.
- `docs/`: handout PDFs, reference papers, and working notes.
- `examples/`: reference MATLAB material kept for comparison only.

## Deeper Package Guides

The top-level README stays intentionally short. The implementation notes that explain how each reduction step works now live alongside the code:

- [`src/mock_lab/README.md`](src/mock_lab/README.md): package-level map, stage handoff files, and where to edit the workflow.
- [`src/mock_lab/io/README.md`](src/mock_lab/io/README.md): MAT loading, timing reconstruction, and sweep-cutting assumptions.
- [`src/mock_lab/spectroscopy/README.md`](src/mock_lab/spectroscopy/README.md): absorbance reduction, etalon calibration, spectroscopy inputs, Voigt-fit constraints, and state-reduction assumptions.
- [`src/mock_lab/pipelines/README.md`](src/mock_lab/pipelines/README.md): stage orchestration, saved outputs, warm-start behavior, and Monte Carlo flow.
- [`src/mock_lab/plotting/README.md`](src/mock_lab/plotting/README.md) and [`src/mock_lab/reporting/README.md`](src/mock_lab/reporting/README.md): figure-generation and report-summary helpers.

## How The Code Flows

1. `mock_lab.io.matlab_loader` loads each MAT file, identifies the detector and reference channels, reconstructs the time axis, and splits the continuous traces into phase-aligned sweeps.
2. `mock_lab.pipelines.baseline` averages the no-absorption baseline sweeps and saves the edge-fit baseline correction.
3. `mock_lab.pipelines.etalon` fits a polynomial time-to-relative-wavenumber calibration from the etalon peaks.
4. `mock_lab.pipelines.shock_snapshot` converts each shock sweep to baseline-corrected absorbance on the calibrated axis.
5. `mock_lab.pipelines.voigt_fit` fits each usable absorbance sweep with the constrained three-line CO Voigt model and exports per-scan fit statistics.
6. `mock_lab.pipelines.time_history` converts the fitted linewidth and strongest-line area into state history with covariance-based confidence intervals.
7. `mock_lab.pipelines.monte_carlo_state_history` optionally reruns the full fit under sampled spectroscopy perturbations to estimate total uncertainty.

## Running The Workflow

Use the repo virtual environment if it already exists:

```bash
.venv/bin/python scripts/run_full_pipeline.py
```

Useful stage targets:

- `.venv/bin/python scripts/run_baseline.py`
- `.venv/bin/python scripts/run_etalon.py`
- `.venv/bin/python scripts/run_shock_snapshot.py`
- `.venv/bin/python scripts/run_voigt_fit.py`
- `.venv/bin/python scripts/run_time_history.py`
- `.venv/bin/python scripts/run_report_figures.py`
- `.venv/bin/python scripts/run_state_history_monte_carlo.py`

Run the test suite with:

```bash
.venv/bin/python -m pytest -q
```

## Key Outputs

- `data/interim/baseline/baseline_average.npz`: averaged baseline sweep and edge-fit correction.
- `data/interim/etalon/etalon_fit.npz`: relative wavenumber calibration and picked etalon peaks.
- `data/processed/exports/shock_frequency_domain.npz`: corrected shock sweeps, absorbance, and calibration metadata.
- `data/processed/exports/voigt_fit_results.npz`: per-scan Voigt parameters, covariance, and fitted absorbance.
- `results/tables/state_history.npz` and `results/tables/state_history.csv`: reduced temperature, pressure, and CO history.
- `results/reports/analysis_summary.md`: compact map of the generated analysis products.

## Dependencies

Core runtime dependencies are declared in `pyproject.toml` and mirrored in `requirements.txt`:

- `numpy`
- `scipy`
- `matplotlib`
- `pytest` for tests
- `jupyter` and `ipykernel` for notebook work
