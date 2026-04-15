# Pipeline Guide

`src/mock_lab/pipelines/` contains the stage-level workflows that wire the lower-level `io/` and `spectroscopy/` modules into saved analysis products.

## Deterministic Workflow

The deterministic pipeline is:

1. `baseline.py`
2. `etalon.py`
3. `shock_snapshot.py`
4. `voigt_fit.py`
5. `time_history.py`
6. `full_pipeline.py`

`full_pipeline.py` simply orchestrates that chain end-to-end. It does not run the Monte Carlo refit or the fit-progression video unless those stages are called explicitly.

## Stage Details

### `baseline.py`

This stage loads `MockLabData_Baseline.mat`, splits it into aligned sweeps, averages those sweeps, fits a straight line through the first and last `32` samples, and saves:

- `baseline_average.npz`
- optionally `baseline_average.png`

The saved baseline bundle is the reusable reference intensity `I0` for the shock absorbance calculation.

### `etalon.py`

This stage loads the etalon MAT file, removes the simple edge baseline from each sweep, averages the corrected sweeps, detects peaks on that representative trace, and fits the polynomial calibration. By default it uses a 4th-order polynomial and saves:

- `etalon_fit.npz`
- `etalon_sweep.png`
- `etalon_calibration.png`

### `shock_snapshot.py`

This is the preprocessing bridge between raw shock traces and the spectral fit.

It:

- loads the saved etalon calibration
- loads the saved baseline bundle, or regenerates it from the baseline MAT file if the baseline export is missing
- fits and subtracts an edge line from every shock sweep
- scales each corrected shock sweep to the corrected average baseline peak
- evaluates Beer-Lambert absorbance on the calibrated relative-wavenumber axis
- applies the fixed analysis window defined by the corrected baseline signal level

It saves the main handoff file for later stages:

- `shock_frequency_domain.npz`

That bundle contains the calibrated frequency axis, corrected shock sweeps, scale factors, absorbance sweeps, the saved baseline products, and the reference sweeps.

### `voigt_fit.py`

This stage loads `shock_frequency_domain.npz` and fits every usable sweep with the constrained three-transition CO model.

Important behavior:

- sweeps below `minimum_peak_absorbance = 0.02` are skipped by default
- the fit runs sequentially through the sweep stack
- the previous successful fit is reused as the initial guess for the next sweep, so the batch behaves like a warm-started time-history reduction rather than a set of independent fits

It saves:

- `voigt_fit_results.npz`
- `voigt_fit_summary.csv`
- `shock_voigt_fit.png`

The NPZ file is the main spectral-fit handoff into state reduction. It contains the explicit expanded line parameters, reconstructed fitted absorbance, success flags, reduced parameter covariance, and confidence metadata.

### `time_history.py`

This stage converts the fitted spectra into the reported thermochemical history.

It reads:

- `temperature_k`
- `collisional_hwhm_cm_inv`
- `line_areas[:, 0]`
- `reduced_parameter_covariance`

from `voigt_fit_results.npz`, then:

- builds scan time from the fixed sweep frequency
- solves for pressure and `X_CO`
- propagates nominal fit covariance into `T`, `P`, and `X_CO` confidence bands with a finite-difference Jacobian
- writes the table outputs and figure

It saves:

- `state_history.npz`
- `state_history.csv`
- `state_history.png`

The deterministic confidence intervals in this stage come only from the nominal fit covariance. Broader spectroscopy uncertainty is handled separately by the Monte Carlo refit stage.

## Monte Carlo Refit

`monte_carlo_state_history.py` is the optional expensive stage for spectroscopy-driven uncertainty.

It reads the nominal deterministic outputs and then, for each trial:

1. samples a full transition set from uniform bounds derived from the stored spectroscopy uncertainties
2. reruns the constrained Voigt fit across the whole sweep stack with that sampled transition set
3. rebuilds the state history from the trial fit results

The currently sampled quantities are:

- line center
- reference line strength
- CO self-broadening `gamma_ref`
- CO self-broadening temperature exponent
- N2-equivalent bath-gas `gamma_ref`
- N2-equivalent bath-gas temperature exponent

The run is resumable. It checkpoints:

- `state_history_refit_trial_states.npy`
- `state_history_refit_sampled_transitions.npy`
- `state_history_refit_mc_metadata.json`

and summarizes the completed trials into:

- `state_history_monte_carlo_summary.npz`
- `state_history_monte_carlo_summary.csv`
- `state_history_monte_carlo.png`

The saved summary combines two independent pieces of uncertainty:

- trial-to-trial spread from the full refit with sampled spectroscopy
- nominal-fit covariance from `state_history.npz`

Those half-widths are combined by root-sum-square to produce the reported total bounds.

## Script Relationship

The files under `scripts/` are thin wrappers around these pipeline functions. Their settings live in `if __name__ == "__main__":` blocks so they can be edited and run directly from an IDE without adding command-line parsing.
