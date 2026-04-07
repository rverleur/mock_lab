# ME 617 Mock Lab

Python analysis workspace for the ME 617 scanned-wavelength direct-absorption mock lab. This repository processes the provided baseline, etalon, and shock-tube MATLAB datasets into calibrated absorbance spectra, fits those spectra with a three-transition CO Voigt model, and then reduces the fit results to temperature, pressure, and CO mole-fraction histories.

This README is meant to be the practical map for the codebase. It explains the current implementation stage by stage, points to the exact files that own each step, and calls out the main assumptions and fit parameters so you can decide where to start editing when you want to change one part of the analysis.

## Repo Map

The files that matter most for the current workflow are:

- `src/mock_lab/io/matlab_loader.py`: load MAT files, reconstruct time, and split the continuous traces into phase-aligned sweeps
- `src/mock_lab/spectroscopy/etalon_calibration.py`: etalon baseline removal, representative-sweep averaging, peak picking, and polynomial time-to-frequency calibration
- `src/mock_lab/spectroscopy/absorbance.py`: baseline wing fitting, line subtraction, peak scaling, absorbance-window selection, and Beer-Lambert absorbance
- `src/mock_lab/spectroscopy/voigt.py`: three-transition Voigt model, initial guesses, bounds, and nonlinear least-squares fitting
- `src/mock_lab/spectroscopy/state_estimation.py`: convert fitted line parameters into temperature, pressure, and CO mole fraction
- `src/mock_lab/spectroscopy/tips.py`: local HITRAN TIPS partition-sum lookup used by the state-estimation layer
- `third_party/hitran_tips/`: vendored HITRAN TIPS script and partition-sum tables
- `src/mock_lab/plotting/figures.py`: all report-style plotting helpers
- `src/mock_lab/pipelines/etalon.py`: end-to-end etalon calibration pass
- `src/mock_lab/pipelines/shock_snapshot.py`: build the baseline-corrected and absorbance-domain shock spectra
- `src/mock_lab/pipelines/voigt_fit.py`: fit all absorbance sweeps with the three-line Voigt model
- `src/mock_lab/pipelines/time_history.py`: reduce the fitted sweeps into a scan-by-scan state history
- `scripts/run_report_figures.py`: rebuild the figure set used for the report
- `scripts/run_voigt_fit.py`: rebuild spectra and run the Voigt fits
- `scripts/run_time_history.py`: rebuild everything through state history
- `scripts/run_voigt_fit_video.py`: fit all scans and build a QC video of the spectrum fits

There are also two older placeholder pipelines that are not the active implementation path right now:

- `src/mock_lab/pipelines/baseline.py`
- `src/mock_lab/pipelines/full_pipeline.py`

The `data/` tree is now reserved for experimental inputs and generated analysis products. The local HITRAN TIPS resources are kept under `third_party/hitran_tips/` because they are vendored reference assets, not raw lab data.

## Recommended Entry Points

Run the code from the repo root with the local virtual environment:

```bash
.venv/bin/python scripts/run_report_figures.py
.venv/bin/python scripts/run_voigt_fit.py
.venv/bin/python scripts/run_time_history.py
.venv/bin/python scripts/run_voigt_fit_video.py
```

The wrapper scripts are the easiest place to change which dataset is used, where outputs are written, and which sweep index is plotted. For example, `scripts/run_report_figures.py` currently uses `plot_sweep_index=60` when building the report figures.

## End-to-End Workflow

### 1. Load MATLAB data and reconstruct time

The raw datasets live in:

- `data/raw/MockLabData_Baseline.mat`
- `data/raw/MockLabData_Etalon.mat`
- `data/raw/MockLabData_Shock.mat`

The loading logic is in `src/mock_lab/io/matlab_loader.py`.

Current behavior:

- `load_mat_trace()` reads one `.mat` file with `scipy.io.loadmat`, identifies the detector channel and the TTL-like reference channel, and returns a `MatlabTrace` dataclass.
- `binarize_reference()` thresholds the analog reference waveform into a binary waveform.
- `get_time()` finds rising edges in the reference signal, estimates the number of samples per sweep from the median edge spacing, and reconstructs the sample period assuming `DEFAULT_REFERENCE_FREQUENCY_HZ = 300_000.0`.
- `split_samples()` cuts the continuous signal into equal-length sweeps after applying the phase offset `DEFAULT_PHASE_START_S = 2.2e-6`.
- `split_trace()` applies the same sweep extraction to both the detector and reference channels and returns a `SweepCollection`.

Important implementation detail:

- The code intentionally discards the leading partial sweep by starting at `t0 = 2.2 us`.
- Each extracted sweep therefore starts with a little baseline, contains the ramp, and ends with baseline again.
- The local sweep time returned by `split_trace()` still starts at `t0`, not at `0`.
- Some plotting code later subtracts that offset for display only.

If you want to change how the raw data are cut up, start in `src/mock_lab/io/matlab_loader.py`.

### 2. Build the etalon time-to-frequency calibration

The active etalon pipeline is `src/mock_lab/pipelines/etalon.py`.

It does the following:

1. Loads `data/raw/MockLabData_Etalon.mat`
2. Splits the detector signal into phase-aligned sweeps with `split_trace()`
3. Removes a simple edge baseline from every sweep with `remove_edge_baseline()` in `src/mock_lab/spectroscopy/etalon_calibration.py`
4. Averages all corrected sweeps into one low-noise representative sweep with `build_representative_sweep()`
5. Detects the etalon peaks on that averaged sweep with `find_etalon_peaks()`
6. Assigns equally spaced relative wavenumbers to those peaks using the etalon free spectral range
7. Fits a polynomial mapping time to relative wavenumber with `fit_relative_wavenumber()`
8. Saves the calibration to `data/interim/etalon/etalon_fit.npz`
9. Builds the figure outputs `results/figures/etalon_sweep.png` and `results/figures/etalon_calibration.png`

The etalon physics and fitting helpers are in `src/mock_lab/spectroscopy/etalon_calibration.py`.

Key assumptions in the current etalon implementation:

- The etalon free spectral range is fixed by:
  - `ETALON_LENGTH_CM = 3.0 * 2.54`
  - `ETALON_REFRACTIVE_INDEX = 4.019`
  - `ETALON_FSR_CM_INV = 1 / (2 n L)`
- Peak spacing is assumed uniform in relative wavenumber.
- The current pipeline default in `src/mock_lab/pipelines/etalon.py` is `polynomial_order=4`.
- The lower residual panel in the calibration plot is just a diagnostic of the polynomial fit at the detected peak locations.

Files to edit if you want to change the etalon behavior:

- Change peak detection thresholds: `src/mock_lab/spectroscopy/etalon_calibration.py`
- Change the fit order or which run script uses which order: `src/mock_lab/pipelines/etalon.py`
- Change the look of the figure: `src/mock_lab/plotting/figures.py`

Note on plotting:

- The fit itself is evaluated on the local sweep time axis that starts at `2.2 us`.
- The displayed etalon time axis is shifted to start at `0` in `src/mock_lab/pipelines/etalon.py`.

### 3. Build the average baseline and shock absorbance spectra

The active shock preprocessing pipeline is `src/mock_lab/pipelines/shock_snapshot.py`.

This is where the baseline dataset and the shock dataset are combined.

Current behavior:

1. Load the saved etalon calibration from `data/interim/etalon/etalon_fit.npz`
2. Load the baseline MAT file and split it into sweeps
3. Average all baseline sweeps with `average_sweeps()` from `src/mock_lab/spectroscopy/absorbance.py`
4. Fit a straight line through the first and last `32` samples of that average baseline sweep with `fit_edge_line()`
5. Subtract that fitted line from the average baseline with `subtract_edge_line()`
6. Load the shock MAT file and split it into sweeps
7. Fit and subtract the same kind of wing line from every shock sweep with `fit_edge_lines()` and `subtract_edge_lines()`
8. Scale each corrected shock sweep so its peak matches the peak of the corrected average baseline with `scale_sweeps_to_reference_peak()`
9. Convert the sweep time axis into relative wavenumber by evaluating the etalon polynomial with `evaluate_relative_wavenumber()`
10. Select the absorbance-analysis window with `find_analysis_window()`
11. Compute Beer-Lambert absorbance with `beer_lambert_absorbance()`
12. Save the processed output to `data/processed/exports/shock_frequency_domain.npz`
13. Save the average baseline to `data/interim/baseline/baseline_average.npz`
14. Write the overlay and absorbance figures to `results/figures/`

The main absorbance helpers are all in `src/mock_lab/spectroscopy/absorbance.py`.

Current absorbance assumptions:

- The baseline correction is a line fit through the two sweep "wings" rather than a constant offset.
- The wing width is fixed at `32` samples on each end of the sweep.
- The corrected shock sweep is scaled so its peak equals the peak of the corrected average baseline sweep.
- Absorbance is computed as `A = -ln(I / I0)`, assuming the detector voltage is proportional to optical power.
- The analysis window begins at or after sample `32` where the corrected average baseline first reaches `1.0 V`.
- The analysis window ends when the corrected average baseline exceeds `2.75 V`.
- Outside that window, the absorbance array is left as `nan`.

Important output fields in `data/processed/exports/shock_frequency_domain.npz`:

- `time_axis_s`
- `relative_wavenumber_cm_inv`
- `corrected_baseline_sweep`
- `corrected_signal_sweeps`
- `scaled_signal_sweeps`
- `absorbance_sweeps`
- `scale_factors`
- `etalon_coefficients`

Files to edit if you want to change the shock preprocessing:

- Change the wing-fit or absorbance window logic: `src/mock_lab/spectroscopy/absorbance.py`
- Change how baseline and shock data are combined or which figures are generated: `src/mock_lab/pipelines/shock_snapshot.py`
- Change the time-domain or frequency-domain plot style: `src/mock_lab/plotting/figures.py`

Note on plotting:

- The baseline/shock overlay plot uses a time axis shifted to start at `0`.
- The shock absorbance time plot still uses the local sweep time values from the extracted analysis window.

### 4. Fit each absorbance sweep with three Voigt profiles

The current spectral model is implemented in `src/mock_lab/spectroscopy/voigt.py`, and the batch pipeline is `src/mock_lab/pipelines/voigt_fit.py`.

The transition metadata are hard-coded in `DEFAULT_CO_TRANSITIONS` in `src/mock_lab/spectroscopy/voigt.py` using the three lines from the handout:

- `P(0,31)`
- `P(2,20)`
- `P(3,14)`

The fitting model is:

- total absorbance = sum of three Voigt profiles + linear baseline

The current optimizer uses a constrained parameterization for each fitted sweep. The free quantities are:

- one shared temperature `temperature_k`
- one anchor center for `P(0,31)`
- one small center adjustment for `P(2,20)` relative to its nominal offset from `P(0,31)`
- one collisional half-width for `P(0,31)`
- one shared collisional half-width for `P(2,20)` and `P(3,14)`
- one integrated area for the strongest line `P(0,31)`
- one linear baseline offset
- one linear baseline slope

The stored fit result is still expanded back into explicit per-line centers, widths, and areas so the saved output files remain easy to inspect.

The current fixed ingredients are:

- the number of transitions: 3
- the transition metadata in `DEFAULT_CO_TRANSITIONS`
- the molecular mass used for Doppler broadening
- the Voigt profile functional form from `scipy.special.voigt_profile`
- optimizer bounds built in `_parameter_bounds()`
- the optimization method in `fit_voigt_spectrum()`
- the fixed center offset of `P(3,14)` from `P(0,31)`
- the shared collisional width constraint between `P(2,20)` and `P(3,14)`
- the temperature-dependent line-strength ratios that tie all three integrated areas to the strongest line area

The actual nonlinear fit is done with `scipy.optimize.least_squares` in `fit_voigt_spectrum()` using:

- `method="trf"`
- `loss="soft_l1"`
- `f_scale=0.01`
- `max_nfev=1500`

Uncertainty on the fitted quantities is currently estimated from the local least-squares Jacobian at the optimum:

- `src/mock_lab/spectroscopy/voigt.py` builds an approximate reduced-parameter covariance from the fitted Jacobian
- that covariance is propagated to the reported temperature, line centers, line widths, line areas, and mean apparent pressure with a local linearization
- the current exported intervals are approximate two-sided `95%` confidence intervals

How the fit is initialized:

- `estimate_initial_parameters()` first fits a line to the spectrum edges to estimate a baseline.
- The strongest peak in the baseline-corrected absorbance is used to seed the `P(0,31)` anchor center.
- The handout transition center spacings are used to seed the weaker lines.
- `P(2,20)` gets one local refinement near its nominal position.
- `P(3,14)` is kept at its nominal offset from the anchor line from the start.
- The strong-line and weak-line collisional width guesses are both initialized at `0.04 cm^-1`.
- The strongest-line area is seeded from an approximate peak-height-to-area conversion with `integrated_area_guess()`.
- The weaker line areas are then derived from temperature-dependent line-strength ratios instead of being guessed independently.

How sweep-to-sweep fitting works:

- `fit_voigt_spectra()` in `src/mock_lab/spectroscopy/voigt.py` loops over the absorbance sweeps in order.
- Sweeps with too little usable signal are skipped if their peak absorbance is below `minimum_peak_absorbance=0.02`.
- The previous successful fit is used as the initial guess for the next sweep.

This means the current fitter behaves like a sequential, warm-started least-squares pass rather than fully independent fits on every scan, while also keeping the three transition identities from swapping when the peaks overlap strongly.

Important fit outputs written by `src/mock_lab/pipelines/voigt_fit.py`:

- `data/processed/exports/voigt_fit_results.npz`
- `results/tables/voigt_fit_summary.csv`
- `results/figures/shock_voigt_fit.png`

The saved arrays include:

- fitted absorbance sweeps
- success flags
- fitted temperatures
- fitted temperature confidence intervals
- mean apparent pressures
- mean apparent pressure confidence intervals
- fitted line centers
- fitted line-center confidence intervals
- fitted collisional widths
- fitted collisional-width confidence intervals
- fitted line areas
- fitted line-area confidence intervals
- RMSE values

If you want the current Python fit to look more or less like the MATLAB demo, compare:

- `src/mock_lab/spectroscopy/voigt.py`
- `examples/matlab_voigt_demo/Demo_Absorbance_Fit.m`
- `examples/matlab_voigt_demo/Voigt_Approx_McLean_Vectorized_Fit_Demo.m`
- `examples/matlab_voigt_demo/get_IntArea_guess.m`

Files to edit if you want to change the spectral fit:

- Change the fixed transition list or spectroscopic constants: `src/mock_lab/spectroscopy/voigt.py`
- Change initial guesses, bounds, or fit parameterization: `src/mock_lab/spectroscopy/voigt.py`
- Change batch-fitting behavior or output tables: `src/mock_lab/pipelines/voigt_fit.py`
- Change the diagnostic fit figure: `src/mock_lab/plotting/figures.py`

### 5. Reduce the fit results to temperature, pressure, and CO mole fraction

The current state-reduction logic is in `src/mock_lab/spectroscopy/state_estimation.py`, and the wrapper pipeline is `src/mock_lab/pipelines/time_history.py`.

Current behavior:

1. Load the saved Voigt fit results from `data/processed/exports/voigt_fit_results.npz`
2. Use the fitted shared temperature directly as the scan temperature
3. Compute one apparent pressure for each line from its fitted collisional width with `apparent_pressure_atm()` in `src/mock_lab/spectroscopy/voigt.py`
4. Average those three apparent pressures per scan in `src/mock_lab/pipelines/voigt_fit.py`
5. Apply the handout-style correction factor `DEFAULT_PRESSURE_BROADENING_SCALE = 0.84` in `corrected_pressure_from_broadening()`
6. Estimate CO mole fraction from the integrated area of the strongest line only, using `estimate_co_mole_fraction()`
7. Propagate the saved fit covariance into approximate `95%` confidence intervals on temperature, corrected pressure, and CO mole fraction
8. Save the history arrays to `results/tables/state_history.csv` and `results/tables/state_history.npz`
9. Save the plot `results/figures/state_history.png`

Partition sums:

- The code no longer uses the earlier RRHO approximation.
- `src/mock_lab/spectroscopy/tips.py` reads the local HITRAN TIPS tables from `third_party/hitran_tips/`.
- `src/mock_lab/spectroscopy/voigt.py` now owns the temperature-dependent line-strength helper and uses the local TIPS data when evaluating `Q(T) / Q(T_ref)`.
- `src/mock_lab/spectroscopy/state_estimation.py` reuses that shared spectroscopy helper when converting fitted line area into CO mole fraction.

Current state-estimation assumptions:

- Pressure is inferred from fitted collisional widths using the `gamma_n2` and `n_n2` values stored in `DEFAULT_CO_TRANSITIONS`.
- The three line pressures are averaged before the `0.84` correction factor is applied.
- CO mole fraction is estimated from the strongest line area only.
- The optical path length is fixed by `DEFAULT_OPTICAL_PATH_LENGTH_CM = 10.32`.
- The ideal-gas number-density relation is used to map pressure and temperature to total number density.
- The exported uncertainty bands are local covariance-based intervals, not a full Bayesian posterior or bootstrap result.

Files to edit if you want to change the final derived quantities:

- Change partition-sum handling: `src/mock_lab/spectroscopy/tips.py`
- Change line-strength or mole-fraction calculation: `src/mock_lab/spectroscopy/state_estimation.py`
- Change the state-history pipeline outputs: `src/mock_lab/pipelines/time_history.py`
- Change the state-history plot: `src/mock_lab/plotting/figures.py`

### 6. Build a QC video of the fit progression

If you want a fast way to inspect the fit quality scan by scan, use:

- `scripts/run_voigt_fit_video.py`

This rebuilds the precursor products, fits all sweeps, and then renders a video with:

- measured absorbance
- fitted total spectrum
- the three component spectra
- dotted vertical center markers for the fitted line positions
- residual
- scan annotation with sweep index, fit status, fitted temperature, and RMSE

The video intentionally does not annotate pressure right now.

The video renderer lives in `src/mock_lab/pipelines/voigt_fit_video.py`, and the output is written to `results/videos/shock_voigt_fit_progression.mp4`.

## Where To Start If You Want To Change One Thing

Use this as the shortest route to the right file:

| If you want to change... | Start here |
|---|---|
| MAT variable loading or time reconstruction | `src/mock_lab/io/matlab_loader.py` |
| Where sweeps start and how incomplete ramps are discarded | `src/mock_lab/io/matlab_loader.py` |
| How the etalon peaks are detected | `src/mock_lab/spectroscopy/etalon_calibration.py` |
| The order of the etalon polynomial fit | `src/mock_lab/pipelines/etalon.py` |
| The etalon FSR constants | `src/mock_lab/spectroscopy/etalon_calibration.py` |
| How the baseline wing line is fit or subtracted | `src/mock_lab/spectroscopy/absorbance.py` |
| The absorbance voltage window (`1.0 V` to `2.75 V`) | `src/mock_lab/spectroscopy/absorbance.py` |
| The peak scaling between baseline and shock sweeps | `src/mock_lab/spectroscopy/absorbance.py` |
| How the shock preprocessing is wired together | `src/mock_lab/pipelines/shock_snapshot.py` |
| Which transitions are fit | `src/mock_lab/spectroscopy/voigt.py` |
| Which fit parameters are free and what bounds they have | `src/mock_lab/spectroscopy/voigt.py` |
| How weak scans are skipped | `src/mock_lab/spectroscopy/voigt.py` and `src/mock_lab/pipelines/voigt_fit.py` |
| How pressure is derived from linewidth | `src/mock_lab/spectroscopy/voigt.py` and `src/mock_lab/spectroscopy/state_estimation.py` |
| How CO mole fraction is derived from fitted area | `src/mock_lab/spectroscopy/state_estimation.py` |
| Partition sums and TIPS access | `src/mock_lab/spectroscopy/tips.py` |
| Plot formatting and labels | `src/mock_lab/plotting/figures.py` |
| Which sweep is shown in the report figures | `scripts/run_report_figures.py` |

## Current Output Files

The main generated products are:

- `data/interim/etalon/etalon_fit.npz`
- `data/interim/baseline/baseline_average.npz`
- `data/processed/exports/shock_frequency_domain.npz`
- `data/processed/exports/voigt_fit_results.npz`
- `results/figures/etalon_sweep.png`
- `results/figures/etalon_calibration.png`
- `results/figures/baseline_shock_overlay.png`
- `results/figures/shock_absorbance_time.png`
- `results/figures/shock_absorbance_frequency.png`
- `results/figures/shock_voigt_fit.png`
- `results/figures/state_history.png`
- `results/tables/voigt_fit_summary.csv`
- `results/tables/state_history.csv`
- `results/tables/state_history.npz`
- `results/videos/shock_voigt_fit_progression.mp4`

## Where The Three Transition Constants Live

The spectroscopic constants from the handout for `P(0,31)`, `P(2,20)`, and `P(3,14)` are currently stored in one place:

- `src/mock_lab/spectroscopy/voigt.py` in `DEFAULT_CO_TRANSITIONS`

Each `Transition` entry holds:

- `center_cm_inv`
- `line_strength_ref`
- `lower_state_energy_cm_inv`
- `gamma_n2_cm_inv_atm`
- `n_n2`
- `gamma_co_cm_inv_atm`
- `n_co`

Those values are used in two main parts of the code:

- `src/mock_lab/spectroscopy/voigt.py`
  - `center_cm_inv` is used for Doppler-width calculations and to seed the initial line-center guesses
  - `line_strength_ref`, `lower_state_energy_cm_inv`, and `center_cm_inv` are used in the temperature-dependent line-strength ratios that tie the fitted line areas together
  - `gamma_n2_cm_inv_atm` and `n_n2` are used to convert fitted Lorentz widths into apparent pressures
- `src/mock_lab/spectroscopy/state_estimation.py`
  - the strongest transition, `DEFAULT_CO_TRANSITIONS[0]`, is used by default when converting fitted integrated area into CO mole fraction
  - the actual line-strength evaluation is imported from `src/mock_lab/spectroscopy/voigt.py` so the spectroscopy stays consistent between fitting and post-processing

So if you want to edit the handout spectroscopic inputs themselves, start in `src/mock_lab/spectroscopy/voigt.py`. If you want to change how those constants are used after the fit, start in `src/mock_lab/spectroscopy/state_estimation.py`.

## Tests

The current tests live in `tests/` and can be run with:

```bash
.venv/bin/python -m pytest -q
```

They cover MAT loading, sweep extraction, TIPS access, Voigt fitting, and state estimation.

## Known Simplifications And Caveats

- `src/mock_lab/pipelines/baseline.py` is still a placeholder. The real baseline handling currently lives inside `src/mock_lab/pipelines/shock_snapshot.py`.
- `src/mock_lab/pipelines/full_pipeline.py` is still a placeholder.
- The etalon calibration is relative, not absolute. The fitted shock spectra live on a relative wavenumber axis.
- The current state history uses the strongest transition only for CO mole fraction.
- The current pressure reduction uses the `N2` broadening coefficients and a global `0.84` correction factor.
- The current fit structure is a pragmatic translation of the MATLAB example, but the optimizer is SciPy `least_squares`, not a custom hand-written gradient descent loop.

## Remaining Analysis TODO

- [ ] Decide whether the etalon polynomial order should stay at the current pipeline default or be changed again.
- [ ] Review the warm-start Voigt fit on the full sweep stack and decide whether specific scan ranges should be masked out before final reporting.
- [ ] Tighten the physical constraints in the state-reduction layer if you want pressure broadening and mole-fraction estimates to be less empirical.
- [ ] Move the active baseline logic out of `src/mock_lab/pipelines/shock_snapshot.py` into `src/mock_lab/pipelines/baseline.py` if you want a cleaner stage separation.
- [ ] Finalize the report figures and then fold the exact commands and file outputs into `report/README.md` if you want the reporting workflow documented separately.
