# ME 617 Mock Lab

This repository is the current Python reduction, fitting, and reporting workflow for the ME 617 scanned-wavelength direct-absorption mock lab. It takes the provided baseline, etalon, and shock-tube MATLAB datasets and reduces them into:

- phase-aligned sweeps
- a relative time-to-wavenumber calibration from the etalon scan
- baseline-corrected absorbance spectra
- constrained three-transition CO Voigt fits
- scan-by-scan temperature, pressure, and CO mole-fraction histories
- approximate 95% confidence intervals on the fitted and reported quantities

The code is not intended to be a general spectroscopy package. It is a pragmatic analysis codebase for this specific assignment, dataset structure, and transition set.

## What The Current Code Does

The active workflow is:

1. Load each MAT file and identify the detector and TTL-like reference channels.
2. Reconstruct the sample time axis from the reference waveform.
3. Split each continuous trace into phase-aligned sweeps, discarding the leading incomplete sweep.
4. Use the etalon dataset to build a relative frequency calibration.
5. Use the baseline dataset to build an average baseline sweep.
6. Convert each shock sweep into a baseline-corrected absorbance spectrum.
7. Fit each absorbance spectrum with a constrained sum of three CO Voigt profiles.
8. Convert the fitted parameters into temperature, pressure, and CO mole fraction.
9. Estimate local covariance-based 95% confidence intervals from the least-squares Jacobian and propagate them into the reported state variables.

The main raw inputs are:

- `data/raw/MockLabData_Baseline.mat`
- `data/raw/MockLabData_Etalon.mat`
- `data/raw/MockLabData_Shock.mat`

The local HITRAN partition-sum tables used by the spectroscopy code are vendored under:

- `third_party/hitran_tips/`

## Reduction Summary

### MAT Loading And Sweep Alignment

`src/mock_lab/io/matlab_loader.py` owns the raw-data handling.

The current implementation:

- loads each `.mat` file with `scipy.io.loadmat`
- identifies one signal channel and one reference channel
- thresholds the reference channel to recover a binary sweep marker
- detects rising edges and infers the sample period from the reference frequency
- assumes `DEFAULT_REFERENCE_FREQUENCY_HZ = 300000.0`
- applies a fixed phase offset of `DEFAULT_PHASE_START_S = 2.2e-6`
- reshapes the continuous traces into phase-aligned sweeps

The phase offset is intentional. The code throws away the partial leading sweep and keeps each extracted sweep starting with a small amount of baseline, then the ramp, then a small amount of trailing baseline. That is the working assumption throughout the rest of the repo.

### Etalon Calibration

The etalon reduction is implemented in:

- `src/mock_lab/spectroscopy/etalon_calibration.py`
- `src/mock_lab/pipelines/etalon.py`

The current etalon workflow:

- removes a simple edge baseline from each etalon sweep
- averages all corrected sweeps into one representative sweep
- detects the etalon peaks on that averaged trace
- assigns equally spaced relative wavenumbers using the fixed etalon free spectral range
- fits a polynomial mapping sweep time to relative wavenumber

The calibration is relative, not absolute. The shock spectra are therefore placed on a relative wavenumber axis rather than an absolute line-center scale.

At the moment the pipeline default in `src/mock_lab/pipelines/etalon.py` is a 4th-order polynomial fit.

### Baseline And Shock Absorbance Reduction

The shock preprocessing is implemented in:

- `src/mock_lab/spectroscopy/absorbance.py`
- `src/mock_lab/pipelines/shock_snapshot.py`

The current reduction choices are:

- baseline sweeps are averaged before use
- the average baseline is corrected with a straight line fit through the first and last 32 samples
- each shock sweep is corrected the same way
- each corrected shock sweep is scaled so its peak matches the peak of the corrected average baseline
- absorbance is computed as `A = -ln(I / I0)` under the assumption that detector voltage is linear in optical power
- the absorbance window starts once the corrected average baseline reaches 1.0 V and ends once it exceeds 2.75 V

This means the code is intentionally excluding the low-signal early part of the ramp, where the absorbance estimate becomes noisy because the signal-to-noise ratio is poor and no useful absorbance should be reported there anyway.

## Spectral Fit Summary

The spectral fit is implemented in:

- `src/mock_lab/spectroscopy/voigt.py`
- `src/mock_lab/pipelines/voigt_fit.py`

The active Python implementation uses:

- `scipy.special.voigt_profile` for direct Voigt line-shape evaluation
- `scipy.optimize.least_squares` for the nonlinear fit

The legacy MATLAB McLean approximation is kept only in `examples/matlab_voigt_demo/` as a reference from the course handout. It is not used by the Python workflow.

### Transitions Used

The three handout transitions are hard-coded in `DEFAULT_CO_TRANSITIONS` inside `src/mock_lab/spectroscopy/voigt.py`:

- `P(0,31)`
- `P(2,20)`
- `P(3,14)`

Each transition stores:

- `center_cm_inv`
- `line_strength_ref`
- `lower_state_energy_cm_inv`
- `gamma_n2_cm_inv_atm`
- `n_n2`
- `gamma_co_cm_inv_atm`
- `n_co`

`line_strength_ref` is not kept in raw HITRAN units. The conversion to
`cm^-2/atm` is applied immediately in
`src/mock_lab/spectroscopy/voigt.py` by
`hitran_line_strength_to_cm2_atm()`, using

- `S_ref = S_hitran * P_atm / (k_B * T_ref)`

with `P_atm = 1.01325e6 dyn/cm^2` and `k_B` in CGS. At `296 K`, this is the
same as multiplying by approximately `2.4794e19`.

So from the point where `DEFAULT_CO_TRANSITIONS` is defined onward, the code
stores and uses reference line strengths in `cm^-2/atm`.

These values drive both the fit model and the later state-reduction formulas.

### Model Form

Each line is modeled as

- `A_i(nu) = Area_i * Voigt(nu - nu0_i, sigmaD_i(T), gammaL_i)`

where:

- `nu0_i` is the line center
- `sigmaD_i(T)` is the Doppler Gaussian width set by the shared fitted temperature
- `gammaL_i` is the collisional Lorentz half-width
- `Area_i` is the integrated absorbance area of the line

The total model is the sum of the three line absorbances plus a linear baseline.

### Why The Fit Is Constrained

The current fitter is intentionally not a fully free three-line Voigt fit.

When all three centers, widths, and areas are allowed to float independently, the optimizer tends to swap which transition is assigned to the dominant peak and which transition is assigned to the smaller shoulder. The current implementation constrains the model to stop that line-identity swapping.

The present reduced parameterization is:

- one shared temperature `temperature_k`
- one anchor center for `P(0,31)`
- one small center adjustment for `P(2,20)` relative to its nominal offset from `P(0,31)`
- one collisional half-width for `P(0,31)`
- one shared collisional half-width for `P(2,20)` and `P(3,14)`
- one integrated area for `P(0,31)`
- one linear baseline offset
- one linear baseline slope

The explicit stored line parameters are expanded back out after the fit, so the saved results still contain all three centers, widths, and areas even though the optimizer works on the smaller constrained vector.

### Why Only One Free Integrated Area Is Fitted

Only the strongest line area, `line_areas[0]`, is fitted freely.

That is a deliberate modeling choice, not an omission.

The reasoning is:

- the three transitions strongly overlap
- the weaker lines are much more prone to unstable area swapping if they are left fully free
- the relative line strengths are not arbitrary once temperature is specified

So the code uses the spectroscopy to tie the three areas together:

- `line_strength_at_temperature()` in `src/mock_lab/spectroscopy/voigt.py` evaluates the temperature-dependent line strength for each transition in `cm^-2/atm`
- `transition_strength_ratios()` converts those line strengths into ratios relative to `P(0,31)`
- the optimizer fits only the strongest-line area
- the other two line areas are derived from that strongest-line area and the temperature-dependent line-strength ratios

So the one fitted area should be read as the overall integrated absorbance scale of the CO spectrum for that sweep. It is not itself temperature, pressure, or mole fraction. It becomes useful for the final reported state only after the code combines it with the fitted temperature, corrected pressure, the `cm^-2/atm` line-strength model, and the optical path length.

### Which Free Parameters Drive Which Final Quantities

The main fit quantities do not all feed the final reported state variables in the same way.

`temperature_k`

- directly becomes the reported scan temperature
- sets the Doppler width of every line
- sets the temperature-dependent line-strength ratios
- enters the linewidth-to-pressure conversion through the broadening temperature law
- enters the mole-fraction calculation through the temperature-dependent line strength

`collisional_hwhm_cm_inv`

- is converted to per-line apparent pressure in `apparent_pressure_atm()`
- is the main driver of the reported pressure
- only affects mole fraction indirectly through the pressure estimate

`line_areas`

- are integrated absorbance areas, not temperatures or pressures
- `line_areas[0]` is the magnitude parameter that ultimately feeds CO mole fraction
- the weaker line areas are derived from `line_areas[0]` and the temperature-dependent ratios

`line_centers_relative_cm_inv`

- primarily improve the alignment of the fit to the measured spectrum
- matter for fit quality and diagnostic interpretation
- are not used directly in the present `T`, `P`, or `X_CO` calculations

`baseline_offset` and `baseline_slope`

- are nuisance parameters that absorb small broadband baseline mismatch
- are not used directly in the final reported state variables

### Current Structural Assumptions In The Fit

The current fit also enforces two handout-motivated structural constraints:

- `P(3,14)` stays at a fixed center offset from `P(0,31)`
- `P(2,20)` and `P(3,14)` share the same collisional width

So the current Python fit is best thought of as a physically stabilized three-line fit, not a completely free one.

### Initialization And Sweep Ordering

The fit is initialized by:

- estimating a linear baseline from the spectrum edges
- seeding the anchor center from the dominant measured peak
- seeding `P(2,20)` near its nominal offset and refining it locally
- keeping `P(3,14)` at its nominal offset from the anchor line
- seeding the strongest line area from an approximate peak-to-area conversion
- deriving the weaker areas from the temperature-dependent strength ratios

The batch fit is warm-started:

- weak spectra can be skipped if their peak absorbance is below the configured threshold
- the previous successful sweep fit is used as the initial guess for the next sweep

That is why the fit behaves as a sequential time-history reduction rather than a collection of independent single-spectrum fits.

## State-Reduction Summary

The state-reduction code is implemented in:

- `src/mock_lab/spectroscopy/state_estimation.py`
- `src/mock_lab/pipelines/time_history.py`

The current reported quantities are built as follows:

- temperature is taken directly from the fitted shared temperature
- apparent pressure is computed from the fitted collisional widths using the tabulated `gamma_n2` and `n_n2` values
- the three apparent pressures are averaged
- the mean apparent pressure is scaled by `1 / 0.84` to form the reported pressure
- CO mole fraction is estimated from the strongest fitted line area only

The strongest-line area is converted to mole fraction using:

- the fitted temperature
- the corrected pressure
- the optical path length
- the temperature-dependent line strength `S(T)`

Because the stored line strengths are already in `cm^-2/atm`, the reduction is
written directly as

- `Area = S(T) * P * X_CO * L`

with:

- `Area` in `cm^-1`
- `S(T)` in `cm^-2/atm`
- `P` in `atm`
- `L` in `cm`

So the reported mole fraction is computed in
`src/mock_lab/spectroscopy/state_estimation.py` as

- `X_CO = Area / (S(T) * P * L)`

The current optical path length is fixed in `src/mock_lab/spectroscopy/state_estimation.py` as `DEFAULT_OPTICAL_PATH_LENGTH_CM = 10.32`.

## TIPS And Spectroscopy Support

The local partition sums are accessed through:

- `src/mock_lab/spectroscopy/tips.py`

which reads the vendored HITRAN TIPS resources from:

- `third_party/hitran_tips/`

The line-strength model used in the fit and in the mole-fraction calculation is
therefore consistent with the same local TIPS data and the same handout
transition metadata. The temperature scaling applied in
`line_strength_at_temperature()` is

- `S(T) = S(T0) * Q(T0)/Q(T) * T0/T * exp[-c2 E'' (1/T - 1/T0)]`
- `       * (1 - exp[-c2 nu0 / T]) / (1 - exp[-c2 nu0 / T0])`

which matches the pressure-normalized `cm^-2/atm` convention used by the rest
of the reduction.

## Uncertainty Summary

The current uncertainty treatment is a local covariance approach based on the fitted Jacobian.

In `src/mock_lab/spectroscopy/voigt.py`:

- the `least_squares` Jacobian is used to estimate a reduced-parameter covariance matrix
- that covariance is propagated to the expanded fit quantities with a local linearization
- approximate two-sided 95% confidence intervals are exported for temperature, line centers, collisional widths, line areas, and mean apparent pressure

In `src/mock_lab/pipelines/time_history.py`:

- the saved fit covariance is propagated into the reported temperature, corrected pressure, and CO mole fraction
- the state-history figure is drawn with shaded 95% confidence bands

This is a reasonable first uncertainty estimate, but it is still a local linearized approximation. It is not a bootstrap, Monte Carlo, or full Bayesian posterior.

## Main Generated Outputs

The current workflow writes the main intermediate and report products to:

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

## Code Ownership By Module

If one part of the workflow needs to change, these are the primary files to touch:

- `src/mock_lab/io/matlab_loader.py`: MAT loading, timing reconstruction, sweep cutting
- `src/mock_lab/spectroscopy/etalon_calibration.py`: etalon peak finding and polynomial calibration
- `src/mock_lab/spectroscopy/absorbance.py`: baseline wing correction, scaling, absorbance windowing
- `src/mock_lab/spectroscopy/voigt.py`: spectral model, fit parameterization, uncertainty on fit quantities
- `src/mock_lab/spectroscopy/state_estimation.py`: pressure and mole-fraction reduction
- `src/mock_lab/spectroscopy/tips.py`: local partition-sum access
- `src/mock_lab/pipelines/shock_snapshot.py`: end-to-end baseline/shock preprocessing
- `src/mock_lab/pipelines/voigt_fit.py`: batch fitting, fit-result export, fit summary table
- `src/mock_lab/pipelines/time_history.py`: propagated uncertainty on `T`, `P`, and `X_CO`
- `src/mock_lab/plotting/figures.py`: all report-style plotting

## Wrapper Scripts

The main entry points are:

- `scripts/run_report_figures.py`
- `scripts/run_voigt_fit.py`
- `scripts/run_time_history.py`
- `scripts/run_voigt_fit_video.py`

They are thin wrappers around the underlying package code and are the easiest place to change dataset paths, output locations, or which representative sweep is shown in the figures.

## Current Limitations

- `src/mock_lab/pipelines/baseline.py` is still a placeholder; the active baseline handling currently lives in `src/mock_lab/pipelines/shock_snapshot.py`
- `src/mock_lab/pipelines/full_pipeline.py` is still a placeholder
- the etalon calibration is relative rather than absolute
- the pressure reduction remains empirical because it depends on fitted linewidths and the global `0.84` scaling
- the mole-fraction reduction currently uses only the strongest transition area
- the uncertainty bands are local covariance bands, not a full stochastic uncertainty analysis
