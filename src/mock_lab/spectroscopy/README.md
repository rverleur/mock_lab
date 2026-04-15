# Spectroscopy Guide

`src/mock_lab/spectroscopy/` contains the physical reduction logic: baseline correction, etalon calibration, spectroscopy metadata, Voigt fitting, and the state-reduction formulas used to report `T`, `P`, and `X_CO`.

## Baseline And Absorbance Reduction

`absorbance.py` handles the preprocessing shared by the baseline and shock stages.

The active reduction choices are:

- baseline sweeps are averaged before use
- a straight line is fit through the first and last `32` samples of the average baseline sweep
- each shock sweep gets the same style of edge-line correction independently
- each corrected shock sweep is scaled so its peak matches the peak of the corrected average baseline
- absorbance is evaluated as `A = -ln(I / I0)` and is only reported where both signals stay positive
- the default analysis window begins once the corrected baseline reaches `1.0 V` and ends before it exceeds `2.75 V`

That windowing is deliberate. It suppresses the low-signal early part of the ramp where the absorbance estimate is poorly conditioned.

## Etalon Calibration

`etalon_calibration.py` turns the etalon trace into a relative wavenumber axis.

The workflow is:

1. Remove a simple edge baseline from each etalon sweep.
2. Average the corrected sweeps into one representative trace.
3. Detect etalon peaks on that representative sweep with `scipy.signal.find_peaks`.
4. Assign equally spaced relative wavenumbers using the fixed free spectral range
   `1 / (2 n L)`, with the local constants `ETALON_REFRACTIVE_INDEX` and `ETALON_LENGTH_CM`.
5. Fit a polynomial from sweep time to relative wavenumber.

The saved calibration is relative, not absolute. The shock spectra are therefore fitted on a relative wavenumber axis. The current etalon pipeline defaults to a 4th-order polynomial.

## Spectroscopy Assets And Units

Three supporting modules provide the spectroscopy inputs used by both the fitter and the state reduction:

- `hitemp.py`: parses fixed-width HiTEMP/HITRAN `.par` records and decodes uncertainty metadata.
- `tips.py`: reads the vendored HITRAN TIPS resources from `third_party/hitran_tips/`.
- `collisional_broadening.py`: stores the active species-resolved broadening coefficients and the N2-equivalent bath-gas helpers used by the pressure model.

### Transitions Used

The active fit uses three CO transitions, exposed as `DEFAULT_CO_TRANSITIONS` in `voigt.py`:

- `P(0,31)`
- `P(2,20)`
- `P(3,14)`

At runtime the code prefers the curated file:

- `third_party/HiTEMP/mock_lab_co_transitions.csv`

If that file is missing, it falls back to the full local HiTEMP `.par` asset:

- `third_party/HiTEMP/05_HITEMP2019.par`

Each `Transition` combines:

- line-center and line-strength metadata from HiTEMP
- decoded uncertainty information from the HiTEMP uncertainty codes
- paper-table species-resolved broadening coefficients and temperature exponents for `CO`, `N2`, `CO2`, `H2O`, `OH`, `H2`, `O2`, `O`, and `H`

### Line-Strength Convention

The raw HiTEMP line strengths are stored in HITRAN units. `voigt.py` converts them immediately to `cm^-2/atm` at `T_REF_K = 296 K` using

`S_ref = S_hitran * P_atm / (k_B * T_ref)`

with `P_atm` in CGS units. From the point where `DEFAULT_CO_TRANSITIONS` is built onward, the active code uses the pressure-normalized `cm^-2/atm` convention consistently.

`line_strength_at_temperature()` then applies the local TIPS partition sums and lower-state energy correction so both the fit and the state reduction share the same `S(T)` model.

## Voigt Model

`voigt.py` contains the active three-line spectral model. Each transition is modeled as

`A_i(nu) = Area_i * Voigt(nu - nu0_i, sigma_D_i(T), gamma_L_i)`

and the total model is the sum of the three line absorbances plus a linear baseline.

The implementation uses:

- `scipy.special.voigt_profile` for direct line-shape evaluation
- `scipy.optimize.least_squares` for nonlinear fitting

The legacy MATLAB McLean approximation is retained only under `examples/` for reference and is not used anywhere in the Python workflow.

## Why The Fit Is Constrained

The active fitter is intentionally not a fully free three-line fit. When all centers, widths, and areas float independently, the optimizer can swap line identities between the dominant peak and the smaller shoulder. The reduced parameterization keeps the fit physically stable sweep-to-sweep.

The fitted parameter vector currently contains:

- one shared temperature `temperature_k`
- one anchor line center for `P(0,31)`
- one small center adjustment for `P(2,20)` near its nominal offset from the anchor
- one collisional HWHM for `P(0,31)`
- one shared collisional HWHM for `P(2,20)` and `P(3,14)`
- one free integrated area for `P(0,31)`
- one linear baseline offset
- one linear baseline slope

The explicit per-line centers, widths, and areas are expanded back out after the fit and saved in the exported results file.

### Structural Constraints That Remain Active

- `P(3,14)` stays at a fixed center offset from `P(0,31)`.
- `P(2,20)` and `P(3,14)` share one collisional width.
- `P(0,31)` is treated as the anchor transition and remains the strongest fitted line.

### Why Only One Free Area Is Fitted

Only the strongest line area is fitted freely. The weaker-line areas are derived from temperature-dependent line-strength ratios, which makes the overlapping three-line fit much more stable. In practice that means the free area parameter is the overall absorbance scale for the sweep, not a direct state variable by itself.

## Which Fitted Quantities Control The Final State

- `temperature_k`: directly becomes the reported temperature, sets Doppler width, changes the line-strength ratios, and enters both the pressure and mole-fraction reductions.
- `collisional_hwhm_cm_inv`: is the main input to the pressure solve after conversion from Lorentz HWHM to collisional FWHM.
- `line_areas[0]`: is the strongest-line integrated absorbance area and is the main absorbance quantity used to recover `X_CO`.
- `line_centers_relative_cm_inv`: mainly improve fit quality and diagnostics; they are not used directly in the current `T`, `P`, or `X_CO` calculations.
- `baseline_offset` and `baseline_slope`: are nuisance parameters only.

## State Reduction

`state_estimation.py` converts the fitted temperature, linewidth, and strongest-line area into reported state variables.

The active reduction is:

- temperature is taken directly from the shared fitted temperature
- pressure and `X_CO` are solved simultaneously from the strongest-line linewidth and strongest-line area
- the pressure model uses a two-partner mixture: `CO` plus one N2-equivalent bath gas representing every non-CO collider
- the tabulated broadening coefficients are multiplied by the handout `0.84` scale factor inside the solve
- the optical path length is fixed at `DEFAULT_OPTICAL_PATH_LENGTH_CM = 10.32`

The strongest-line solve uses:

- `Delta_nu_C = 2 P * 0.84 * gamma_eff(T, X_CO)`
- `Area = S(T) * P * X_CO * L`

with

- `gamma_eff(T, X_CO) = X_CO * gamma_CO(T) + (1 - X_CO) * gamma_bath(T)`
- `gamma_k(T) = gamma_k,ref * (T_ref / T)^n_k`

This means line-strength uncertainty can influence pressure as well as mole fraction, because `P` and `X_CO` are solved together rather than one after the other.

## Key Assumptions

These are the assumptions that matter most for interpretation of the reported results:

- the etalon calibration is relative, not absolute
- the fit is a constrained three-line model, not a fully free three-line decomposition
- the weaker line areas are tied to the strongest line through spectroscopy-based ratios
- the pressure reduction treats every non-CO collision partner as N2-equivalent
- the final `P` and `X_CO` solve uses only the strongest transition directly
- the optical path length is fixed at `10.32 cm`
