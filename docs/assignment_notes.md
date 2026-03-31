# Assignment Notes

## Objective

Use scanned-wavelength direct-absorption measurements of CO to estimate gas temperature, pressure, and CO mole fraction in a high-temperature shock-tube experiment.

## Provided Experiments

The handout describes three datasets:

- Baseline data with no CO present.
- Etalon data for converting scan time to relative frequency.
- Shock-tube data containing the actual absorption measurement of interest.

Each file is described as containing both the detector signal and a TTL/reference signal from the function generator. The reference signal is intended to help align datasets that may be sampled on slightly different grids.

## Required Analysis Flow

1. Process the etalon scan to identify useful peaks and fit a time-to-frequency mapping.
2. Use the baseline data to handle background emission and transmission normalization.
3. Select the useful shock-tube transmission region after the laser scan turnaround.
4. Convert transmission to absorbance.
5. Fit the absorbance spectrum with Voigt profiles for the CO transitions discussed in the handout.
6. Recover temperature, pressure, and CO mole fraction from the fit.
7. Extend the single-spectrum workflow to a time history through Regions 2 and 5.

## Deliverables Called Out In The Handout

- Working data-reduction code for the supplied experiments.
- Figures that mirror the tutorial checkpoints in the handout:
  raw etalon with identified peaks, polynomial frequency fit, corrected transmission, absorbance, and measured-versus-fit spectra.
- A technical report covering theory, setup/procedure, results, uncertainty, and conclusions.

## Technical Reminders From The Handout

- The target CO transitions are the same three transitions highlighted in the spectroscopic-parameter table in the handout.
- The handout suggests using the same Doppler width for both transitions during fitting and starting with a value consistent with about 3750 K until the first successful temperature history is recovered.
- The handout notes that the tabulated broadening values come from lower-temperature data and suggests scaling the collisional widths by `0.84` when using them to estimate pressure.
- The strongest transition may dominate the fit; the handout suggests constraining one line center relative to another if needed.
- The handout allows the same broadening coefficients to be used for CO-CO2 broadening as for CO-N2 broadening.
