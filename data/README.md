# Data Layout

- `raw/` stores the original MAT files exactly as provided for the mock lab.
- `interim/` stores reusable stage outputs such as the averaged baseline sweep and etalon calibration.
- `processed/` stores durable exports such as the shock absorbance bundle and Voigt-fit results.

Guideline:

- Keep `raw/` read-only.
- Recompute anything in `interim/` and `processed/` from code.
