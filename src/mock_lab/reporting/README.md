# Reporting Helpers

`src/mock_lab/reporting/` contains small report-facing summaries built from the saved pipeline outputs.

## Current Scope

- `summary.py`: writes `results/reports/analysis_summary.md`

The reporting layer is intentionally thin. It should summarize what the pipeline produced, where the outputs were written, and a few headline counts or ranges. It should not duplicate the main analysis logic from `spectroscopy/` or `pipelines/`.

## Current Summary Output

`build_report_summary()` reads the saved deterministic outputs and reports:

- baseline sweep count
- etalon peak count
- shock sweep count
- successful Voigt-fit count
- min/max ranges for temperature, pressure, and CO mole fraction
- repo-relative paths to the main deterministic outputs

That file is meant to help a reader navigate the generated artifacts quickly after a run, not to replace the stage-specific documentation in the package READMEs.
