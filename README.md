# ME 617 Mock Lab

Python-first workspace for the ME 617 scanned-wavelength direct-absorption mock lab. The assignment is built around three provided datasets: a baseline case with no CO present, an etalon scan for frequency calibration, and a shock-tube experiment used to recover CO temperature, pressure, and mole fraction in a high-temperature shock tube.

This repository has been reorganized into a GitHub-friendly structure without implementing the analysis itself yet. The Python files in `scripts/` and `src/mock_lab/` are placeholders that show where each processing step should live and how the stages should be wired together from an IDE or terminal.

## What This Project Does

- Organizes the course handout, reference paper, raw MAT files, demo videos, and legacy MATLAB example code.
- Sets up a Python package layout for loading MAT data, calibrating the etalon scan, computing absorbance, fitting spectra, and generating report-ready outputs.
- Provides wrapper scripts with `if __name__ == "__main__":` entry points so you can run each stage directly while you build out the underlying utilities.

## Handout TODO

- [ ] Review the provided baseline, etalon, and shock datasets and confirm the detector and TTL reference channels available in each file.
- [ ] Implement MAT-file loading and a consistent path/config setup for the three experiments.
- [ ] Build the etalon-processing workflow shown in the handout: identify usable peaks, handle scan turnaround, and map scan time to relative frequency with a polynomial fit.
- [ ] Build the baseline-processing workflow so the no-CO experiment can be reused for emission subtraction and laser-intensity correction.
- [ ] Isolate the useful shock-tube transmission window, subtract the baseline/background contribution, and convert transmission to absorbance with Beer's law.
- [ ] Implement the spectral-fit routine for the CO transitions called out in the handout, using a Voigt-profile model and the provided spectroscopic parameters.
- [ ] Recover temperature, pressure, and CO mole fraction for a single usable Region 5 spectrum before extending the code to a time history.
- [ ] Extend the analysis to process the shock-tube time history through Regions 2 and 5 and save the resulting state histories.
- [ ] Recreate the checkpoint-style figures implied by the handout: etalon calibration, baseline-corrected transmission, absorbance, and measured-versus-fit spectra.
- [ ] Assemble the technical report with theory, experimental setup, data-reduction procedure, results, uncertainty discussion, and conclusions.

## Suggested Development Order

1. Start in `src/mock_lab/io/matlab_loader.py` and decide how you want to load MATLAB v5 files in Python.
2. Implement the reusable spectroscopy utilities in `src/mock_lab/spectroscopy/`.
3. Wire those utilities into stage-level pipelines in `src/mock_lab/pipelines/`.
4. Use the wrappers in `scripts/` as your IDE entry points while developing and debugging.
5. Save intermediate products under `data/interim/`, polished outputs under `data/processed/`, and report-ready figures/tables under `results/`.

## Repository Layout

```text
.
|-- README.md
|-- pyproject.toml
|-- requirements.txt
|-- assets/
|   `-- videos/               # Local demo footage (ignored by git by default)
|-- data/
|   |-- raw/                  # Original MAT files from the assignment
|   |-- interim/              # Reusable intermediate outputs
|   `-- processed/            # Final data products exported by your code
|-- docs/
|   |-- handouts/             # Original course handout
|   |-- references/           # Background/reference paper
|   `-- assignment_notes.md   # Condensed notes from the handout
|-- examples/
|   `-- matlab_voigt_demo/    # Legacy MATLAB reference code
|-- notebooks/                # Optional scratch analysis
|-- report/                   # LaTeX report source files
|-- results/
|   |-- figures/
|   |-- reports/
|   `-- tables/
|-- scripts/                  # Thin wrapper entry points for IDE runs
|-- src/
|   `-- mock_lab/             # Actual Python package
`-- tests/                    # Future unit/integration tests
```

## Running The Placeholder Wrappers

The wrappers are set up for direct execution from an IDE or terminal. Right now they only define the intended entry points and route to placeholder functions that raise `NotImplementedError`.

Example order:

1. `python scripts/run_baseline.py`
2. `python scripts/run_etalon.py`
3. `python scripts/run_shock_snapshot.py`
4. `python scripts/run_time_history.py`
5. `python scripts/run_full_pipeline.py`

## Python Environment

The local virtual environment for this repo is expected at `.venv/`.

Typical activation commands:

- macOS/Linux: `source .venv/bin/activate`

Core packages for this project are listed in `requirements.txt`.

## Report Writing

The LaTeX source for the technical report lives in `report/`.

- Main source file: `report/main.tex`
- Recommended build command: `latexmk -xelatex main.tex`

The template is configured to match the handout requirements with 12 pt Times New Roman, 1 inch margins, and single spacing.

## Notes

- The demo videos are preserved locally under `assets/videos/`, but `.mov` files are ignored in git because one file is larger than standard GitHub file-size limits. If you want those in a remote repo later, use Git LFS.
- The MATLAB demo in `examples/matlab_voigt_demo/` is kept only as a reference for translating the fitting approach into Python.
- The handout summary and technical reminders live in `docs/assignment_notes.md`.
