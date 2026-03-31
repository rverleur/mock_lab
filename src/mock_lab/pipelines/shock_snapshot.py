"""Single-spectrum shock-processing placeholder pipeline."""

from pathlib import Path


def run_shock_snapshot_pipeline(
    raw_data: Path,
    baseline_dir: Path,
    etalon_dir: Path,
    output_dir: Path,
) -> None:
    """Process one usable shock-tube spectrum and recover fit parameters.

    Planned responsibilities:
    - load the shock experiment
    - apply baseline and etalon calibration products
    - isolate one useful Region 5 transmission window
    - convert transmission to absorbance
    - fit the spectrum and export snapshot-level results
    """

    raise NotImplementedError("Implement single-spectrum shock analysis in this module.")
