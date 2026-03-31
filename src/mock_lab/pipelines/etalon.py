"""Etalon-calibration placeholder pipeline."""

from pathlib import Path


def run_etalon_pipeline(raw_data: Path, output_dir: Path) -> None:
    """Build the time-to-frequency calibration from the etalon scan.

    Planned responsibilities:
    - load etalon detector and reference channels
    - identify valid peaks and ignore the turnaround region
    - fit a calibration curve that maps scan time to relative frequency
    - save calibration products under ``data/interim/etalon/``
    """

    raise NotImplementedError("Implement etalon calibration in this module.")
