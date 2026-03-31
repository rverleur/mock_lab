"""Time-history placeholder pipeline for Regions 2 and 5."""

from pathlib import Path


def run_time_history_pipeline(
    raw_data: Path,
    baseline_dir: Path,
    etalon_dir: Path,
    output_dir: Path,
) -> None:
    """Extend the single-spectrum workflow across the shock time history.

    Planned responsibilities:
    - iterate across scans or windows in Regions 2 and 5
    - reuse the same baseline and frequency-calibration products
    - fit each selected spectrum
    - save temperature, pressure, and composition histories
    """

    raise NotImplementedError("Implement time-history processing in this module.")
