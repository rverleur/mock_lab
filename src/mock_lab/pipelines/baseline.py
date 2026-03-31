"""Baseline-processing placeholder pipeline."""

from pathlib import Path


def run_baseline_pipeline(raw_data: Path, output_dir: Path) -> None:
    """Prepare no-CO baseline products for downstream shock processing.

    Planned responsibilities:
    - load baseline detector and reference channels
    - align the signals in time or sample index
    - estimate background emission and baseline intensity terms
    - save reusable corrections under ``data/interim/baseline/``
    """

    raise NotImplementedError("Implement baseline preprocessing in this module.")
