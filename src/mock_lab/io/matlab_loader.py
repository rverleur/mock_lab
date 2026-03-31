"""Placeholders for loading MATLAB-formatted experiment files in Python."""

from pathlib import Path


def load_mat_file(path: Path) -> dict:
    """Load one assignment MAT file and return a normalized dictionary.

    Planned responsibilities:
    - read MATLAB v5 files in Python
    - standardize variable names across baseline, etalon, and shock cases
    - expose detector and TTL/reference channels in a consistent shape

    This project skeleton intentionally does not implement the loader yet.
    """

    raise NotImplementedError(
        "Implement MAT-file loading here, likely with scipy.io.loadmat or an equivalent reader."
    )
