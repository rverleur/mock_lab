"""Extract the vendored CO HiTEMP `.par` line list into a CSV file."""

from __future__ import annotations

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mock_lab.spectroscopy.hitemp import write_hitemp_par_csv


DEFAULT_PAR_FILE = REPO_ROOT / "third_party" / "HiTEMP" / "05_HITEMP2019.par"
DEFAULT_CSV_FILE = REPO_ROOT / "third_party" / "HiTEMP" / "05_HITEMP2019.csv"


def main(
    input_path: Path = DEFAULT_PAR_FILE,
    output_path: Path = DEFAULT_CSV_FILE,
) -> None:
    """Convert the configured HiTEMP `.par` file and report the row count."""

    row_count = write_hitemp_par_csv(input_path, output_path)
    print(f"Wrote {row_count} rows to {output_path}")


if __name__ == "__main__":
    input_path = DEFAULT_PAR_FILE
    output_path = DEFAULT_CSV_FILE

    main(
        input_path=input_path,
        output_path=output_path,
    )
