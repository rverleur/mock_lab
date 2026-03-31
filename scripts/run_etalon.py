"""Run etalon calibration from an IDE or terminal."""

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mock_lab.pipelines.etalon import run_etalon_pipeline


def main() -> None:
    """Wire local paths into the etalon-calibration pipeline."""

    raw_data = REPO_ROOT / "data" / "raw" / "MockLabData_Etalon.mat"
    output_dir = REPO_ROOT / "data" / "interim" / "etalon"
    run_etalon_pipeline(raw_data=raw_data, output_dir=output_dir)


if __name__ == "__main__":
    main()
