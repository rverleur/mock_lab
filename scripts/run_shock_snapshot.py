"""Run one shock-spectrum analysis pass from an IDE or terminal."""

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mock_lab.pipelines.shock_snapshot import run_shock_snapshot_pipeline


def main() -> None:
    """Wire local paths into the single-spectrum shock pipeline."""

    raw_data = REPO_ROOT / "data" / "raw" / "MockLabData_Shock.mat"
    baseline_dir = REPO_ROOT / "data" / "interim" / "baseline"
    etalon_dir = REPO_ROOT / "data" / "interim" / "etalon"
    output_dir = REPO_ROOT / "data" / "processed" / "exports"
    run_shock_snapshot_pipeline(
        raw_data=raw_data,
        baseline_dir=baseline_dir,
        etalon_dir=etalon_dir,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()
