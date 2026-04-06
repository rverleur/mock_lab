"""Run Region 2 and Region 5 history processing from an IDE or terminal."""

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mock_lab.pipelines.time_history import run_time_history_pipeline
from mock_lab.pipelines.etalon import run_etalon_pipeline
from mock_lab.pipelines.shock_snapshot import run_shock_snapshot_pipeline
from mock_lab.pipelines.voigt_fit import run_voigt_fit_pipeline


def main() -> None:
    """Wire local paths into the time-history pipeline."""

    raw_data = REPO_ROOT / "data" / "raw" / "MockLabData_Shock.mat"
    etalon_data = REPO_ROOT / "data" / "raw" / "MockLabData_Etalon.mat"
    baseline_dir = REPO_ROOT / "data" / "interim" / "baseline"
    etalon_dir = REPO_ROOT / "data" / "interim" / "etalon"
    shock_output_dir = REPO_ROOT / "data" / "processed" / "exports"
    figure_dir = REPO_ROOT / "results" / "figures"
    table_dir = REPO_ROOT / "results" / "tables"

    run_etalon_pipeline(
        raw_data=etalon_data,
        output_dir=etalon_dir,
        figure_output_dir=figure_dir,
    )
    run_shock_snapshot_pipeline(
        raw_data=raw_data,
        baseline_dir=baseline_dir,
        etalon_dir=etalon_dir,
        output_dir=shock_output_dir,
        figure_output_dir=figure_dir,
    )
    run_voigt_fit_pipeline(
        shock_frequency_data=shock_output_dir / "shock_frequency_domain.npz",
        output_dir=shock_output_dir,
        figure_output_dir=figure_dir,
        table_output_dir=table_dir,
    )
    run_time_history_pipeline(
        voigt_fit_data=shock_output_dir / "voigt_fit_results.npz",
        output_dir=table_dir,
        figure_output_dir=figure_dir,
    )


if __name__ == "__main__":
    main()
