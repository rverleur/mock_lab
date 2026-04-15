from pathlib import Path

from mock_lab.pipelines.baseline import run_baseline_pipeline
from mock_lab.pipelines.full_pipeline import run_full_pipeline


def _symlink_raw_data(raw_data_dir: Path, destination: Path) -> None:
    """Mirror the raw MAT inputs into a temporary data root via symlinks."""

    destination.mkdir(parents=True, exist_ok=True)
    for mat_path in raw_data_dir.glob("*.mat"):
        (destination / mat_path.name).symlink_to(mat_path)


def test_run_baseline_pipeline_exports_average_products(
    raw_data_dir: Path,
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "baseline"
    figure_dir = tmp_path / "figures"

    result = run_baseline_pipeline(
        raw_data=raw_data_dir / "MockLabData_Baseline.mat",
        output_dir=output_dir,
        figure_output_dir=figure_dir,
    )

    assert result.sweep_count > 0
    assert result.average_baseline_sweep.shape == result.corrected_baseline_sweep.shape
    assert result.time_axis_s.shape == result.average_baseline_sweep.shape
    assert (output_dir / "baseline_average.npz").is_file()
    assert (figure_dir / "baseline_average.png").is_file()


def test_run_full_pipeline_smoke(
    raw_data_dir: Path,
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data"
    raw_dir = data_root / "raw"
    _symlink_raw_data(raw_data_dir, raw_dir)
    (data_root / "interim").mkdir(parents=True, exist_ok=True)
    (data_root / "processed").mkdir(parents=True, exist_ok=True)
    results_dir = tmp_path / "results"

    result = run_full_pipeline(data_root=data_root, results_dir=results_dir)

    assert result.analysis_summary.summary_path.is_file()
    assert (data_root / "interim" / "baseline" / "baseline_average.npz").is_file()
    assert (data_root / "interim" / "etalon" / "etalon_fit.npz").is_file()
    assert (data_root / "processed" / "exports" / "shock_frequency_domain.npz").is_file()
    assert (data_root / "processed" / "exports" / "voigt_fit_results.npz").is_file()
    assert (results_dir / "tables" / "state_history.npz").is_file()
    assert (results_dir / "tables" / "state_history.csv").is_file()
    assert (results_dir / "tables" / "voigt_fit_summary.csv").is_file()
