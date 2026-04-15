"""End-to-end orchestration for the active mock-lab workflow."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from mock_lab.pipelines.baseline import BaselinePipelineResult, run_baseline_pipeline
from mock_lab.pipelines.etalon import EtalonPipelineResult, run_etalon_pipeline
from mock_lab.pipelines.shock_snapshot import (
    ShockFrequencyDomainResult,
    run_shock_snapshot_pipeline,
)
from mock_lab.pipelines.time_history import (
    TimeHistoryPipelineResult,
    run_time_history_pipeline,
)
from mock_lab.pipelines.voigt_fit import VoigtFitPipelineResult, run_voigt_fit_pipeline
from mock_lab.reporting.summary import AnalysisSummary, build_report_summary
from mock_lab.spectroscopy.state_estimation import DEFAULT_OPTICAL_PATH_LENGTH_CM


@dataclass(frozen=True)
class FullPipelineResult:
    """Outputs from the baseline-to-state-history workflow."""

    baseline: BaselinePipelineResult
    etalon: EtalonPipelineResult
    shock_snapshot: ShockFrequencyDomainResult
    voigt_fit: VoigtFitPipelineResult
    time_history: TimeHistoryPipelineResult
    analysis_summary: AnalysisSummary


def run_full_pipeline(
    data_root: Path | str,
    results_dir: Path | str,
    *,
    minimum_peak_absorbance: float = 0.02,
    optical_path_length_cm: float = DEFAULT_OPTICAL_PATH_LENGTH_CM,
) -> FullPipelineResult:
    """Run the active reduction chain from raw MAT files to summary outputs."""

    data_root = Path(data_root)
    results_dir = Path(results_dir)

    raw_dir = data_root / "raw"
    interim_dir = data_root / "interim"
    processed_dir = data_root / "processed"

    baseline_raw_data = raw_dir / "MockLabData_Baseline.mat"
    etalon_raw_data = raw_dir / "MockLabData_Etalon.mat"
    shock_raw_data = raw_dir / "MockLabData_Shock.mat"

    baseline_output_dir = interim_dir / "baseline"
    etalon_output_dir = interim_dir / "etalon"
    shock_output_dir = processed_dir / "exports"
    figure_dir = results_dir / "figures"
    table_dir = results_dir / "tables"
    report_dir = results_dir / "reports"

    baseline_result = run_baseline_pipeline(
        raw_data=baseline_raw_data,
        output_dir=baseline_output_dir,
        figure_output_dir=figure_dir,
    )
    etalon_result = run_etalon_pipeline(
        raw_data=etalon_raw_data,
        output_dir=etalon_output_dir,
        figure_output_dir=figure_dir,
    )
    shock_result = run_shock_snapshot_pipeline(
        raw_data=shock_raw_data,
        baseline_dir=baseline_output_dir,
        etalon_dir=etalon_output_dir,
        output_dir=shock_output_dir,
        figure_output_dir=figure_dir,
    )
    voigt_fit_result = run_voigt_fit_pipeline(
        shock_frequency_data=shock_output_dir / "shock_frequency_domain.npz",
        output_dir=shock_output_dir,
        figure_output_dir=figure_dir,
        table_output_dir=table_dir,
        minimum_peak_absorbance=minimum_peak_absorbance,
    )
    time_history_result = run_time_history_pipeline(
        voigt_fit_data=shock_output_dir / "voigt_fit_results.npz",
        output_dir=table_dir,
        figure_output_dir=figure_dir,
        optical_path_length_cm=optical_path_length_cm,
    )
    analysis_summary = build_report_summary(
        baseline_data=baseline_output_dir / "baseline_average.npz",
        etalon_data=etalon_output_dir / "etalon_fit.npz",
        shock_data=shock_output_dir / "shock_frequency_domain.npz",
        voigt_fit_data=shock_output_dir / "voigt_fit_results.npz",
        state_history_data=table_dir / "state_history.npz",
        output_path=report_dir / "analysis_summary.md",
    )

    return FullPipelineResult(
        baseline=baseline_result,
        etalon=etalon_result,
        shock_snapshot=shock_result,
        voigt_fit=voigt_fit_result,
        time_history=time_history_result,
        analysis_summary=analysis_summary,
    )
