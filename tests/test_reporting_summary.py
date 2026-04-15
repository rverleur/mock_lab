import numpy as np

from mock_lab.reporting.summary import build_report_summary


def test_build_report_summary_writes_markdown_map(tmp_path) -> None:
    baseline_data = tmp_path / "baseline_average.npz"
    etalon_data = tmp_path / "etalon_fit.npz"
    shock_data = tmp_path / "shock_frequency_domain.npz"
    voigt_fit_data = tmp_path / "voigt_fit_results.npz"
    state_history_data = tmp_path / "state_history.npz"
    output_path = tmp_path / "reports" / "analysis_summary.md"

    np.savez(
        baseline_data,
        sweep_count=12,
    )
    np.savez(
        etalon_data,
        peak_indices=np.array([10, 20, 30], dtype=np.int64),
    )
    np.savez(
        shock_data,
        absorbance_sweeps=np.zeros((5, 16), dtype=float),
    )
    np.savez(
        voigt_fit_data,
        success=np.array([True, False, True, True, False], dtype=bool),
    )
    np.savez(
        state_history_data,
        temperature_k=np.array([1800.0, 2100.0, np.nan], dtype=float),
        pressure_atm=np.array([1.1, 1.8, np.nan], dtype=float),
        co_mole_fraction=np.array([0.12, 0.18, np.nan], dtype=float),
    )

    summary = build_report_summary(
        baseline_data=baseline_data,
        etalon_data=etalon_data,
        shock_data=shock_data,
        voigt_fit_data=voigt_fit_data,
        state_history_data=state_history_data,
        output_path=output_path,
    )

    contents = output_path.read_text(encoding="utf-8")

    assert summary.baseline_sweep_count == 12
    assert summary.etalon_peak_count == 3
    assert summary.successful_voigt_fits == 3
    assert "Analysis Summary" in contents
    assert "Successful Voigt fits: 3 / 5" in contents
    assert "Temperature range [K]: 1.8e+03 to 2.1e+03" in contents
