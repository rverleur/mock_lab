from pathlib import Path

import numpy as np

from mock_lab.io.matlab_loader import (
    DEFAULT_PHASE_START_S,
    load_mat_file,
    load_mat_trace,
    split_trace,
)
from mock_lab.spectroscopy.absorbance import (
    beer_lambert_absorbance,
    find_analysis_window,
    fit_edge_line,
    subtract_edge_line,
)


def test_load_mat_file_reads_baseline_dataset(raw_data_dir: Path) -> None:
    signal, reference = load_mat_file(raw_data_dir / "MockLabData_Baseline.mat")

    assert isinstance(signal, np.ndarray)
    assert isinstance(reference, np.ndarray)
    assert signal.shape == (625000,)
    assert reference.shape == (625000,)
    assert np.isfinite(signal).all()
    assert np.isfinite(reference).all()


def test_split_trace_extracts_phase_aligned_etalon_sweeps(raw_data_dir: Path) -> None:
    trace = load_mat_trace(raw_data_dir / "MockLabData_Etalon.mat")
    sweeps = split_trace(trace)

    assert sweeps.signal.shape == (498, 2500)
    assert sweeps.reference.shape == (498, 2500)
    assert sweeps.time_s.shape == (498, 2500)
    assert np.allclose(sweeps.time_s[0], sweeps.time_s[-1])
    assert np.isclose(sweeps.time_axis_s[0], DEFAULT_PHASE_START_S)
    assert sweeps.time_axis_s[-1] > sweeps.time_axis_s[0]


def test_subtract_edge_line_removes_linear_background() -> None:
    sample_index = np.arange(128, dtype=float)
    feature = np.exp(-0.5 * ((sample_index - 64.0) / 8.0) ** 2)
    signal = 0.01 * sample_index + 3.0 + feature

    fitted_line = fit_edge_line(signal, edge_samples=16)
    corrected = subtract_edge_line(signal, edge_samples=16)

    assert np.allclose(fitted_line[:16], signal[:16], atol=1.0e-2)
    assert np.allclose(fitted_line[-16:], signal[-16:], atol=1.0e-2)
    assert corrected[64] > 0.9


def test_absorbance_uses_positive_ratio_and_window_stops_at_peak() -> None:
    baseline = np.linspace(0.2, 1.6, 80)
    shock = baseline.copy()
    shock[32:50] *= 0.8
    shock[50:65] *= np.linspace(0.8, 1.0, 15)

    window = find_analysis_window(
        baseline,
        shock,
        start_index=32,
        minimum_reference_signal=1.0,
    )
    absorbance = beer_lambert_absorbance(baseline[window], shock[window])

    assert window.start > 32
    assert window.stop == 80
    assert np.nanmax(absorbance) > 0.0
    assert np.isclose(absorbance[-1], 0.0)
