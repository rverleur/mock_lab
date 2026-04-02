"""Utilities for etalon peak picking and relative-wavenumber calibration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.signal import find_peaks


ETALON_LENGTH_CM = 3.0 * 2.54
ETALON_REFRACTIVE_INDEX = 4.019
ETALON_FSR_CM_INV = 1.0 / (2.0 * ETALON_REFRACTIVE_INDEX * ETALON_LENGTH_CM)

Array1D = NDArray[np.float64]
Array2D = NDArray[np.float64]
Index1D = NDArray[np.int64]


@dataclass(frozen=True)
class EtalonFit:
    """Second-order time-to-relative-wavenumber calibration."""

    coefficients: Array1D
    time_axis_s: Array1D
    relative_wavenumber_cm_inv: Array1D
    peak_indices: Index1D
    peak_time_s: Array1D
    peak_wavenumber_cm_inv: Array1D
    rms_residual_cm_inv: float


def remove_edge_baseline(sweeps: Array2D, edge_samples: int = 30) -> Array2D:
    """Subtract the mean of the first and last edge samples from each sweep."""

    edge_region = np.hstack([sweeps[:, :edge_samples], sweeps[:, -edge_samples:]])
    baseline = np.mean(edge_region, axis=1, keepdims=True)
    return sweeps - baseline


def build_representative_sweep(sweeps: Array2D) -> Array1D:
    """Average many aligned sweeps into one low-noise representative trace."""

    return np.mean(sweeps, axis=0)


def find_etalon_peaks(
    signal: Array1D,
    prominence: float = 0.2,
    distance: int = 20,
    width: int = 15,
    drop_last_peak: bool = True,
) -> Index1D:
    """Detect etalon fringes on a single representative sweep."""

    peaks, _ = find_peaks(
        signal,
        prominence=prominence,
        distance=distance,
        width=width,
        height=0.0,
    )

    if drop_last_peak and peaks.size > 0:
        peaks = peaks[:-1]

    return peaks.astype(np.int64)


def evaluate_relative_wavenumber(time_axis_s: Array1D, coefficients: Array1D) -> Array1D:
    """Evaluate the polynomial calibration on a time axis."""

    return np.polyval(coefficients, time_axis_s)


def fit_relative_wavenumber(
    time_axis_s: Array1D,
    peak_indices: Index1D,
    polynomial_order: int = 2,
    fsr_cm_inv: float = ETALON_FSR_CM_INV,
) -> EtalonFit:
    """Fit a polynomial mapping sweep time to relative wavenumber."""

    peak_time_s = time_axis_s[peak_indices]
    peak_wavenumber_cm_inv = -fsr_cm_inv * np.arange(peak_indices.size, dtype=float)
    coefficients = np.polyfit(peak_time_s, peak_wavenumber_cm_inv, polynomial_order)
    relative_wavenumber_cm_inv = evaluate_relative_wavenumber(time_axis_s, coefficients)
    residuals = peak_wavenumber_cm_inv - evaluate_relative_wavenumber(peak_time_s, coefficients)
    rms_residual_cm_inv = float(np.sqrt(np.mean(residuals**2)))

    return EtalonFit(
        coefficients=np.asarray(coefficients, dtype=float),
        time_axis_s=np.asarray(time_axis_s, dtype=float),
        relative_wavenumber_cm_inv=np.asarray(relative_wavenumber_cm_inv, dtype=float),
        peak_indices=np.asarray(peak_indices, dtype=np.int64),
        peak_time_s=np.asarray(peak_time_s, dtype=float),
        peak_wavenumber_cm_inv=np.asarray(peak_wavenumber_cm_inv, dtype=float),
        rms_residual_cm_inv=rms_residual_cm_inv,
    )


def save_etalon_fit(
    path: Path | str,
    fit: EtalonFit,
    representative_sweep: Array1D,
    plot_sweep: Array1D,
) -> None:
    """Persist the calibration and a couple of useful reference traces."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        coefficients=fit.coefficients,
        time_axis_s=fit.time_axis_s,
        relative_wavenumber_cm_inv=fit.relative_wavenumber_cm_inv,
        peak_indices=fit.peak_indices,
        peak_time_s=fit.peak_time_s,
        peak_wavenumber_cm_inv=fit.peak_wavenumber_cm_inv,
        rms_residual_cm_inv=fit.rms_residual_cm_inv,
        representative_sweep=representative_sweep,
        plot_sweep=plot_sweep,
    )


def load_etalon_fit(path: Path | str) -> EtalonFit:
    """Load a saved etalon calibration from disk."""

    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(f"Etalon calibration file not found: {path}")

    with np.load(path) as data:
        return EtalonFit(
            coefficients=np.asarray(data["coefficients"], dtype=float),
            time_axis_s=np.asarray(data["time_axis_s"], dtype=float),
            relative_wavenumber_cm_inv=np.asarray(
                data["relative_wavenumber_cm_inv"], dtype=float
            ),
            peak_indices=np.asarray(data["peak_indices"], dtype=np.int64),
            peak_time_s=np.asarray(data["peak_time_s"], dtype=float),
            peak_wavenumber_cm_inv=np.asarray(data["peak_wavenumber_cm_inv"], dtype=float),
            rms_residual_cm_inv=float(data["rms_residual_cm_inv"]),
        )
