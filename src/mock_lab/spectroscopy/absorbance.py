"""Utilities for sweep background removal and absorbance calculation."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


Array1D = NDArray[np.float64]
Array2D = NDArray[np.float64]


def average_sweeps(sweeps: Array2D) -> Array1D:
    """Average a stack of phase-aligned sweeps into one representative trace."""

    return np.mean(sweeps, axis=0)


def _edge_indices(signal_length: int, edge_samples: int) -> NDArray[np.int64]:
    """Return the sample indices used to fit the leading/trailing wing line."""

    if signal_length < 2 * edge_samples:
        raise ValueError("Signal is too short for the requested edge-sample count.")

    return np.concatenate(
        [
            np.arange(edge_samples, dtype=np.int64),
            np.arange(signal_length - edge_samples, signal_length, dtype=np.int64),
        ]
    )


def fit_edge_line(signal: Array1D, edge_samples: int = 32) -> Array1D:
    """Fit a straight line through the first and last `edge_samples` points."""

    if signal.ndim != 1:
        raise ValueError("Line fitting expects a 1D signal.")

    signal_length = signal.size
    fit_indices = _edge_indices(signal_length, edge_samples)
    x_fit = fit_indices.astype(float)
    coefficients = np.polyfit(x_fit, signal[fit_indices], 1)
    x_full = np.arange(signal_length, dtype=float)
    return np.polyval(coefficients, x_full)


def fit_edge_lines(sweeps: Array2D, edge_samples: int = 32) -> Array2D:
    """Fit one straight wing line to every sweep in a stack."""

    if sweeps.ndim != 2:
        raise ValueError("Batch line fitting expects a 2D array of sweeps.")

    sweep_length = sweeps.shape[1]
    fit_indices = _edge_indices(sweep_length, edge_samples)
    x_fit = fit_indices.astype(float)
    design_matrix = np.column_stack([x_fit, np.ones_like(x_fit)])
    pseudo_inverse = np.linalg.pinv(design_matrix)
    coefficients = sweeps[:, fit_indices] @ pseudo_inverse.T
    x_full = np.arange(sweep_length, dtype=float)
    return coefficients[:, [0]] * x_full[np.newaxis, :] + coefficients[:, [1]]


def subtract_edge_line(signal: Array1D, edge_samples: int = 32) -> Array1D:
    """Subtract the fitted wing line from a single sweep."""

    return signal - fit_edge_line(signal, edge_samples=edge_samples)


def subtract_edge_lines(sweeps: Array2D, edge_samples: int = 32) -> Array2D:
    """Subtract the fitted wing line from every sweep in a stack."""

    return sweeps - fit_edge_lines(sweeps, edge_samples=edge_samples)


def scale_sweeps_to_reference_peak(
    reference_signal: Array1D,
    sweeps: Array2D,
) -> tuple[Array2D, Array1D]:
    """Scale each sweep so its peak matches the peak of the reference signal."""

    reference_peak = float(np.max(reference_signal))

    if reference_peak <= 0.0:
        raise ValueError("Reference signal must have a positive peak for scaling.")

    sweep_peaks = np.max(sweeps, axis=1)
    scale_factors = np.ones_like(sweep_peaks)
    positive_mask = sweep_peaks > 0.0
    scale_factors[positive_mask] = reference_peak / sweep_peaks[positive_mask]
    return sweeps * scale_factors[:, np.newaxis], scale_factors


def find_analysis_window(
    reference_signal: Array1D,
    start_index: int = 32,
    minimum_reference_signal: float = 1.0,
    maximum_reference_signal: float = 2.75,
) -> slice:
    """Select the absorbance-analysis window on a corrected sweep.

    The window begins at the first sample, at or after `start_index`, where the
    corrected baseline/reference signal reaches `minimum_reference_signal`. It
    ends immediately before the corrected baseline rises above
    `maximum_reference_signal`.
    """

    if reference_signal.ndim != 1:
        raise ValueError("Window detection expects a 1D reference signal.")

    if not 0 <= start_index < reference_signal.size:
        raise ValueError("The requested start index must lie within the signal.")

    threshold_indices = np.flatnonzero(reference_signal[start_index:] >= minimum_reference_signal)

    if threshold_indices.size == 0:
        raise ValueError("Reference signal never reaches the requested minimum level.")

    window_start = start_index + int(threshold_indices[0])
    upper_indices = np.flatnonzero(reference_signal[window_start:] > maximum_reference_signal)

    if upper_indices.size == 0:
        window_stop = reference_signal.size
    else:
        window_stop = window_start + int(upper_indices[0])

    return slice(window_start, window_stop)


def beer_lambert_absorbance(reference_signal: Array1D, transmitted_signal: Array1D) -> Array1D:
    """Compute absorbance from linearly responding detector voltages.

    Any non-positive samples are marked as `nan` because Beer-Lambert
    absorbance is only defined for positive transmission ratios.
    """

    if reference_signal.shape != transmitted_signal.shape:
        raise ValueError("Reference and transmitted signals must have the same shape.")

    absorbance = np.full(reference_signal.shape, np.nan, dtype=float)
    valid = (reference_signal > 0.0) & (transmitted_signal > 0.0)
    absorbance[valid] = -np.log(transmitted_signal[valid] / reference_signal[valid])
    return absorbance
