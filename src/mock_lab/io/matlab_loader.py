"""Utilities for loading MATLAB traces and splitting phase-aligned sweeps."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
import scipy.io as sio


DEFAULT_REFERENCE_FREQUENCY_HZ = 300_000.0
DEFAULT_PHASE_START_S = 2.2e-6

Array1D = NDArray[np.float64]
Array2D = NDArray[np.float64]


@dataclass(frozen=True)
class MatlabTrace:
    """Raw detector and reference channels loaded from one MAT file."""

    source_path: Path
    signal_name: str
    reference_name: str
    signal: Array1D
    reference: Array1D


@dataclass(frozen=True)
class SweepCollection:
    """Phase-aligned signal and reference windows spanning one full sweep."""

    signal: Array2D
    reference: Array2D
    time_s: Array2D
    sample_period_s: float
    samples_per_sweep: int
    phase_start_s: float

    @property
    def time_axis_s(self) -> Array1D:
        """Return the common time axis shared by all extracted sweeps."""

        return self.time_s[0]


def _as_float_vector(values: object) -> Array1D:
    """Convert a MATLAB array into a 1D float vector."""

    array = np.asarray(values, dtype=float).squeeze()

    if array.ndim != 1:
        raise ValueError("Expected a 1D signal after squeezing the MAT array.")

    return array


def load_mat_trace(path: Path | str) -> MatlabTrace:
    """Load one MAT file and identify the detector and reference channels."""

    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(f"MAT file not found: {path}")

    mat_contents = sio.loadmat(str(path))
    variable_names = [key for key in mat_contents if not key.startswith("__")]
    reference_names = [key for key in variable_names if "ref" in key.lower()]
    signal_names = [key for key in variable_names if key not in reference_names]

    if len(signal_names) != 1 or len(reference_names) != 1:
        raise ValueError(
            "Expected exactly one detector channel and one reference channel in the MAT file."
        )

    signal_name = signal_names[0]
    reference_name = reference_names[0]
    signal = _as_float_vector(mat_contents[signal_name])
    reference = _as_float_vector(mat_contents[reference_name])

    if signal.shape != reference.shape:
        raise ValueError("Signal and reference arrays must have the same shape.")

    return MatlabTrace(
        source_path=path,
        signal_name=signal_name,
        reference_name=reference_name,
        signal=signal,
        reference=reference,
    )


def load_mat_file(path: Path | str) -> tuple[Array1D, Array1D]:
    """Load one MAT file and return the detector and reference vectors."""

    trace = load_mat_trace(path)
    return trace.signal, trace.reference


def binarize_reference(reference: Array1D, threshold: float | None = None) -> NDArray[np.int8]:
    """Convert the analog TTL reference signal into a binary waveform."""

    if threshold is None:
        threshold = 0.5 * (float(np.min(reference)) + float(np.max(reference)))

    return np.where(reference > threshold, 1, 0).astype(np.int8)


def get_time(
    signal: Array1D,
    reference: Array1D,
    reference_freq: float = DEFAULT_REFERENCE_FREQUENCY_HZ,
) -> Array1D:
    """Estimate a time axis using the TTL reference period.

    The time origin is set at the first detected rising edge in the reference
    signal so negative times correspond to samples collected before that edge.
    """

    if signal.shape != reference.shape:
        raise ValueError("Signal and reference arrays must have the same shape.")

    binary_ref = binarize_reference(reference)
    rising_edges = np.where(np.diff(binary_ref) > 0)[0] + 1

    if rising_edges.size < 2:
        raise ValueError("At least two rising edges are required to estimate timing.")

    samples_per_sweep = int(round(float(np.median(np.diff(rising_edges)))))
    sample_period_s = 1.0 / (reference_freq * samples_per_sweep)
    sample_indices = np.arange(signal.size) - int(rising_edges[0])

    return sample_indices * sample_period_s


def split_samples(
    signal: Array1D,
    time: Array1D,
    reference_freq: float = DEFAULT_REFERENCE_FREQUENCY_HZ,
    t0: float = DEFAULT_PHASE_START_S,
) -> tuple[Array2D, Array2D]:
    """Split a continuous trace into phase-aligned sweep windows.

    Each extracted row begins at `t0` within the ramp and spans one full
    function-generator period. This preserves the phase-aligned behavior you
    were already using, but makes it explicit and repeatable.
    """

    if signal.shape != time.shape:
        raise ValueError("Signal and time arrays must have the same shape.")

    if signal.ndim != 1:
        raise ValueError("Signal and time must be 1D arrays.")

    if signal.size < 2:
        raise ValueError("At least two samples are required to split sweeps.")

    sample_period_s = float(np.median(np.diff(time)))
    samples_per_sweep = int(round(1.0 / (reference_freq * sample_period_s)))
    start_index = int(np.searchsorted(time, t0, side="left"))
    remaining_samples = signal.size - start_index
    sweep_count = remaining_samples // samples_per_sweep

    if sweep_count <= 0:
        raise ValueError("No complete sweeps remain after the requested phase offset.")

    stop_index = start_index + sweep_count * samples_per_sweep
    signal_sweeps = signal[start_index:stop_index].reshape(sweep_count, samples_per_sweep)
    time_sweeps = time[start_index:stop_index].reshape(sweep_count, samples_per_sweep)

    return signal_sweeps, time_sweeps


def split_trace(
    trace: MatlabTrace,
    reference_freq: float = DEFAULT_REFERENCE_FREQUENCY_HZ,
    t0: float = DEFAULT_PHASE_START_S,
) -> SweepCollection:
    """Split both channels into matching sweep windows with a local phase axis."""

    time_s = get_time(trace.signal, trace.reference, reference_freq=reference_freq)
    signal_sweeps, time_sweeps = split_samples(
        trace.signal,
        time_s,
        reference_freq=reference_freq,
        t0=t0,
    )
    reference_sweeps, _ = split_samples(
        trace.reference,
        time_s,
        reference_freq=reference_freq,
        t0=t0,
    )
    local_time_sweeps = time_sweeps - time_sweeps[:, :1] + t0
    sample_period_s = float(np.median(np.diff(time_sweeps[0])))

    return SweepCollection(
        signal=signal_sweeps,
        reference=reference_sweeps,
        time_s=local_time_sweeps,
        sample_period_s=sample_period_s,
        samples_per_sweep=signal_sweeps.shape[1],
        phase_start_s=t0,
    )
