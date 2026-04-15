# I/O And Sweep Alignment

`src/mock_lab/io/` owns the conversion from raw MATLAB traces into the aligned sweep arrays used everywhere else in the repository.

## What The Module Assumes

- Each MAT file contains exactly one detector channel and one reference channel.
- The reference channel name contains `ref`; every other non-private variable is treated as the detector channel.
- The detector and reference vectors have the same length and represent uniformly sampled data.
- The reference waveform is a TTL-like sweep marker whose repetition frequency is `300 kHz` by default.
- The working sweep window starts at `DEFAULT_PHASE_START_S = 2.2e-6`, which intentionally skips the leading partial sweep and keeps a consistent baseline-ramp-baseline segment from every period.

If any of those assumptions stop being true, this package is the first place that needs to change.

## Main Entry Points

- `load_mat_trace(path)`: loads one MAT file, identifies the signal and reference arrays, and returns a `MatlabTrace`.
- `load_mat_file(path)`: compatibility helper that returns only the detector and reference vectors.
- `binarize_reference(reference)`: converts the analog reference waveform into `0/1` samples using a midpoint threshold.
- `get_time(signal, reference)`: reconstructs a time axis by locating rising edges in the binary reference, estimating the samples per sweep from the median edge spacing, and assigning the first rising edge as time zero.
- `split_samples(signal, time)`: reshapes one continuous 1D signal into a 2D `sweep_count x samples_per_sweep` array after applying the fixed phase offset.
- `split_trace(trace)`: applies the same sweep cutting to both detector and reference channels and returns a `SweepCollection` with a shared local time axis.

## Timing Reconstruction

The code does not rely on a stored oscilloscope time vector. Instead it reconstructs time from the reference waveform:

1. Threshold the reference into a binary waveform.
2. Find the rising-edge sample indices.
3. Estimate `samples_per_sweep` from the median spacing between those edges.
4. Compute the sample period as `1 / (reference_frequency * samples_per_sweep)`.
5. Build a signed time axis relative to the first detected rising edge.

This works well for the supplied lab traces because the function-generator period is stable and the reference channel is clean. The code does not try to repair missing edges or variable-frequency ramps.

## Sweep Cutting

`split_samples()` applies the fixed phase offset before reshaping:

- it finds the first sample at or after `t0`
- discards the incomplete leading segment before that point
- keeps only an integer number of full sweeps
- drops any incomplete tail at the end of the trace

The returned 2D arrays are phase-aligned by construction. `split_trace()` then shifts each sweep's local time axis so every row starts at the same `t0`. That is why the later baseline, etalon, and shock stages can average sweeps sample-by-sample without any extra registration step.

## Why The Fixed Phase Offset Matters

The rest of the reduction assumes that each extracted sweep contains:

- a short baseline region before the scan reaches the useful ramp
- the useful ramp itself
- a short trailing baseline region

That structure is used repeatedly:

- baseline stages fit wing lines to the start and end of the sweep
- etalon stages remove edge offsets before averaging peaks
- shock stages window the absorbance only after the corrected baseline reaches a useful signal level

Changing `DEFAULT_PHASE_START_S` changes more than just the plotted x-axis. It changes the samples available for baseline fitting, peak detection, and absorbance windowing throughout the pipeline.
