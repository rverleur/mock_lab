"""Video-rendering pipeline for sweep-by-sweep Voigt-fit diagnostics."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from mock_lab.plotting.figures import plot_voigt_fit, save_figure
from mock_lab.plotting.mpl import plt
from mock_lab.spectroscopy.voigt import (
    DEFAULT_CO_TRANSITIONS,
    VoigtFitParameters,
    evaluate_voigt_spectrum,
)


Array1D = NDArray[np.float64]
Array2D = NDArray[np.float64]
Bool1D = NDArray[np.bool_]


@dataclass(frozen=True)
class VoigtFitVideoResult:
    """Outputs from the Voigt-fit video rendering pass."""

    video_path: Path
    frame_count: int
    fps: int
    first_sweep_index: int
    last_sweep_index: int


def _load_fit_products(
    voigt_fit_data: Path,
) -> tuple[Array1D, Array2D, Array2D, Bool1D, Array1D, Array2D, Array2D, Array2D, Array1D, Array1D]:
    """Load the saved Voigt-fit arrays needed to reconstruct video frames."""

    with np.load(voigt_fit_data) as data:
        return (
            np.asarray(data["frequency_cm_inv"], dtype=float),
            np.asarray(data["absorbance_sweeps"], dtype=float),
            np.asarray(data["fitted_absorbance_sweeps"], dtype=float),
            np.asarray(data["success"], dtype=bool),
            np.asarray(data["temperature_k"], dtype=float),
            np.asarray(data["line_centers_relative_cm_inv"], dtype=float),
            np.asarray(data["collisional_hwhm_cm_inv"], dtype=float),
            np.asarray(data["line_areas"], dtype=float),
            np.asarray(data["baseline_offset"], dtype=float),
            np.asarray(data["baseline_slope"], dtype=float),
        )


def _global_plot_limits(
    absorbance_sweeps: Array2D,
    fitted_absorbance_sweeps: Array2D,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Choose fixed y-limits so all video frames use the same scale."""

    absorbance_values = np.concatenate(
        [
            absorbance_sweeps[np.isfinite(absorbance_sweeps)],
            fitted_absorbance_sweeps[np.isfinite(fitted_absorbance_sweeps)],
        ]
    )

    if absorbance_values.size == 0:
        absorbance_limits = (-0.05, 0.05)
    else:
        absorbance_span = float(np.max(absorbance_values) - np.min(absorbance_values))
        absorbance_margin = max(0.02, 0.08 * absorbance_span)
        absorbance_limits = (
            float(np.min(absorbance_values) - absorbance_margin),
            float(np.max(absorbance_values) + absorbance_margin),
        )

    residual_values = absorbance_sweeps - fitted_absorbance_sweeps
    finite_residuals = residual_values[np.isfinite(residual_values)]

    if finite_residuals.size == 0:
        residual_limits = (-0.05, 0.05)
    else:
        residual_extent = float(np.nanpercentile(np.abs(finite_residuals), 99.0))
        residual_extent = max(0.02, 1.15 * residual_extent)
        residual_limits = (-residual_extent, residual_extent)

    return absorbance_limits, residual_limits


def _annotation_text(
    sweep_index: int,
    success: bool,
    temperature_k: float,
    rmse_absorbance: float,
) -> str:
    """Build the per-frame fit summary shown in the upper panel."""

    if success and np.isfinite(temperature_k):
        return "\n".join(
            [
                f"Sweep {sweep_index:03d}",
                "Fit converged",
                f"T = {temperature_k:.0f} K",
                f"RMSE = {rmse_absorbance:.4f}",
            ]
        )

    return "\n".join([f"Sweep {sweep_index:03d}", "Fit unavailable"])


def render_voigt_fit_video(
    voigt_fit_data: Path | str,
    video_path: Path | str,
    *,
    fps: int = 12,
) -> VoigtFitVideoResult:
    """Render one Voigt-fit diagnostic frame per sweep and encode an MP4."""

    voigt_fit_data = Path(voigt_fit_data)
    video_path = Path(video_path)
    video_path.parent.mkdir(parents=True, exist_ok=True)

    ffmpeg_path = shutil.which("ffmpeg")

    if ffmpeg_path is None:
        raise FileNotFoundError("ffmpeg is required to build the Voigt-fit video.")

    (
        frequency_cm_inv,
        absorbance_sweeps,
        fitted_absorbance_sweeps,
        success,
        temperature_k,
        line_centers_relative_cm_inv,
        collisional_hwhm_cm_inv,
        line_areas,
        baseline_offset,
        baseline_slope,
    ) = _load_fit_products(voigt_fit_data)
    rmse_absorbance = np.sqrt(
        np.nanmean((absorbance_sweeps - fitted_absorbance_sweeps) ** 2, axis=1)
    )
    absorbance_limits, residual_limits = _global_plot_limits(
        absorbance_sweeps,
        fitted_absorbance_sweeps,
    )
    frame_count = absorbance_sweeps.shape[0]

    with tempfile.TemporaryDirectory(prefix="mock_lab_voigt_frames_") as temp_dir_name:
        temp_dir = Path(temp_dir_name)

        for sweep_index in range(frame_count):
            absorbance = absorbance_sweeps[sweep_index]
            fit_available = bool(success[sweep_index]) and np.any(np.isfinite(fitted_absorbance_sweeps[sweep_index]))
            mask = np.isfinite(absorbance)
            frequency_frame = frequency_cm_inv[mask]
            absorbance_frame = absorbance[mask]

            if fit_available:
                parameters = VoigtFitParameters(
                    temperature_k=float(temperature_k[sweep_index]),
                    line_centers_relative_cm_inv=np.asarray(
                        line_centers_relative_cm_inv[sweep_index],
                        dtype=float,
                    ),
                    collisional_hwhm_cm_inv=np.asarray(
                        collisional_hwhm_cm_inv[sweep_index],
                        dtype=float,
                    ),
                    line_areas=np.asarray(line_areas[sweep_index], dtype=float),
                    baseline_offset=float(baseline_offset[sweep_index]),
                    baseline_slope=float(baseline_slope[sweep_index]),
                )
                fitted_frame, component_frame, _, _ = evaluate_voigt_spectrum(
                    frequency_frame,
                    parameters,
                )
            else:
                fitted_frame = np.full_like(absorbance_frame, np.nan)
                component_frame = np.empty((0, absorbance_frame.size), dtype=float)

            figure = plot_voigt_fit(
                frequency_frame,
                absorbance_frame,
                fitted_frame,
                component_frame,
                component_labels=tuple(
                    transition.label for transition in DEFAULT_CO_TRANSITIONS[: component_frame.shape[0]]
                ),
                component_centers_cm_inv=np.asarray(
                    line_centers_relative_cm_inv[sweep_index],
                    dtype=float,
                )
                if fit_available
                else None,
                absorbance_limits=absorbance_limits,
                residual_limits=residual_limits,
                annotation_text=_annotation_text(
                    sweep_index,
                    fit_available,
                    float(temperature_k[sweep_index]),
                    float(rmse_absorbance[sweep_index]),
                ),
            )
            save_figure(figure, temp_dir / f"frame_{sweep_index:04d}.png")
            plt.close(figure)

        subprocess.run(
            [
                ffmpeg_path,
                "-y",
                "-framerate",
                str(fps),
                "-i",
                str(temp_dir / "frame_%04d.png"),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(video_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )

    return VoigtFitVideoResult(
        video_path=video_path,
        frame_count=frame_count,
        fps=fps,
        first_sweep_index=0,
        last_sweep_index=frame_count - 1,
    )
