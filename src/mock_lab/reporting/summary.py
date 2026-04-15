"""Helpers for generating concise report-facing workflow summaries."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


Array1D = NDArray[np.float64]


@dataclass(frozen=True)
class AnalysisSummary:
    """Compact summary statistics for one full pipeline run."""

    summary_path: Path
    baseline_sweep_count: int
    etalon_peak_count: int
    shock_sweep_count: int
    successful_voigt_fits: int
    total_voigt_fits: int
    temperature_range_k: tuple[float, float]
    pressure_range_atm: tuple[float, float]
    co_mole_fraction_range: tuple[float, float]


def _finite_range(values: Array1D) -> tuple[float, float]:
    """Return the finite min/max pair for one saved state array."""

    finite_values = np.asarray(values, dtype=float)
    finite_values = finite_values[np.isfinite(finite_values)]

    if finite_values.size == 0:
        return float("nan"), float("nan")

    return float(np.min(finite_values)), float(np.max(finite_values))


def _relative_path(path: Path, root: Path) -> str:
    """Return a stable repo-relative path for summary text."""

    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def build_report_summary(
    *,
    baseline_data: Path | str,
    etalon_data: Path | str,
    shock_data: Path | str,
    voigt_fit_data: Path | str,
    state_history_data: Path | str,
    output_path: Path | str,
) -> AnalysisSummary:
    """Write a short markdown map of the generated analysis products."""

    baseline_data = Path(baseline_data)
    etalon_data = Path(etalon_data)
    shock_data = Path(shock_data)
    voigt_fit_data = Path(voigt_fit_data)
    state_history_data = Path(state_history_data)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    repo_root = output_path.resolve().parents[2]

    with np.load(baseline_data) as data:
        baseline_sweep_count = int(np.asarray(data["sweep_count"]).item())

    with np.load(etalon_data) as data:
        etalon_peak_count = int(np.asarray(data["peak_indices"], dtype=np.int64).size)

    with np.load(shock_data) as data:
        shock_sweep_count = int(np.asarray(data["absorbance_sweeps"], dtype=float).shape[0])

    with np.load(voigt_fit_data) as data:
        success = np.asarray(data["success"], dtype=bool)
        successful_voigt_fits = int(np.count_nonzero(success))
        total_voigt_fits = int(success.size)

    with np.load(state_history_data) as data:
        temperature_range_k = _finite_range(np.asarray(data["temperature_k"], dtype=float))
        pressure_range_atm = _finite_range(np.asarray(data["pressure_atm"], dtype=float))
        co_mole_fraction_range = _finite_range(np.asarray(data["co_mole_fraction"], dtype=float))

    summary = AnalysisSummary(
        summary_path=output_path,
        baseline_sweep_count=baseline_sweep_count,
        etalon_peak_count=etalon_peak_count,
        shock_sweep_count=shock_sweep_count,
        successful_voigt_fits=successful_voigt_fits,
        total_voigt_fits=total_voigt_fits,
        temperature_range_k=temperature_range_k,
        pressure_range_atm=pressure_range_atm,
        co_mole_fraction_range=co_mole_fraction_range,
    )

    output_path.write_text(
        "\n".join(
            [
                "# Analysis Summary",
                "",
                "## Workflow",
                "1. Average the baseline sweeps and remove their edge-fit background.",
                "2. Fit the etalon peaks to obtain a relative wavenumber calibration.",
                "3. Convert the shock sweeps to absorbance on the calibrated axis.",
                "4. Fit each usable sweep with the constrained three-transition Voigt model.",
                "5. Reduce the fitted linewidth and integrated area to scan-by-scan state history.",
                "",
                "## Output Map",
                f"- Baseline products: `{_relative_path(baseline_data, repo_root)}`",
                f"- Etalon calibration: `{_relative_path(etalon_data, repo_root)}`",
                f"- Shock absorbance export: `{_relative_path(shock_data, repo_root)}`",
                f"- Voigt fit export: `{_relative_path(voigt_fit_data, repo_root)}`",
                f"- State history export: `{_relative_path(state_history_data, repo_root)}`",
                "",
                "## Run Statistics",
                f"- Baseline sweeps averaged: {summary.baseline_sweep_count}",
                f"- Etalon peaks used in calibration: {summary.etalon_peak_count}",
                f"- Shock sweeps processed: {summary.shock_sweep_count}",
                (
                    "- Successful Voigt fits: "
                    f"{summary.successful_voigt_fits} / {summary.total_voigt_fits}"
                ),
                (
                    "- Temperature range [K]: "
                    f"{summary.temperature_range_k[0]:.3g} to {summary.temperature_range_k[1]:.3g}"
                ),
                (
                    "- Pressure range [atm]: "
                    f"{summary.pressure_range_atm[0]:.3g} to {summary.pressure_range_atm[1]:.3g}"
                ),
                (
                    "- CO mole-fraction range [-]: "
                    f"{summary.co_mole_fraction_range[0]:.3g} to {summary.co_mole_fraction_range[1]:.3g}"
                ),
                "",
            ]
        ),
        encoding="utf-8",
    )

    return summary
