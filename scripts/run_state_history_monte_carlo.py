"""Run or resume Monte Carlo state-history uncertainty propagation."""

from __future__ import annotations

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mock_lab.pipelines.monte_carlo_state_history import run_monte_carlo_state_history_pipeline


DEFAULT_VOIGT_FIT_DATA = REPO_ROOT / "data" / "processed" / "exports" / "voigt_fit_results.npz"
DEFAULT_STATE_HISTORY_DATA = REPO_ROOT / "results" / "tables" / "state_history.npz"
DEFAULT_MONTE_CARLO_OUTPUT_DIR = (
    REPO_ROOT / "results" / "monte_carlo" / "bath_gas_full_refit"
)
DEFAULT_FIGURE_OUTPUT_DIR = REPO_ROOT / "results" / "figures"


def main(
    voigt_fit_data: Path = DEFAULT_VOIGT_FIT_DATA,
    state_history_data: Path = DEFAULT_STATE_HISTORY_DATA,
    output_dir: Path = DEFAULT_MONTE_CARLO_OUTPUT_DIR,
    figure_output_dir: Path = DEFAULT_FIGURE_OUTPUT_DIR,
    *,
    trial_count: int = 1000,
    chunk_size: int = 100,
    workers: int = 1,
    seed: int = 617,
    confidence_level: float = 0.95,
    minimum_peak_absorbance: float = 0.02,
    force_restart: bool = False,
) -> None:
    """Run the Monte Carlo pipeline with IDE-friendly local defaults."""

    result = run_monte_carlo_state_history_pipeline(
        voigt_fit_data=voigt_fit_data,
        state_history_data=state_history_data,
        output_dir=output_dir,
        figure_output_dir=figure_output_dir,
        trial_count=trial_count,
        chunk_size=chunk_size,
        seed=seed,
        confidence_level=confidence_level,
        minimum_peak_absorbance=minimum_peak_absorbance,
        force=force_restart,
        workers=workers,
    )
    print(
        "Completed "
        f"{result.completed_trials} Monte Carlo trials at "
        f"{100.0 * result.confidence_level:.1f}% confidence."
    )


if __name__ == "__main__":
    voigt_fit_data = DEFAULT_VOIGT_FIT_DATA
    state_history_data = DEFAULT_STATE_HISTORY_DATA
    output_dir = DEFAULT_MONTE_CARLO_OUTPUT_DIR
    figure_output_dir = DEFAULT_FIGURE_OUTPUT_DIR

    trial_count = 1000
    chunk_size = 100
    workers = 12
    seed = 617
    confidence_level = 0.95
    minimum_peak_absorbance = 0.02
    force_restart = True

    main(
        voigt_fit_data=voigt_fit_data,
        state_history_data=state_history_data,
        output_dir=output_dir,
        figure_output_dir=figure_output_dir,
        trial_count=trial_count,
        chunk_size=chunk_size,
        workers=workers,
        seed=seed,
        confidence_level=confidence_level,
        minimum_peak_absorbance=minimum_peak_absorbance,
        force_restart=force_restart,
    )
