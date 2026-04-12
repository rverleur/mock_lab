"""Plot Monte Carlo state-history uncertainty from a saved summary file."""

from __future__ import annotations

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mock_lab.pipelines.monte_carlo_state_history import plot_monte_carlo_state_history


DEFAULT_SUMMARY_DATA = (
    REPO_ROOT
    / "results"
    / "monte_carlo"
    / "bath_gas_full_refit"
    / "state_history_monte_carlo_summary.npz"
)
DEFAULT_FIGURE_OUTPUT_DIR = REPO_ROOT / "results" / "figures"


def main(
    summary_data: Path = DEFAULT_SUMMARY_DATA,
    figure_output_dir: Path = DEFAULT_FIGURE_OUTPUT_DIR,
) -> None:
    """Rebuild the MC state-history figure from saved data."""

    output_path = plot_monte_carlo_state_history(
        summary_data,
        figure_output_dir=figure_output_dir,
    )
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    summary_data = DEFAULT_SUMMARY_DATA
    figure_output_dir = DEFAULT_FIGURE_OUTPUT_DIR

    main(
        summary_data=summary_data,
        figure_output_dir=figure_output_dir,
    )
