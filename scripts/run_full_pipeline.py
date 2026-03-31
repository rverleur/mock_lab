"""Run the end-to-end mock-lab workflow from an IDE or terminal."""

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mock_lab.pipelines.full_pipeline import run_full_pipeline


def main() -> None:
    """Wire repo-local paths into the top-level pipeline."""

    data_root = REPO_ROOT / "data"
    results_dir = REPO_ROOT / "results"
    run_full_pipeline(data_root=data_root, results_dir=results_dir)


if __name__ == "__main__":
    main()
