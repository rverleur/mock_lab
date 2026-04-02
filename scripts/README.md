# Wrapper Scripts

These scripts are intended to be your run targets from an IDE.

Each wrapper:

- resolves repo-local paths
- imports the stage-level pipeline from `src/mock_lab/`
- calls a `main()` function under an `if __name__ == "__main__":` block

Suggested order while building the project:

1. `run_baseline.py`
2. `run_etalon.py`
3. `run_shock_snapshot.py`
4. `run_time_history.py`
5. `run_full_pipeline.py`
6. `run_report_figures.py`

Right now the wrappers call placeholder pipeline functions that intentionally raise `NotImplementedError`.
