# Plotting Helpers

`src/mock_lab/plotting/` keeps the figure-building code separate from the numerical reduction.

## What Lives Here

- `figures.py`: reusable plotting functions for time traces, frequency-domain sweeps, Voigt-fit overlays, and state-history figures.
- `mpl.py`: the shared Matplotlib entry point used by the rest of the repository.

## Backend Behavior

All pipeline code should import `plt` from `mock_lab.plotting.mpl`, not directly from `matplotlib.pyplot`.

That helper does two things centrally:

- forces the non-interactive `Agg` backend so scripts run cleanly in headless environments
- points `MPLCONFIGDIR` at a writable temporary directory so Matplotlib does not fail when the default config location is unavailable or read-only

This keeps plotting side effects out of the analysis code and avoids each script reimplementing backend setup.
