# Third-Party Resources

This directory holds vendored external reference assets that are used by the analysis code but are not part of the experimental lab datasets.

Current contents:

- `hitran_tips/`: local HITRAN TIPS partition-sum script and lookup tables used by `src/mock_lab/spectroscopy/tips.py`

These files are kept here instead of under `data/` so the `data/` tree can stay focused on raw experimental inputs plus generated analysis products.
