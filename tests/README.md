# Tests

This suite covers the active analysis path:

- MAT loading and sweep alignment
- absorbance preprocessing helpers
- HiTEMP parsing and curated transition loading
- TIPS and line-strength calculations
- Voigt fitting
- state estimation and uncertainty helpers
- Monte Carlo transition sampling
- baseline and full-pipeline smoke runs

Run everything with:

```bash
.venv/bin/python -m pytest -q
```
