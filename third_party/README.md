# Third-Party Resources

This directory holds vendored external reference assets used by the analysis code.

Current contents:

- `hitran_tips/`: local HITRAN TIPS lookup tables used by `src/mock_lab/spectroscopy/tips.py`.
- `HiTEMP/mock_lab_co_transitions.csv`: the curated three-transition export used by the runtime code.

Optional larger local assets can also live here when you need to regenerate the curated transition file:

- `HiTEMP/05_HITEMP2019.par`
- `HiTEMP/05_HITEMP2019.csv`

These resources are kept out of `data/` so that the `data/` tree stays focused on experimental inputs and generated products.
