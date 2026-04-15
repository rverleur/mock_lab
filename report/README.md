# Report Workspace

This folder contains the LaTeX source for the ME 617 mock lab report.

## Build

Recommended command from this folder:

```bash
latexmk -xelatex main.tex
```

If you prefer a one-shot build:

```bash
xelatex main.tex
```

## Contents

- `main.tex` is the main report source.
- `references.bib` is the BibTeX database used by the report.
- `figures/` is for report figures.
- `tables/` is for exported table snippets or manually maintained table inputs.
- `sections/` is available if you want to split the report into separate files without changing the top-level layout.
