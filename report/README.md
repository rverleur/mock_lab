# Report Workspace

This folder contains the LaTeX source for the ME 617 mock lab report.

## Formatting

The handout specifies:

- single spacing
- 1 inch margins
- 12 point Times New Roman font

The template in `main.tex` is set up to satisfy those requirements.

## Build

Recommended command from this folder:

```bash
latexmk -xelatex main.tex
```

If you prefer a one-shot build:

```bash
xelatex main.tex
```

## Suggested Organization

- `main.tex` is the main report source.
- `figures/` is for report figures.
- `tables/` is for table inputs or exported table snippets.
- `sections/` is available if you later want to split the report into separate files.
