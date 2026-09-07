# Certifying Cooperation — reproducibility repository

This repository reproduces every plot, table and derived data file used in
_Certifying Cooperation: A Novel Approach to Cooperative Multi-Agent Task
Generation_ (Molinghen & Charels). It contains:

- [`scripts/`](scripts/) — the Python pipeline that turns raw experiment logs and
  layout pools into the paper's figures, tables and aggregated CSVs.
- [`data/`](data/) — the aggregated/derived CSVs the scripts produce (tracked in
  git; see [`data/toc.md`](data/toc.md) for a file-by-file description).
- [`latex/`](latex/) — the paper's LaTeX source.
- `plots/` — every generated figure, in PDF format.
- `logs/` and `layouts/` — raw experiment data and generated layout pools,
  downloaded separately (see below) and not tracked in git.

## 1. Requirements

- [`uv`](https://docs.astral.sh/uv/) (manages the Python 3.14 virtual environment
  and dependencies; this project does not run under a plain `python`/`pip`
  setup).
- `latexmk`/`pdflatex` on `PATH` if you want to compile the paper itself
  (not needed to regenerate its plots/tables).
- Enough disk space for the raw logs archive (~2.5 GB compressed, ~12 GB
  extracted) and the layouts archive (~15 MB compressed).

Install the Python dependencies with:

```bash
uv sync
```

All Python commands below must be run with `uv run python ...` from the repo
root — this project does not work with a bare `python`.

## 2. Download the raw data

Two archives are not tracked in git and must be downloaded before running any
script that reads from `logs/` or `layouts/`:

1. **Experiment logs** (training/evaluation CSVs for every run):
   [permalink.ulb.be/t/0geWoPXRKJ](https://permalink.ulb.be/t/0geWoPXRKJ).
   Download and extract so that the result is a `logs/` directory at the repo
   root (e.g. `logs/canonical-asymmetric-dqn-1M-0.5k/run-0/...`):

```sh
wget https://permalink.ulb.be/t/0geWoPXRKJ -O logs.zip
unzip logs.zip
```

2. **Layout pools** (the generated LLE layout files):
   [zenodo.org/records/22640557/files/layouts.zip](https://zenodo.org/records/22640557/files/layouts.zip).
   Download and extract so that the result is a `layouts/` directory at the
   repo root, containing `layouts/canonical/<profile>/` and
   `layouts/tuning/{cooperative,independent}/`.

```sh
wget https://zenodo.org/records/22640557/files/layouts.zip
unzip layouts.zip
```

## 3. Generate everything

Once `logs/` and `layouts/` are in place:

```bash
uv run python scripts/generate_data.py
uv run python scripts/plot_data.py
```

The first command scans the downloaded logs/layouts and writes reusable CSVs;
the second reads those CSVs and renders the figures, LaTeX fragments, and
console reports. This separation lets plots be restyled or selectively rendered
without repeating the expensive analysis. A full data pass takes a few minutes
on a modern multi-core machine and is dominated by the cooperation-profile
sweep. Together, the commands write:

- CSVs to `data/`
- every generated figure as a PDF in `plots/`
- generated LaTeX table fragments into `tables/` (gitignored scratch output —
  the paper's own tables are hand-written in
  `latex/content.tex`/`latex/appendix.tex`; use `tables/*.tex` to check their
  numbers still match, and update the paper source if they don't)

By default, the `pool-characteristics` target only characterizes the first
1000 layouts of each canonical pool (a few minutes; the full pools run into
the tens of thousands and take much longer to characterize one predicate at a
time). Pass `--layout-limit` to change that, e.g. to characterize every layout
of every pool instead:

```bash
uv run python scripts/generate_data.py --layout-limit 0
```

## 4. Generate or plot selected data

List the available data and presentation targets:

```bash
uv run python scripts/generate_data.py --list
uv run python scripts/plot_data.py --list
```

Run one or more targets either positionally or with repeatable keyword options:

```bash
uv run python scripts/generate_data.py generalization sat-duration
uv run python scripts/plot_data.py generalization sat-duration

# Equivalent keyword form
uv run python scripts/generate_data.py --generalization --sat-duration
uv run python scripts/plot_data.py --generalization --sat-duration
```

The cooperation-profile presentation can also be selected at figure granularity:

```bash
uv run python scripts/plot_data.py --coop-gap --coop-heatmap
```

Existing generated data is reused by default. Pass `--overwrite` to rerun the
selected data target and replace its outputs:

```bash
uv run python scripts/generate_data.py --generalization --overwrite
```

`generate_data.py --help` documents shared input/output controls such as
`--logs`, `--data`, `--layouts`, `--refresh`, and `--layout-limit`.
`plot_data.py --help` documents output destinations and rendering controls.
These are the project's only two executable entry points; implementation code
is grouped by scientific domain under `src/paper_plots/`.

## 5. Compiling the paper

The LaTeX source is in [`latex/`](latex/main.tex). With `latexmk` on `PATH`:

```bash
cd latex && latexmk -pdf main.tex
```

The paper's tables are hand-written (not `\input`-ed from `tables/`), so after
regenerating data, diff the relevant `tables/*.tex` fragment against the
corresponding table in `latex/content.tex`/`latex/appendix.tex` to confirm the
numbers still agree before editing the paper source.

## 6. Additional tools (not part of the default data/plot pass)

These operations are excluded from the default pass because they are slow or
require a specific input. They remain available as opt-in targets:

```bash
# Rebuild the raw 100k-layout acceptance experiment (slow)
uv run python scripts/generate_data.py --acceptance-raw

# Rerun the SAT clause-ablation measurements
uv run python scripts/generate_data.py --sat-ablation

# Generate an Independent-predicate layout pool
uv run python scripts/generate_data.py --generate-layouts \
    --layout-output layouts/generated/independent

# Extract laser counts from a particular pool
uv run python scripts/generate_data.py --laser-counts \
    --layout-source layouts/canonical/asymmetric

# Report which experiment runs and checkpoints are available
uv run python scripts/generate_data.py --availability

# Solve or replay one trajectory and render it
uv run python scripts/plot_data.py --trajectory \
    --layout layouts/example.txt --labelled
```

The SAT-duration and acceptance-rate summaries depend on precomputed raw
experiment CSVs. If those files are stored elsewhere, provide them with
`--sat-source`, `--acceptance-source`, and `--negative-query-source`. A default
pass reports and skips an unavailable source instead of aborting the remaining
targets.

## Notes on the data layout

- Run directories under `logs/` follow the pattern
  `canonical-<profile>-<algorithm>-1M-0.5k/run-<seed>/`, each holding the
  final-policy evaluation files (`test-policy-on-{train,test}-envs.csv`) that
  every analysis script reads, plus the older `test.csv`/`train.csv` on-policy
  endpoint that the `availability` target still tracks for completeness.
- `layouts/canonical/<profile>/` holds the layout pools used by the
  cooperation-profile sweep (`asymmetric`, `convergent-2`, `divergent-2`,
  `sequential-2`, `interdependent-2`); `layouts/tuning/{cooperative,independent}/`
  holds the 5x5 pools used by the pool-size generalization sweep.
