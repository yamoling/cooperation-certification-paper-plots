# Certifying Cooperation — reproducibility repository

This repository reproduces every plot, table and derived data file used in
_Certifying Cooperation: A Novel Approach to Cooperative Multi-Agent Task
Generation_ (Molinghen & Charels). It contains:

- [`scripts/`](scripts/) — the Python pipeline that turns raw experiment logs and
  layout pools into the paper's figures, tables and aggregated CSVs.
- [`data/`](data/) — the aggregated/derived CSVs the scripts produce (tracked in
  git; see [`data/toc.md`](data/toc.md) for a file-by-file description).
- [`latex/`](latex/) — the paper's LaTeX source, including the actual `plots/`
  and `pictures/` the compiled paper embeds.
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
uv run python scripts/generate_all.py
```

This regenerates every figure and table backed by the downloaded data (a few
minutes on a modern multi-core machine, dominated by the cooperation-profile
sweep). It writes:

- CSVs to `data/`
- the paper's figures directly into `latex/plots/` and `latex/pictures/`
  (so `git diff` shows exactly what changed against the committed originals)
- generated LaTeX table fragments into `tables/` (gitignored scratch output —
  the paper's own tables are hand-written in
  `latex/content.tex`/`latex/appendix.tex`; use `tables/*.tex` to check their
  numbers still match, and update the paper source if they don't)
- a few exploratory, paper-unrelated figures into `plots/` (also gitignored)

By default, the `pool-characteristics` target only characterizes the first
1000 layouts of each canonical pool (a few minutes; the full pools run into
the tens of thousands and take much longer to characterize one predicate at a
time). Pass `--layout-limit` to change that, e.g. to characterize every layout
of every pool instead:

```bash
uv run python scripts/generate_all.py --layout-limit 0
```

## 4. Generate a single plot or table

List the available targets:

```bash
uv run python scripts/generate_all.py --list
```

Then run one or more of them by name:

```bash
uv run python scripts/generate_all.py coop-profiles
uv run python scripts/generate_all.py generalization sat-duration
```

Each target is a thin wrapper around one of the scripts in `scripts/`; every
script also runs standalone with its own `--help` for finer control (custom
input/output paths, a single figure instead of a full run, etc.):

```bash
uv run python scripts/coop_profiles.py --help
```

| Target                           | What it reproduces                                                                                                                                       |
| -------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `coop-profiles`                  | The cooperation-profile sweep: `coop-profiles-{difficulty,gap,heatmaps}` figures, the algorithm x profile table, and ~30 supporting CSVs in `data/`.     |
| `coop-profile-certificates`      | Certificate/completeness-theorem validation and outcome-conditioned predicate tables (`apx:tab:coop-profile-exit-outcomes`, `tab:coop-profile-summary`). |
| `coop-profile-failures`          | The death/timeout/joint-success table (`tab:coop-profile-failures`).                                                                                     |
| `laser-colour`                   | Whether an agent exits more often when a laser of its own colour is present.                                                                             |
| `pool-characteristics`           | Cooperation-predicate incidence across each canonical layout pool.                                                                                       |
| `layouts-picture`                | `pictures/layouts.pdf` — sampled layouts per canonical profile pool.                                                                                     |
| `generalization-layouts-picture` | `pictures/generalization-layouts-5x5.pdf` — sampled 5x5 independent/cooperative layouts.                                                                 |
| `generalization`                 | `latex/plots/generalization-joint-5x5.pdf` — the pool-size generalization figure.                                                                        |
| `sat-duration`                   | `latex/plots/solving_duration-comparison.pdf` — SAT construction/solving duration vs. horizon.                                                           |
| `acceptance-rates`               | The certification-yield table and the certification acceptance-rate/timing plots.                                                                        |
| `exit-outcomes`                  | `tab:exit-outcomes` — the 5x5-sweep exit-outcome table.                                                                                                  |
| `trajectory-profiles`            | Per-episode cooperation-predicate rates backing the results-section prose.                                                                               |
| `exit-rate-distribution`         | A second, independent computation of the exit-outcome distribution.                                                                                      |

## 5. Compiling the paper

The LaTeX source is in [`latex/`](latex/main.tex). With `latexmk` on `PATH`:

```bash
cd latex && latexmk -pdf main.tex
```

The paper's tables are hand-written (not `\input`-ed from `tables/`), so after
regenerating data, diff the relevant `tables/*.tex` fragment against the
corresponding table in `latex/content.tex`/`latex/appendix.tex` to confirm the
numbers still agree before editing the paper source.

## 6. Additional tools (not part of `generate_all.py`)

These either regenerate raw measurement data from scratch (slow, and the
results are already committed under `data/`) or are one-off/manual tools, so
they are documented here rather than wired into the default pipeline:

- `scripts/acceptance_rates_experiment.py` — resamples and re-certifies random
  layouts to rebuild `data/acceptance-rates.csv` (hours for the full 100k-layout
  run used in the paper).
- `scripts/sat_measurements_mode.py` — reruns the SAT clause-ablation sweep
  that produces `data/sat-measurements-mode.csv`.
- `scripts/generate_layouts.py` — generates new `Independent`-predicate
  layouts; the profile pools themselves come from the downloaded
  `layouts.zip`, not from this script.
- `scripts/extract_test_set_lasers.py` — recomputes a `<pool>-laser-colours.csv`
  from a directory of layout files, given the specific layout subset used as a
  pool.
- `scripts/draw_trajectory.py` — solves (or replays a supplied trajectory
  through) a single layout and renders it; used to hand-compose the
  `pictures/{detour,free-help-*}.png` illustrations.
- `scripts/data-availability-summary.py` — prints a markdown snapshot of which
  `logs/` runs are complete; a diagnostic tool, not a paper figure/table.

## Notes on the data layout

- Run directories under `logs/` follow the pattern
  `canonical-<profile>-<algorithm>-1M-0.5k/run-<seed>/`, each holding the
  final-policy evaluation files (`test-policy-on-{train,test}-envs.csv`) that
  every analysis script reads, plus the older `test.csv`/`train.csv` on-policy
  endpoint that some analyses (e.g. `data-availability-summary.py`) still
  track for completeness.
- `layouts/canonical/<profile>/` holds the layout pools used by the
  cooperation-profile sweep (`asymmetric`, `convergent-2`, `divergent-2`,
  `sequential-2`, `interdependent-2`); `layouts/tuning/{cooperative,independent}/`
  holds the 5x5 pools used by the pool-size generalization sweep.
