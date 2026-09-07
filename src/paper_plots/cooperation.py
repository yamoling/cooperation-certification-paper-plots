"""Cooperation-profile sweep analysis and presentation."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.lines import Line2D

PROFILES = ("asymmetric", "convergent", "divergent", "sequential", "interdependent")
ALGORITHMS = ("dqn", "ippo", "mappo", "qmix", "vdn")
N_AGENTS = 3
FINAL_STEP = 1_000_000
EXPECTED_CHECKPOINTS = 21
EVAL_EPISODES = 500
# Cells below this many completed runs are excluded from pooled profile comparisons.
MIN_CELL_RUNS = 30
N_PERMUTATIONS = 20_000
N_BOOTSTRAPS = 20_000
RNG_SEED = 20260731

# The final-policy evaluation files replace test.csv/train.csv: both are full time
# series over the same 21 checkpoints, both recorded under the greedy test policy, so
# the train/test difference is a pure layout-pool effect.
POOL_FILES = {
    "train": "test-policy-on-train-envs.csv",
    "test": "test-policy-on-test-envs.csv",
}
# `env.offset` / `test_env.offset` in experiment.json, with `sequential = True`: the
# k-th episode of a checkpoint replays layout `offset + k` of the pool directory.
POOL_OFFSETS = {"train": 0, "test": 500}
PREDICATES = (
    "cooperative",
    "asymmetric",
    "chained",
    "convergent",
    "divergent",
    "interdependent",
)
AGENT_EXIT_COLUMNS = tuple(f"agent-{i}-exited" for i in range(N_AGENTS))
AGENT_ALIVE_COLUMNS = tuple(f"agent-{i}-alive" for i in range(N_AGENTS))

PROFILE_LABELS = {
    "asymmetric": r"$\mathrm{Asym}$",
    "convergent": r"$\mathrm{Conv}_2$",
    "divergent": r"$\mathrm{Div}_2$",
    "sequential": r"$\mathrm{Seq}_2$",
    "interdependent": r"$\mathrm{Inter}_2$",
}
# Stem of the per-layout tables in `data/`; only the asymmetric pool drops the `-2`.
LAYOUT_FAMILIES = {
    "asymmetric": "asymmetric",
    "convergent": "convergent-2",
    "divergent": "divergent-2",
    "sequential": "sequential-2",
    "interdependent": "interdependent-2",
}
# The sequential profile is generated from the `chained-2` layout family, and its
# trajectory predicate is logged as `chained-trajectory`.
PROFILE_PREDICATES = {
    "asymmetric": "asymmetric",
    "convergent": "convergent",
    "divergent": "divergent",
    "sequential": "chained",
    "interdependent": "interdependent",
}
ALGORITHM_LABELS = {
    "dqn": "DQN",
    "ippo": "IPPO",
    "mappo": "MAPPO",
    "qmix": "QMIX",
    "vdn": "VDN",
}
TABLE_ALGORITHM_ORDER = ("vdn", "ippo", "dqn", "mappo", "qmix")
TABLE_PROFILE_ORDER = (
    "asymmetric",
    "convergent",
    "sequential",
    "interdependent",
    "divergent",
)
TABLE_PROFILE_LABELS = {
    "asymmetric": r"$\asym$",
    "convergent": r"$\conv_2$",
    "sequential": r"$\seq_2$",
    "interdependent": r"$\inter_2$",
    "divergent": r"$\div_2$",
}
# Algorithms and profiles are never the same categorical dimension in a figure, but they
# do appear in adjacent figures throughout the paper, so they draw from two different
# qualitative matplotlib colormaps to avoid a colour accidentally meaning "algorithm" in
# one panel and "profile" in the next.
ALGORITHM_COLORS = dict(zip(ALGORITHMS, plt.colormaps["tab10"].colors))
PROFILE_COLORS = dict(zip(PROFILES, plt.colormaps["Set2"].colors))

AVERAGE_COLOR = "black"

# Name -> help text for the --<name> flag that restricts rendering to that one figure.
PLOT_CHOICES = {
    "curves": "Per-algorithm train/test exit rate curves (coop-profiles-test/-train).",
    "gap": "Train/test generalisation gap over training (coop-profiles-gap).",
    "difficulty": "Algorithm-averaged test curve and final pool-level performance (coop-profiles-difficulty).",
    "heatmap": "Algorithm-by-profile endpoint heatmap (coop-profiles-algorithms).",
}

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "text.latex.preamble": r"\usepackage{amsmath}",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.axisbelow": True,
        "grid.linewidth": 0.4,
        "grid.alpha": 0.4,
    }
)


def discover_experiments(log_root: Path) -> list[tuple[str, str, Path]]:
    """List the ``(profile, algorithm, directory)`` triples of the profile sweep.

    Directory names are ``canonical-<profile>[-<k>]-<algo>-1M-0.5k``. Failed reruns
    are skipped.

    @ai-generated
    """
    found: list[tuple[str, str, Path]] = []
    for directory in sorted(log_root.iterdir()):
        if not directory.is_dir() or directory.name.endswith("-failed"):
            continue
        parts = directory.name.split("-")
        if not directory.name.startswith("canonical-") or len(parts) < 5:
            continue
        profile, algorithm = parts[1], parts[-3]
        if profile not in PROFILES or algorithm not in ALGORITHMS:
            continue
        found.append((profile, algorithm, directory))
    if not found:
        raise FileNotFoundError(
            f"No cooperation-profile experiment found in {log_root}"
        )
    return found


def has_data(path: Path) -> bool:
    """Tell whether a log file exists and is not a zero-byte placeholder.

    @ai-generated
    """
    return path.exists() and path.stat().st_size > 0


def read_eval_curve(path: Path) -> pl.DataFrame | None:
    """Summarise one final-policy evaluation file into one row per checkpoint.

    Returns ``None`` when the file is still being written, i.e. when it does not hold
    the full 21 checkpoints of 500 episodes. Besides the exit rate this extracts the
    outcome distribution, the death count and the six cooperation predicates.

    @ai-generated
    """
    columns = [
        "exit_rate",
        "time_step",
        "test_num",
        *AGENT_EXIT_COLUMNS,
        *AGENT_ALIVE_COLUMNS,
        *(f"{name}-trajectory" for name in PREDICATES),
    ]
    try:
        episodes = pl.read_csv(path, columns=columns)
    except pl.exceptions.PolarsError, OSError:
        # The evaluation sweep may be writing this file right now.
        return None
    curve = (
        episodes.unique(["time_step", "test_num"], keep="last", maintain_order=True)
        .with_columns(
            pl.sum_horizontal(AGENT_EXIT_COLUMNS).alias("n_exited"),
            (N_AGENTS - pl.sum_horizontal(AGENT_ALIVE_COLUMNS)).alias("n_dead"),
        )
        .group_by("time_step")
        .agg(
            pl.col("exit_rate").mean(),
            (pl.col("n_exited") == N_AGENTS).mean().alias("all_exit"),
            (pl.col("n_exited") == 0).mean().alias("none_exit"),
            (pl.col("n_exited") == 1).mean().alias("one_exit"),
            pl.col("n_dead").mean().alias("deaths"),
            (pl.col("n_dead") > 0).mean().alias("any_death"),
            *(pl.col(column).mean().alias(column) for column in AGENT_EXIT_COLUMNS),
            *(
                pl.col(f"{name}-trajectory").mean().alias(f"predicate_{name}")
                for name in PREDICATES
            ),
            pl.len().alias("n_episodes"),
        )
        .sort("time_step")
    )
    if (
        curve.height != EXPECTED_CHECKPOINTS
        or curve["n_episodes"].min() != EVAL_EPISODES
    ):
        return None
    return curve


def collect_curves(log_root: Path) -> pl.DataFrame:
    """Read the final-policy evaluations of every run.

    @ai-generated
    """
    evaluation: list[pl.DataFrame] = []
    for profile, algorithm, directory in discover_experiments(log_root):
        for run in sorted(directory.glob("run-*")):
            seed = int(run.name.removeprefix("run-"))
            context = {
                "profile": pl.lit(profile),
                "algorithm": pl.lit(algorithm),
                "seed": pl.lit(seed),
            }
            for pool, filename in POOL_FILES.items():
                if not has_data(run / filename):
                    continue
                curve = read_eval_curve(run / filename)
                if curve is not None:
                    evaluation.append(curve.with_columns(pool=pl.lit(pool), **context))
    if not evaluation:
        raise FileNotFoundError("No complete final-policy evaluation found")
    return pl.concat(evaluation)


def read_final_layout_scores(path: Path) -> pl.DataFrame | None:
    """Read one final checkpoint as one score per replayed test layout.

    Evaluation replays the 500 layouts sequentially at every checkpoint, so row order
    within the final checkpoint identifies the layout. Keeping these rows unaggregated
    permits resampling whole layouts across every seed and algorithm that evaluated them.
    Duplicate appended evaluation blocks are collapsed by their test index.

    @ai-generated
    """
    try:
        episodes = pl.read_csv(path, columns=["time_step", "test_num", "exit_rate"])
    except pl.exceptions.PolarsError, OSError:
        return None
    endpoint = episodes.filter(pl.col("time_step") == FINAL_STEP).unique(
        "test_num", keep="last", maintain_order=True
    )
    if endpoint.height != EVAL_EPISODES:
        return None
    return endpoint.select("exit_rate").with_row_index("layout_index")


def collect_final_layout_scores(log_root: Path) -> pl.DataFrame:
    """Collect final test scores while retaining the shared layout identifier.

    @ai-generated
    """
    frames: list[pl.DataFrame] = []
    for profile, algorithm, directory in discover_experiments(log_root):
        for run in sorted(directory.glob("run-*")):
            path = run / POOL_FILES["test"]
            if not has_data(path):
                continue
            scores = read_final_layout_scores(path)
            if scores is None:
                continue
            frames.append(
                scores.with_columns(
                    profile=pl.lit(profile),
                    algorithm=pl.lit(algorithm),
                    seed=pl.lit(int(run.name.removeprefix("run-"))),
                )
            )
    if not frames:
        raise FileNotFoundError("No complete final-policy test evaluation found")
    return pl.concat(frames)


def load_final_layout_scores(
    log_root: Path, cache_dir: Path, *, refresh: bool
) -> pl.DataFrame:
    """Load or build the cache used by the layout-block bootstrap.

    @ai-generated
    """
    cache = cache_dir / "final_test_layout_scores.csv"
    if not refresh and cache.exists():
        return pl.read_csv(cache)
    scores = collect_final_layout_scores(log_root)
    cache_dir.mkdir(parents=True, exist_ok=True)
    scores.write_csv(cache)
    return scores


def read_test_layout_curve(path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    """Read one test evaluation as a checkpoint-by-layout exit-rate matrix.

    Rows within each checkpoint follow the deterministic layout replay order, so the
    column index identifies the same test layout throughout training. Duplicate appended
    evaluation blocks are collapsed by checkpoint and test index.

    @ai-generated
    """
    try:
        episodes = pl.read_csv(path, columns=["time_step", "test_num", "exit_rate"])
    except pl.exceptions.PolarsError, OSError:
        return None
    episodes = episodes.unique(
        ["time_step", "test_num"], keep="last", maintain_order=True
    )
    steps = episodes["time_step"].unique().sort().to_numpy()
    if steps.size != EXPECTED_CHECKPOINTS:
        return None
    rows: list[np.ndarray] = []
    for step in steps:
        values = episodes.filter(pl.col("time_step") == step)["exit_rate"].to_numpy()
        if values.size != EVAL_EPISODES:
            return None
        rows.append(values)
    return steps, np.vstack(rows)


def collect_test_layout_curves(
    log_root: Path,
    *,
    selected_seeds: set[int] | None = None,
) -> pl.DataFrame:
    """Average selected seeds while retaining algorithm, checkpoint and layout identity.

    Accumulating numeric matrices avoids materializing every episode from all 750 runs
    in one large data frame. The returned cache remains algorithm-specific so later
    analyses can choose an algorithm subset without rebuilding it.

    @ai-generated
    """
    sums: dict[tuple[str, str], np.ndarray] = {}
    counts: dict[tuple[str, str], int] = {}
    reference_steps: np.ndarray | None = None
    for profile, algorithm, directory in discover_experiments(log_root):
        key = (profile, algorithm)
        for run in sorted(directory.glob("run-*")):
            seed = int(run.name.removeprefix("run-"))
            if selected_seeds is not None and seed not in selected_seeds:
                continue
            path = run / POOL_FILES["test"]
            if not has_data(path):
                continue
            result = read_test_layout_curve(path)
            if result is None:
                continue
            steps, values = result
            if reference_steps is None:
                reference_steps = steps
            elif not np.array_equal(reference_steps, steps):
                raise ValueError(f"Checkpoint grid differs in {path}")
            if key not in sums:
                sums[key] = np.zeros_like(values, dtype=float)
                counts[key] = 0
            sums[key] += values
            counts[key] += 1
    if reference_steps is None or not sums:
        raise FileNotFoundError("No complete per-layout test curves found")

    frames: list[pl.DataFrame] = []
    layout_indices = np.tile(np.arange(EVAL_EPISODES), reference_steps.size)
    time_steps = np.repeat(reference_steps, EVAL_EPISODES)
    for (profile, algorithm), total in sums.items():
        seed_mean = total / counts[(profile, algorithm)]
        frames.append(
            pl.DataFrame(
                {
                    "profile": [profile] * seed_mean.size,
                    "algorithm": [algorithm] * seed_mean.size,
                    "time_step": time_steps,
                    "layout_index": layout_indices,
                    "exit_rate": seed_mean.reshape(-1),
                    "n_seeds": [counts[(profile, algorithm)]] * seed_mean.size,
                }
            )
        )
    return pl.concat(frames).sort("profile", "algorithm", "time_step", "layout_index")


def load_test_layout_curves(
    log_root: Path,
    cache_dir: Path,
    *,
    refresh: bool,
    selected_seeds: set[int] | None = None,
) -> pl.DataFrame:
    """Load or build algorithm-level per-layout curves for selected seeds.

    @ai-generated
    """
    cache = cache_dir / "test_layout_curves_by_algorithm.csv"
    if not refresh and cache.exists():
        cached = pl.read_csv(cache)
        if selected_seeds is None or set(cached["n_seeds"].unique()) == {
            len(selected_seeds)
        }:
            return cached
    curves = collect_test_layout_curves(log_root, selected_seeds=selected_seeds)
    cache_dir.mkdir(parents=True, exist_ok=True)
    curves.write_csv(cache)
    return curves


def load_curves(log_root: Path, cache_dir: Path, *, refresh: bool) -> pl.DataFrame:
    """Return the per-run curves, reusing the cached CSV unless a refresh is asked for.

    @ai-generated
    """
    evaluation_cache = cache_dir / "per_run_eval_curves.csv"
    if not refresh and evaluation_cache.exists():
        return pl.read_csv(evaluation_cache)
    evaluation = collect_curves(log_root)
    cache_dir.mkdir(parents=True, exist_ok=True)
    evaluation.write_csv(evaluation_cache)
    return evaluation


def coverage(curves: pl.DataFrame) -> pl.DataFrame:
    """Count the complete runs available per profile, algorithm and pool.

    @ai-generated
    """
    evaluation = (
        curves.group_by("profile", "algorithm", "pool")
        .agg(pl.col("seed").n_unique().alias("n_runs"))
        .pivot(on="pool", index=["profile", "algorithm"], values="n_runs")
    )
    for pool in POOL_FILES:
        if pool not in evaluation.columns:
            evaluation = evaluation.with_columns(pl.lit(None, pl.Int64).alias(pool))
    return (
        evaluation.rename({p: f"eval_{p}" for p in POOL_FILES})
        .fill_null(0)
        .sort("profile", "algorithm")
    )


def restrict_to_common_seeds(curves: pl.DataFrame) -> tuple[pl.DataFrame, list[int]]:
    """Keep only seed IDs available in every profile-algorithm-pool setting.

    Using the shared intersection balances all settings while preserving seed IDs as
    blocks for cross-profile comparisons. As more runs finish, the retained set grows
    automatically once those seeds are available everywhere.

    @ai-generated
    """
    seed_lists = (
        curves.group_by("profile", "algorithm", "pool")
        .agg(pl.col("seed").unique())
        .get_column("seed")
        .to_list()
    )
    common_seeds = sorted(set.intersection(*(set(seeds) for seeds in seed_lists)))
    if not common_seeds:
        raise ValueError(
            "No seed is available across every profile/algorithm/pool setting."
        )
    return curves.filter(pl.col("seed").is_in(common_seeds)), common_seeds


def drop_thin_cells(curves: pl.DataFrame, min_cell_runs: int) -> pl.DataFrame:
    """Remove profile-algorithm-pool cells that too few runs have reached.

    The evaluation sweep runs configuration by configuration, so a cell can hold a
    single run; plotting it would show a curve with no usable confidence band.

    @ai-generated
    """
    counts = curves.group_by("profile", "algorithm", "pool").agg(
        pl.col("seed").n_unique().alias("n_seeds")
    )
    return curves.join(
        counts.filter(pl.col("n_seeds") >= min_cell_runs).drop("n_seeds"),
        on=["profile", "algorithm", "pool"],
        how="inner",
    )


def aggregate_curve(curves: pl.DataFrame, metric: str = "exit_rate") -> pl.DataFrame:
    """Average a per-run metric over seeds with a 95% normal confidence interval.

    @ai-generated
    """
    return (
        curves.group_by("profile", "algorithm", "pool", "time_step")
        .agg(
            pl.col(metric).mean().alias("mean"),
            pl.col(metric).std().alias("std"),
            pl.len().alias("n"),
        )
        .with_columns(
            (1.96 * pl.col("std") / pl.col("n").sqrt()).fill_null(0.0).alias("ci")
        )
        .sort("profile", "algorithm", "time_step")
    )


def gap_trajectory(curves: pl.DataFrame) -> pl.DataFrame:
    """Pair the two pools at every checkpoint, giving the transfer gap over training.

    A gap that widens as training proceeds is the signature of overfitting to the
    training layouts; a gap that is constant from early on is a fixed generalisation
    deficit instead. Both pools use the same greedy policy, so the difference is
    attributable to the layout pool alone.

    @ai-generated
    """
    paired = curves.pivot(
        on="pool",
        index=["profile", "algorithm", "seed", "time_step"],
        values="exit_rate",
    )
    if "train" not in paired.columns or "test" not in paired.columns:
        return pl.DataFrame()
    return (
        paired.drop_nulls()
        .rename({"train": "train_exit_rate", "test": "test_exit_rate"})
        .with_columns(
            (pl.col("train_exit_rate") - pl.col("test_exit_rate")).alias("gap")
        )
        .sort("profile", "algorithm", "seed", "time_step")
    )


def gap_widening_tests(
    trajectory: pl.DataFrame, rng: np.random.Generator, midpoint: int = FINAL_STEP // 2
) -> pl.DataFrame:
    """Test whether the transfer gap keeps widening over the second half of training.

    Widening late is the discriminating evidence: every run's gap grows early simply
    because it starts at zero, so only the late trend separates continuing overfitting
    from a gap that has settled.

    @ai-generated
    """
    if trajectory.is_empty():
        return pl.DataFrame()
    span = trajectory.filter(pl.col("time_step").is_in([midpoint, FINAL_STEP])).pivot(
        on="time_step", index=["profile", "algorithm", "seed"], values="gap"
    )
    early, late = str(midpoint), str(FINAL_STEP)
    if early not in span.columns or late not in span.columns:
        return pl.DataFrame()
    span = span.drop_nulls().with_columns(
        (pl.col(late) - pl.col(early)).alias("widening")
    )
    tests: list[dict[str, object]] = []
    for keys, panel in span.group_by(["profile", "algorithm"]):
        differences = panel["widening"].to_numpy()
        tests.append(
            {
                "profile": keys[0],
                "algorithm": keys[1],
                "gap_at_midpoint": float(panel[early].mean()),
                "gap_at_end": float(panel[late].mean()),
                "widening": float(differences.mean()),
                "frac_seeds_widened": float((differences > 0).mean()),
                "p_value": paired_permutation_p_value(differences, rng),
            }
        )
    return pl.DataFrame(holm_correct(tests)).sort(
        "profile", "widening", descending=[False, True]
    )


def plot_gap_trajectory(trajectory: pl.DataFrame, output_stem: Path) -> None:
    """Draw both pool curves and the transfer gap over training, per algorithm."""
    if trajectory.is_empty():
        return
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4), constrained_layout=True)
    summary = (
        trajectory.group_by("algorithm", "time_step")
        .agg(
            pl.col("train_exit_rate").mean().alias("train"),
            pl.col("test_exit_rate").mean().alias("test"),
            pl.col("gap").mean().alias("gap"),
            (1.96 * pl.col("gap").std() / pl.len().sqrt()).alias("gap_ci"),
        )
        .sort("algorithm", "time_step")
    )
    for algorithm in ALGORITHMS:
        values = summary.filter(pl.col("algorithm") == algorithm)
        if values.is_empty():
            continue
        steps = values["time_step"].to_numpy() / 1e6
        colour = ALGORITHM_COLORS[algorithm]
        axes[0].plot(
            steps,
            values["train"].to_numpy(),
            color=colour,
            linewidth=1.3,
            linestyle="--",
        )
        axes[0].plot(steps, values["test"].to_numpy(), color=colour, linewidth=1.3)
        gap = values["gap"].to_numpy()
        ci = values["gap_ci"].to_numpy()
        axes[1].plot(steps, gap, color=colour, linewidth=1.4)
        axes[1].fill_between(
            steps, gap - ci, gap + ci, color=colour, alpha=0.12, linewidth=0
        )

    axes[0].set_title("Pool-wise exit rate")
    style_handles = [
        Line2D([], [], color="black", linestyle="--", label="Train pool"),
        Line2D([], [], color="black", linestyle="-", label="Test pool"),
    ]
    axes[0].legend(
        handles=style_handles,
        loc="upper left",
        bbox_to_anchor=(0.0, 0.975),
        frameon=False,
    )
    axes[1].set_title("Train--test pool gap")
    for axis in axes:
        axis.set_xlabel(r"Time steps ($\times 10^6$)")
        axis.set_xlim(0, 1.0)
        axis.set_ylim(bottom=0)
        axis.grid(True)
    axes[0].set_ylabel("Exit rate")
    axes[1].set_ylabel("Exit rate gap")
    handles = [
        Line2D([], [], color=ALGORITHM_COLORS[a], label=ALGORITHM_LABELS[a])
        for a in ALGORITHMS
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.1),
        ncol=len(ALGORITHMS),
        # fontsize=8,
    )
    save(fig, output_stem)


def run_level_metrics(curves: pl.DataFrame) -> pl.DataFrame:
    """Reduce each run to its endpoint, peak, average and learning-gain scores.

    @ai-generated
    """
    endpoint = pl.col("time_step") == FINAL_STEP
    aggregations = [
        pl.col("exit_rate").mean().alias("auc"),
        pl.col("exit_rate").max().alias("peak"),
        pl.col("time_step").sort_by("exit_rate").last().alias("peak_step"),
        pl.col("exit_rate").filter(pl.col("time_step") == 0).first().alias("baseline"),
        pl.col("exit_rate").filter(endpoint).first().alias("final"),
    ]
    aggregations += [
        pl.col(column).filter(endpoint).first().alias(column)
        for column in (
            "all_exit",
            "none_exit",
            "one_exit",
            "deaths",
            "any_death",
            *AGENT_EXIT_COLUMNS,
            *(f"predicate_{name}" for name in PREDICATES),
        )
    ]
    return (
        curves.group_by("profile", "algorithm", "pool", "seed")
        .agg(aggregations)
        .with_columns(
            pl.when(pl.col("peak") > 0)
            .then(pl.col("final") / pl.col("peak"))
            .otherwise(None)
            .alias("retention"),
            (pl.col("final") - pl.col("baseline")).alias("gain"),
        )
        .sort("profile", "algorithm", "pool", "seed")
    )


def summarise(metrics: pl.DataFrame, *by: str) -> pl.DataFrame:
    """Aggregate run-level metrics with 95% normal confidence intervals.

    @ai-generated
    """
    aggregations: list[pl.Expr] = [pl.len().alias("n_runs")]
    for metric in (
        "final",
        "baseline",
        "gain",
        "auc",
        "peak",
        "retention",
        "all_exit",
        "one_exit",
        "none_exit",
        "deaths",
        "any_death",
        *(f"predicate_{name}" for name in PREDICATES),
    ):
        aggregations.extend(
            [
                pl.col(metric).mean().alias(metric),
                (1.96 * pl.col(metric).std() / pl.len().sqrt()).alias(f"{metric}_ci"),
            ]
        )
    aggregations.append(pl.col("peak_step").median().alias("median_peak_step"))
    return metrics.group_by(*by).agg(aggregations).sort(*by)


def average_final_across_profiles(metrics: pl.DataFrame) -> pl.DataFrame:
    """Average each seed across profiles, then form algorithm-level confidence intervals.

    Averaging profiles within seeds keeps every profile equally weighted and retains seeds
    as the sampling units for the 95% normal confidence intervals.

    @ai-generated
    """
    per_seed = metrics.group_by("algorithm", "seed").agg(
        pl.col("final").mean(),
        pl.col("profile").n_unique().alias("n_profiles"),
        pl.len().alias("n_rows"),
    )
    incomplete = per_seed.filter(
        (pl.col("n_profiles") != len(TABLE_PROFILE_ORDER))
        | (pl.col("n_rows") != len(TABLE_PROFILE_ORDER))
    )
    if not incomplete.is_empty():
        raise ValueError(
            "Every algorithm/seed block must contain exactly one run for each profile."
        )
    return per_seed.group_by("algorithm").agg(
        pl.col("final").mean(),
        (1.96 * pl.col("final").std() / pl.len().sqrt()).alias("final_ci"),
    )


def average_final_across_algorithms(metrics: pl.DataFrame) -> pl.DataFrame:
    """Average algorithms within seeds, then form profile-level confidence intervals.

    The row labelled ``average`` also averages profiles within each seed, ensuring that
    every algorithm and profile receives equal weight while seeds remain the sampling
    units for the confidence intervals.

    @ai-generated
    """
    per_profile_seed = metrics.group_by("profile", "seed").agg(
        pl.col("final").mean(),
        pl.col("algorithm").n_unique().alias("n_algorithms"),
        pl.len().alias("n_rows"),
    )
    incomplete = per_profile_seed.filter(
        (pl.col("n_algorithms") != len(TABLE_ALGORITHM_ORDER))
        | (pl.col("n_rows") != len(TABLE_ALGORITHM_ORDER))
    )
    if not incomplete.is_empty():
        raise ValueError(
            "Every profile/seed block must contain exactly one run for each algorithm."
        )

    by_profile = per_profile_seed.group_by("profile").agg(
        pl.col("final").mean(),
        (1.96 * pl.col("final").std() / pl.len().sqrt()).alias("final_ci"),
    )
    overall = (
        per_profile_seed.group_by("seed")
        .agg(pl.col("final").mean())
        .select(
            pl.lit("average").alias("profile"),
            pl.col("final").mean(),
            (1.96 * pl.col("final").std() / pl.len().sqrt()).alias("final_ci"),
        )
    )
    return pl.concat([by_profile, overall])


def format_mean_ci(mean: float, ci: float) -> str:
    """Format a table estimate with a smaller confidence interval.

    @ai-generated
    """
    return rf"${mean:.3f}$ $\pm {ci:.3f}$"


def format_algorithm_profile_table(
    summary: pl.DataFrame,
    averages: pl.DataFrame,
    algorithm_averages: pl.DataFrame,
) -> str:
    """Render final test exit rates and seed-level confidence intervals as LaTeX.

    The table retains the manuscript's algorithm and profile ordering. Its intervals
    summarize variation across training seeds on each fixed test pool; the final column
    averages profiles within each seed before computing its interval. The last row averages
    algorithms within each seed. These intervals are distinct from the layout-bootstrap
    intervals used for profile-pool contrasts.

    @ai-generated
    """
    header = " & ".join(TABLE_PROFILE_LABELS[p] for p in TABLE_PROFILE_ORDER)
    lines = [
        r"\begin{tabular}{lcccccc}",
        r"    \toprule",
        f"    \\textbf{{Algorithm}} & {header} & Average " + r"\\",
        r"    \midrule",
    ]
    for algorithm in TABLE_ALGORITHM_ORDER:
        cells: list[str] = []
        for profile in TABLE_PROFILE_ORDER:
            cell = summary.filter(
                (pl.col("algorithm") == algorithm) & (pl.col("profile") == profile)
            )
            if cell.height != 1:
                raise ValueError(
                    f"Expected one summary row for {algorithm}/{profile}, "
                    f"found {cell.height}."
                )
            mean = float(cell.item(0, "final"))
            ci = float(cell.item(0, "final_ci"))
            cells.append(format_mean_ci(mean, ci))
        average = averages.filter(pl.col("algorithm") == algorithm)
        if average.height != 1:
            raise ValueError(
                f"Expected one average summary row for {algorithm}, "
                f"found {average.height}."
            )
        mean = float(average.item(0, "final"))
        ci = float(average.item(0, "final_ci"))
        cells.append(format_mean_ci(mean, ci))
        lines.append(
            f"    {ALGORITHM_LABELS[algorithm]} & " + " & ".join(cells) + r" \\"
        )

    average_cells: list[str] = []
    for profile in (*TABLE_PROFILE_ORDER, "average"):
        cell = algorithm_averages.filter(pl.col("profile") == profile)
        if cell.height != 1:
            raise ValueError(
                f"Expected one algorithm-average row for {profile}, found {cell.height}."
            )
        average_cells.append(
            format_mean_ci(
                float(cell.item(0, "final")), float(cell.item(0, "final_ci"))
            )
        )
    lines.extend(
        [
            r"    \midrule",
            "    Average & " + " & ".join(average_cells) + r" \\",
            r"    \bottomrule",
            r"\end{tabular}",
        ]
    )
    return "\n".join(lines) + "\n"


def laser_colour_analysis(
    log_root: Path,
    layout_data: Path,
    curves: pl.DataFrame,
) -> pl.DataFrame:
    """Ask whether an agent escapes more often when its own colour has a laser in the layout.

    Every layout carries exactly two lasers of two distinct colours, so one agent colour
    is always unmatched. The episode order is deterministic (``sequential = True``), so
    episode *k* of a checkpoint replays layout ``offset + k``. Because that alignment
    cannot be verified from the logs alone, each pool is also scored under the opposite
    offset as a robustness check.

    @ai-generated
    """
    available = curves.select("profile", "algorithm", "pool", "seed").unique()
    rows: list[dict[str, object]] = []
    for profile, algorithm, directory in discover_experiments(log_root):
        colours_path = layout_data / f"{LAYOUT_FAMILIES[profile]}-laser-colours.csv"
        if not colours_path.exists():
            continue
        colours = pl.read_csv(colours_path)
        for pool, filename in POOL_FILES.items():
            selection = available.filter(
                (pl.col("profile") == profile)
                & (pl.col("algorithm") == algorithm)
                & (pl.col("pool") == pool)
            )
            if selection.is_empty():
                continue
            seeds = set(selection["seed"].to_list())
            episodes: list[pl.DataFrame] = []
            for run in sorted(directory.glob("run-*")):
                if int(run.name.removeprefix("run-")) not in seeds:
                    continue
                try:
                    frame = pl.read_csv(
                        run / filename,
                        columns=["time_step", *AGENT_EXIT_COLUMNS],
                    )
                except pl.exceptions.PolarsError, OSError:
                    continue
                episodes.append(
                    frame.filter(pl.col("time_step") == FINAL_STEP)
                    .with_row_index("episode")
                    .drop("time_step")
                )
            if not episodes:
                continue
            pooled = pl.concat(episodes)
            for assumed_pool, offset in POOL_OFFSETS.items():
                joined = pooled.with_columns(
                    (pl.col("episode") + offset).alias("index")
                ).join(colours, on="index", how="inner")
                for agent in range(N_AGENTS):
                    matched = joined.filter(pl.col(f"laser-{agent}") == 1)
                    unmatched = joined.filter(pl.col(f"laser-{agent}") == 0)
                    rows.append(
                        {
                            "profile": profile,
                            "algorithm": algorithm,
                            "pool": pool,
                            "offset_assumption": assumed_pool,
                            "agent": agent,
                            "exit_when_colour_matched": float(
                                matched[f"agent-{agent}-exited"].mean() or 0.0
                            ),
                            "exit_when_colour_absent": float(
                                unmatched[f"agent-{agent}-exited"].mean() or 0.0
                            ),
                            "n_matched": matched.height,
                            "n_absent": unmatched.height,
                        }
                    )
    if not rows:
        return pl.DataFrame()
    return pl.DataFrame(rows).with_columns(
        (pl.col("exit_when_colour_matched") - pl.col("exit_when_colour_absent")).alias(
            "difference"
        )
    )


def permutation_p_value(
    a: np.ndarray, b: np.ndarray, rng: np.random.Generator
) -> float:
    """Two-sided permutation test on the difference of means.

    Exit rates over seeds are bounded and often bimodal, so a distribution-free test is
    preferable to a t-test here.

    @ai-generated
    """
    observed = abs(float(a.mean() - b.mean()))
    pooled = np.concatenate([a, b])
    n = a.size
    count = 0
    for _ in range(N_PERMUTATIONS):
        rng.shuffle(pooled)
        if abs(float(pooled[:n].mean() - pooled[n:].mean())) >= observed - 1e-12:
            count += 1
    return (count + 1) / (N_PERMUTATIONS + 1)


def paired_permutation_p_value(
    differences: np.ndarray, rng: np.random.Generator
) -> float:
    """Sign-flip permutation test that a paired difference has zero mean.

    @ai-generated
    """
    observed = abs(float(differences.mean()))
    count = 0
    for _ in range(N_PERMUTATIONS):
        signs = rng.choice([-1.0, 1.0], size=differences.size)
        if abs(float((differences * signs).mean())) >= observed - 1e-12:
            count += 1
    return (count + 1) / (N_PERMUTATIONS + 1)


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """Cliff's delta, the rank-based probability that a run of ``a`` beats one of ``b``.

    @ai-generated
    """
    return float(np.sign(a[:, None] - b[None, :]).mean())


def holm_correct(tests: list[dict[str, object]]) -> list[dict[str, object]]:
    """Add a Holm-Bonferroni adjusted p-value to a family of tests, in place.

    @ai-generated
    """
    order = sorted(range(len(tests)), key=lambda i: float(tests[i]["p_value"]))  # type: ignore[arg-type]
    running = 0.0
    for rank, index in enumerate(order):
        raw = float(tests[index]["p_value"])  # type: ignore[arg-type]
        running = max(running, min(1.0, (len(tests) - rank) * raw))
        tests[index]["p_value_holm"] = running
    return tests


def learning_gain_tests(
    metrics: pl.DataFrame, rng: np.random.Generator
) -> pl.DataFrame:
    """Test, per configuration, whether training improved on the untrained policy.

    The step-0 checkpoint evaluates the randomly initialised network on the same
    layouts, so it is a matched per-seed baseline rather than an external one.

    @ai-generated
    """
    tests: list[dict[str, object]] = []
    for (profile, algorithm), panel in metrics.group_by(["profile", "algorithm"]):
        differences = panel["gain"].to_numpy()
        tests.append(
            {
                "profile": profile,
                "algorithm": algorithm,
                "baseline": float(panel["baseline"].mean()),
                "final": float(panel["final"].mean()),
                "gain": float(differences.mean()),
                "frac_seeds_improved": float((differences > 0).mean()),
                "p_value": paired_permutation_p_value(differences, rng),
            }
        )
    return pl.DataFrame(holm_correct(tests)).sort(
        "profile", "gain", descending=[False, True]
    )


def layout_block_bootstrap(
    scores: pl.DataFrame,
    profiles: list[str],
    algorithms: list[str],
    rng: np.random.Generator,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Bootstrap profile means by resampling test layouts as shared blocks.

    Each layout is replayed by every seed and algorithm in a profile. We first average
    seeds within algorithm and then algorithms, producing one value per profile-layout.
    A replicate resamples those layout values independently within each profile. This
    estimates test-layout uncertainty conditional on the fitted policies and their one
    training pool; it cannot estimate variation from regenerating that pool and retraining.

    @ai-generated
    """
    selected = scores.filter(
        pl.col("profile").is_in(profiles) & pl.col("algorithm").is_in(algorithms)
    )
    by_layout = (
        selected.group_by("profile", "algorithm", "layout_index")
        .agg(pl.col("exit_rate").mean().alias("seed_mean"))
        .group_by("profile", "layout_index")
        .agg(
            pl.col("seed_mean").mean().alias("exit_rate"),
            pl.col("algorithm").n_unique().alias("n_algorithms"),
        )
        .filter(pl.col("n_algorithms") == len(algorithms))
        .sort("profile", "layout_index")
    )

    bootstrap_means: dict[str, np.ndarray] = {}
    observed_means: dict[str, float] = {}
    summaries: list[dict[str, object]] = []
    for profile in profiles:
        values = by_layout.filter(pl.col("profile") == profile)["exit_rate"].to_numpy()
        if values.size != EVAL_EPISODES:
            continue
        replicates = np.empty(N_BOOTSTRAPS, dtype=float)
        batch_size = 1_000
        for start in range(0, N_BOOTSTRAPS, batch_size):
            stop = min(start + batch_size, N_BOOTSTRAPS)
            indices = rng.integers(0, values.size, size=(stop - start, values.size))
            replicates[start:stop] = values[indices].mean(axis=1)
        bootstrap_means[profile] = replicates
        observed_means[profile] = float(values.mean())
        low, high = np.quantile(replicates, [0.025, 0.975])
        summaries.append(
            {
                "profile": profile,
                "mean": observed_means[profile],
                "ci_low": float(low),
                "ci_high": float(high),
                "n_layouts": int(values.size),
                "n_algorithms": len(algorithms),
                "n_bootstraps": N_BOOTSTRAPS,
            }
        )

    comparisons: list[dict[str, object]] = []
    available = [profile for profile in profiles if profile in bootstrap_means]
    for i, left in enumerate(available):
        for right in available[i + 1 :]:
            differences = bootstrap_means[left] - bootstrap_means[right]
            low, high = np.quantile(differences, [0.025, 0.975])
            comparisons.append(
                {
                    "profile_a": left,
                    "profile_b": right,
                    "difference": observed_means[left] - observed_means[right],
                    "ci_low": float(low),
                    "ci_high": float(high),
                    "probability_a_greater": float((differences > 0).mean()),
                    "n_layouts_a": EVAL_EPISODES,
                    "n_layouts_b": EVAL_EPISODES,
                    "n_algorithms": len(algorithms),
                    "n_bootstraps": N_BOOTSTRAPS,
                }
            )
    return pl.DataFrame(summaries), pl.DataFrame(comparisons)


def layout_curve_block_bootstrap(
    scores: pl.DataFrame,
    profiles: list[str],
    algorithms: list[str],
    rng: np.random.Generator,
) -> pl.DataFrame:
    """Bootstrap the complete test curve by resampling shared layout blocks.

    A profile uses one sampled list of 500 layout indices for all checkpoints. Because
    scores have already been averaged over seeds within algorithm, averaging algorithms
    leaves one checkpoint curve per layout. Pointwise percentile intervals therefore
    measure sensitivity to the test-layout sample while preserving repeated evaluation
    of each layout throughout training.

    @ai-generated
    """
    selected = scores.filter(
        pl.col("profile").is_in(profiles) & pl.col("algorithm").is_in(algorithms)
    )
    by_layout = (
        selected.group_by("profile", "time_step", "layout_index")
        .agg(
            pl.col("exit_rate").mean().alias("exit_rate"),
            pl.col("algorithm").n_unique().alias("n_algorithms"),
        )
        .filter(pl.col("n_algorithms") == len(algorithms))
    )

    rows: list[dict[str, object]] = []
    for profile in profiles:
        panel = by_layout.filter(pl.col("profile") == profile)
        steps = panel["time_step"].unique().sort().to_list()
        if len(steps) != EXPECTED_CHECKPOINTS:
            continue
        values = np.column_stack(
            [
                panel.filter(pl.col("time_step") == step)
                .sort("layout_index")["exit_rate"]
                .to_numpy()
                for step in steps
            ]
        )
        if values.shape != (EVAL_EPISODES, EXPECTED_CHECKPOINTS):
            continue
        replicates = np.empty((N_BOOTSTRAPS, EXPECTED_CHECKPOINTS), dtype=float)
        batch_size = 1_000
        for start in range(0, N_BOOTSTRAPS, batch_size):
            stop = min(start + batch_size, N_BOOTSTRAPS)
            indices = rng.integers(0, EVAL_EPISODES, size=(stop - start, EVAL_EPISODES))
            replicates[start:stop] = values[indices].mean(axis=1)
        low, high = np.quantile(replicates, [0.025, 0.975], axis=0)
        means = values.mean(axis=0)
        rows.extend(
            {
                "profile": profile,
                "time_step": int(step),
                "mean": float(mean),
                "ci_low": float(lower),
                "ci_high": float(upper),
                "n_layouts": EVAL_EPISODES,
                "n_algorithms": len(algorithms),
                "n_bootstraps": N_BOOTSTRAPS,
            }
            for step, mean, lower, upper in zip(steps, means, low, high, strict=True)
        )
    return pl.DataFrame(rows).sort("profile", "time_step")


def compare_profiles(
    metrics: pl.DataFrame,
    profiles: list[str],
    rng: np.random.Generator,
    metric: str = "final",
) -> pl.DataFrame:
    """Test every pair of cooperation profiles on the pooled per-run scores.

    Every element of ``a`` and ``b`` is one run, i.e. the mean over the 500 evaluation
    episodes of one seed at one checkpoint. This pools the algorithms rather than
    blocking on them, so it is reported only as a reference against the blocked test of
    :func:`compare_profiles_blocked`, which is the one to quote.

    @ai-generated
    """
    tests: list[dict[str, object]] = []
    for i, left in enumerate(profiles):
        for right in profiles[i + 1 :]:
            a = metrics.filter(pl.col("profile") == left)[metric].to_numpy()
            b = metrics.filter(pl.col("profile") == right)[metric].to_numpy()
            if a.size == 0 or b.size == 0:
                continue
            tests.append(
                {
                    "profile_a": left,
                    "profile_b": right,
                    "mean_a": float(a.mean()),
                    "mean_b": float(b.mean()),
                    "difference": float(a.mean() - b.mean()),
                    "cliffs_delta": cliffs_delta(a, b),
                    "p_value": permutation_p_value(a, b, rng),
                }
            )
    if not tests:
        return pl.DataFrame()
    return pl.DataFrame(holm_correct(tests))


def stratified_permutation_p_value(
    strata: list[tuple[np.ndarray, np.ndarray]], rng: np.random.Generator
) -> float:
    """Permutation test on a profile difference that permutes within each algorithm.

    The design crosses profile with algorithm, and nearly all of the variance among the
    runs of one profile is between algorithms rather than between seeds. Permuting the
    profile label freely across that mixture tests a null no one doubts and wastes the
    blocking; reassigning the label only among the runs of the *same* algorithm keeps
    each comparison within its block.

    @ai-generated
    """

    def statistic(pairs: list[tuple[np.ndarray, np.ndarray]]) -> float:
        return float(np.mean([a.mean() - b.mean() for a, b in pairs]))

    observed = abs(statistic(strata))
    sizes = [(a.size, a.size + b.size) for a, b in strata]
    pooled = [np.concatenate([a, b]) for a, b in strata]
    count = 0
    for _ in range(N_PERMUTATIONS):
        shuffled: list[tuple[np.ndarray, np.ndarray]] = []
        for values, (n_a, _) in zip(pooled, sizes, strict=True):
            rng.shuffle(values)
            shuffled.append((values[:n_a], values[n_a:]))
        if abs(statistic(shuffled)) >= observed - 1e-12:
            count += 1
    return (count + 1) / (N_PERMUTATIONS + 1)


def compare_profiles_blocked(
    metrics: pl.DataFrame,
    profiles: list[str],
    rng: np.random.Generator,
    metric: str = "final",
) -> pl.DataFrame:
    """Compare cooperation profiles pairwise, blocking on the algorithm.

    The reported difference and Cliff's delta are averaged over the per-algorithm
    contrasts, so both describe the profile effect a single algorithm experiences rather
    than the effect diluted by the spread between algorithms.

    @ai-generated
    """
    algorithms = [a for a in ALGORITHMS if a in set(metrics["algorithm"])]
    tests: list[dict[str, object]] = []
    for i, left in enumerate(profiles):
        for right in profiles[i + 1 :]:
            strata: list[tuple[np.ndarray, np.ndarray]] = []
            deltas: list[float] = []
            for algorithm in algorithms:
                panel = metrics.filter(pl.col("algorithm") == algorithm)
                a = panel.filter(pl.col("profile") == left)[metric].to_numpy()
                b = panel.filter(pl.col("profile") == right)[metric].to_numpy()
                if a.size == 0 or b.size == 0:
                    continue
                strata.append((a, b))
                deltas.append(cliffs_delta(a, b))
            if not strata:
                continue
            tests.append(
                {
                    "profile_a": left,
                    "profile_b": right,
                    "n_blocks": len(strata),
                    "difference": float(
                        np.mean([a.mean() - b.mean() for a, b in strata])
                    ),
                    "cliffs_delta": float(np.mean(deltas)),
                    "min_block_delta": float(np.min(deltas)),
                    "max_block_delta": float(np.max(deltas)),
                    "p_value": stratified_permutation_p_value(strata, rng),
                }
            )
    if not tests:
        return pl.DataFrame()
    return pl.DataFrame(holm_correct(tests))


def blocked_profile_summary(
    metrics: pl.DataFrame, metric: str = "final"
) -> pl.DataFrame:
    """Summarise each profile as the mean of its per-algorithm cell means.

    Averaging the runs of a profile directly gives the right point estimate under the
    balanced design, but a standard error computed over that mixture is dominated by the
    spread between algorithms, which is common to every profile and cancels in any
    comparison. Combining the independent cell means instead reports the precision that
    actually bears on the profile effect.

    @ai-generated
    """
    cells = metrics.group_by("profile", "algorithm").agg(
        pl.col(metric).mean().alias("cell_mean"),
        pl.col(metric).var().alias("cell_var"),
        pl.len().alias("n_runs"),
    )
    return (
        cells.group_by("profile")
        .agg(
            pl.col("cell_mean").mean().alias(metric),
            (pl.col("cell_var") / pl.col("n_runs")).sum().sqrt().alias("sum_se"),
            pl.len().alias("n_algorithms"),
            pl.col("n_runs").sum().alias("n_runs"),
        )
        .with_columns(
            (1.96 * pl.col("sum_se") / pl.col("n_algorithms")).alias(f"{metric}_ci")
        )
        .drop("sum_se")
        .sort(metric, descending=True)
    )


def compare_algorithms_within_profile(
    metrics: pl.DataFrame, rng: np.random.Generator, metric: str = "final"
) -> pl.DataFrame:
    """Compare every pair of algorithms inside each profile, with Holm-corrected p-values.

    @ai-generated
    """
    rows: list[dict[str, object]] = []
    for profile in metrics["profile"].unique().sort():
        panel = metrics.filter(pl.col("profile") == profile)
        algorithms = [a for a in ALGORITHMS if a in set(panel["algorithm"])]
        tests: list[dict[str, object]] = []
        for i, left in enumerate(algorithms):
            for right in algorithms[i + 1 :]:
                a = panel.filter(pl.col("algorithm") == left)[metric].to_numpy()
                b = panel.filter(pl.col("algorithm") == right)[metric].to_numpy()
                tests.append(
                    {
                        "profile": profile,
                        "algorithm_a": left,
                        "algorithm_b": right,
                        "difference": float(a.mean() - b.mean()),
                        "cliffs_delta": cliffs_delta(a, b),
                        "p_value": permutation_p_value(a, b, rng),
                    }
                )
        if tests:
            rows.extend(holm_correct(tests))
    return pl.DataFrame(rows)


def permutation_spread_p_value(
    groups: list[np.ndarray], rng: np.random.Generator
) -> float:
    """Permutation test on the range of group means, the effect size used for interaction.

    @ai-generated
    """
    observed = max(float(g.mean()) for g in groups) - min(
        float(g.mean()) for g in groups
    )
    sizes = [g.size for g in groups]
    pooled = np.concatenate(groups)
    count = 0
    for _ in range(N_PERMUTATIONS):
        rng.shuffle(pooled)
        means: list[float] = []
        start = 0
        for size in sizes:
            means.append(float(pooled[start : start + size].mean()))
            start += size
        if max(means) - min(means) >= observed - 1e-12:
            count += 1
    return (count + 1) / (N_PERMUTATIONS + 1)


def interaction_test(
    metrics: pl.DataFrame, rng: np.random.Generator, metric: str = "final"
) -> pl.DataFrame:
    """Test whether each algorithm's profile-relative advantage varies across profiles.

    @ai-generated
    """
    centred = metrics.with_columns(
        (pl.col(metric) - pl.col(metric).mean().over("profile")).alias("advantage")
    )
    profiles = list(centred["profile"].unique().sort())
    tests: list[dict[str, object]] = []
    for algorithm in ALGORITHMS:
        panel = centred.filter(pl.col("algorithm") == algorithm)
        groups = [
            panel.filter(pl.col("profile") == p)["advantage"].to_numpy()
            for p in profiles
        ]
        groups = [g for g in groups if g.size > 0]
        if len(groups) < 2:
            continue
        tests.append(
            {
                "algorithm": algorithm,
                "spread": float(
                    max(g.mean() for g in groups) - min(g.mean() for g in groups)
                ),
                "p_value": permutation_spread_p_value(groups, rng),
                **{
                    f"advantage_{p}": float(
                        panel.filter(pl.col("profile") == p)["advantage"].mean() or 0.0
                    )
                    for p in profiles
                },
            }
        )
    if not tests:
        return pl.DataFrame()
    return pl.DataFrame(holm_correct(tests))


def plot_curves(summary: pl.DataFrame, output_stem: Path) -> None:
    """Draw one panel per cooperation profile with a curve per algorithm and their average.

    The y axis is scaled to the data rather than to the [0, 1] range of the metric: no
    configuration exceeds an exit rate of about 0.2, so a full-range axis would collapse
    every curve onto the bottom of the panel.

    @ai-generated
    """
    profiles = [p for p in PROFILES if p in set(summary["profile"])]
    if not profiles:
        return
    ceiling = float((summary["mean"] + summary["ci"]).max() or 1.0)
    fig, axes = plt.subplots(
        1,
        len(profiles),
        figsize=(3.1 * len(profiles), 3.0),
        sharey=True,
        squeeze=False,
        constrained_layout=True,
    )
    for axis, profile in zip(axes[0], profiles, strict=True):
        panel = summary.filter(pl.col("profile") == profile)
        for algorithm in ALGORITHMS:
            values = panel.filter(pl.col("algorithm") == algorithm).sort("time_step")
            if values.is_empty():
                continue
            steps = values["time_step"].to_numpy() / 1e6
            mean = values["mean"].to_numpy()
            ci = values["ci"].to_numpy()
            axis.plot(steps, mean, color=ALGORITHM_COLORS[algorithm], linewidth=1.2)
            axis.fill_between(
                steps,
                mean - ci,
                mean + ci,
                color=ALGORITHM_COLORS[algorithm],
                alpha=0.12,
                linewidth=0,
            )
        average = (
            panel.group_by("time_step").agg(pl.col("mean").mean()).sort("time_step")
        )
        axis.plot(
            average["time_step"].to_numpy() / 1e6,
            average["mean"].to_numpy(),
            color=AVERAGE_COLOR,
            linewidth=1.2,
        )
        axis.set_title(rf"{PROFILE_LABELS[profile]}")
        axis.set_xlabel(r"Time steps ($\times 10^6$)")
        axis.set_ylim(0, 1.05 * ceiling)
        axis.set_xlim(0, 1.0)
        axis.grid(True)
    axes[0][0].set_ylabel("Exit rate")
    handles = [
        Line2D([], [], color=ALGORITHM_COLORS[a], label=ALGORITHM_LABELS[a])
        for a in ALGORITHMS
    ]
    handles.append(Line2D([], [], color=AVERAGE_COLOR, label="Average"))
    fig.legend(handles=handles, loc="outside upper center", ncol=len(handles))
    save(fig, output_stem)


def plot_difficulty(
    profile_curve: pl.DataFrame, layout_summary: pl.DataFrame, output_stem: Path
) -> None:
    """Draw layout-bootstrap intervals for profile curves and endpoint means.

    Both panels resample test layouts as shared blocks across the fixed policy ensemble.
    The left panel reuses each sampled layout list across all checkpoints; the right
    panel shows the corresponding final-checkpoint analysis.

    @ai-generated
    """
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), constrained_layout=True)
    for profile in PROFILES:
        values = profile_curve.filter(pl.col("profile") == profile)
        if values.is_empty():
            continue
        steps = values["time_step"].to_numpy() / 1e6
        mean = values["mean"].to_numpy()
        low = values["ci_low"].to_numpy()
        high = values["ci_high"].to_numpy()
        axes[0].plot(
            steps,
            mean,
            color=PROFILE_COLORS[profile],
            linewidth=1.7,
            label=PROFILE_LABELS[profile],
        )
        axes[0].fill_between(
            steps,
            low,
            high,
            color=PROFILE_COLORS[profile],
            alpha=0.15,
            linewidth=0,
        )
    axes[0].set_xlabel(r"Time steps ($\times 10^6$)")
    axes[0].set_ylabel("Exit rate")
    axes[0].set_title("Average test exit rate")
    ceiling = float(profile_curve["ci_high"].max() or 1.0)
    axes[0].set_ylim(0, 1.05 * ceiling)
    axes[0].set_xlim(0, 1.0)
    axes[0].grid(True)
    axes[1].spines["left"].set_visible(False)

    ranking = layout_summary.sort("mean")
    final = ranking["mean"].to_numpy()
    ci = np.vstack(
        [
            final - ranking["ci_low"].to_numpy(),
            ranking["ci_high"].to_numpy() - final,
        ]
    )
    bars = axes[1].barh(
        [PROFILE_LABELS[p] for p in ranking["profile"]],
        final,
        xerr=ci,
        height=0.62,
        color=[PROFILE_COLORS[p] for p in ranking["profile"]],
        edgecolor="black",
        linewidth=0.6,
        error_kw={"ecolor": "black", "capsize": 3, "elinewidth": 0.8},
    )
    for bar, value, upper_error in zip(bars, final, ci[1], strict=True):
        axes[1].text(
            bar.get_width() + upper_error + max(final) * 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.3f}",
            va="center",
            fontsize=8,
        )
    axes[1].set_xlabel("Exit rate at 1M steps")
    axes[1].set_title("Final pool-level performance")
    axes[1].set_xlim(0, 0.175)
    axes[1].grid(True, axis="x")
    axes[1].tick_params(axis="y", length=0)
    present_profiles = [p for p in PROFILES if p in set(profile_curve["profile"])]
    expected_indices = {
        "asymmetric": 0,
        "convergent": 1,
        "sequential": 2,
        "interdependent": 3,
        "divergent": 4,
    }
    present_profiles.sort(key=lambda p: expected_indices[p])
    print(present_profiles)
    handles = [
        Line2D([], [], color=PROFILE_COLORS[p], label=PROFILE_LABELS[p])
        for p in present_profiles
    ]
    fig.legend(
        handles=handles,
        loc="outside upper center",
        ncol=len(handles),
        frameon=True,
    )
    save(fig, output_stem)


def plot_algorithm_heatmap(metrics: pl.DataFrame, output_stem: Path) -> None:
    """Draw the algorithm-by-profile endpoint matrix and the within-profile advantage.

    @ai-generated
    """
    profiles = [p for p in PROFILES if p in set(metrics["profile"])]
    absolute = np.full((len(ALGORITHMS), len(profiles)), np.nan)
    advantage = np.full_like(absolute, np.nan)
    for column, profile in enumerate(profiles):
        panel = metrics.filter(pl.col("profile") == profile)
        profile_mean = float(panel["final"].mean() or 0.0)
        for row, algorithm in enumerate(ALGORITHMS):
            values = panel.filter(pl.col("algorithm") == algorithm)["final"]
            if values.is_empty():
                continue
            absolute[row, column] = float(values.mean() or 0.0)
            advantage[row, column] = absolute[row, column] - profile_mean

    fig, axes = plt.subplots(
        1, 2, figsize=(7.2, 3.4), constrained_layout=True, sharey=True
    )
    fig.get_layout_engine().set(wspace=0.05, w_pad=0.02)
    ceiling = float(np.nanmax(absolute))
    specs = [
        (absolute, "viridis", None, "Exit rate at 1M steps", "Exit rate", "{:.2f}"),
        (
            advantage,
            "RdBu_r",
            float(np.nanmax(np.abs(advantage))),
            "Advantage over profile mean",
            r"$\Delta$ exit rate",
            "{:+.2f}",
        ),
    ]
    for axis, (matrix, cmap_name, limit, title, cbar_label, value_format) in zip(
        axes, specs, strict=True
    ):
        kwargs = (
            {"vmin": -limit, "vmax": limit} if limit else {"vmin": 0.0, "vmax": ceiling}
        )
        cmap = plt.get_cmap(cmap_name).copy()
        cmap.set_bad("0.85")
        image = axis.imshow(
            np.ma.masked_invalid(matrix), cmap=cmap, aspect="equal", **kwargs
        )
        axis.set_xticks(
            range(len(profiles)),
            [PROFILE_LABELS[p] for p in profiles],
            rotation=0,
            ha="center",
        )
        axis.set_yticks(
            range(len(ALGORITHMS)), [ALGORITHM_LABELS[a] for a in ALGORITHMS]
        )
        axis.set_xticks(np.arange(-0.5, len(profiles), 1), minor=True)
        axis.set_yticks(np.arange(-0.5, len(ALGORITHMS), 1), minor=True)
        axis.grid(which="minor", color="white", linewidth=1.2)
        axis.tick_params(which="minor", length=0)
        axis.set_title(title)
        for side in ("top", "right"):
            axis.spines[side].set_visible(True)
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                value = matrix[row, column]
                if np.isnan(value):
                    axis.text(column, row, "--", ha="center", va="center", color="0.4")
                    continue
                red, green, blue, _ = image.cmap(image.norm(value))
                luminance = 0.299 * red + 0.587 * green + 0.114 * blue
                axis.text(
                    column,
                    row,
                    value_format.format(value),
                    ha="center",
                    va="center",
                    fontsize=10,
                    color="white" if luminance < 0.5 else "black",
                )
        fig.colorbar(image, ax=axis, label=cbar_label, fraction=0.046, pad=0.04)
    axes[1].tick_params(axis="y", labelleft=False)
    save(fig, output_stem)


def save(figure: plt.Figure, output_stem: Path) -> None:
    """Write a figure as a PDF.

    @ai-generated
    """
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def usable_cells(metrics: pl.DataFrame, min_cell_runs: int) -> dict[str, set[str]]:
    """Map each profile to the algorithms it has evaluated on enough seeds.

    A cell with a handful of seeds would otherwise skew a pooled profile mean towards
    whichever algorithms happen to be furthest along in the evaluation sweep.

    @ai-generated
    """
    counts = metrics.group_by("profile", "algorithm").agg(pl.len().alias("n"))
    kept = counts.filter(pl.col("n") >= min_cell_runs)
    return {
        profile: set(panel["algorithm"].to_list())
        for (profile,), panel in kept.group_by(["profile"])
    }


def common_grid(
    metrics: pl.DataFrame, profiles: list[str], min_cell_runs: int
) -> tuple[pl.DataFrame, list[str], list[str]]:
    """Restrict the metrics to a complete profile-by-algorithm grid over ``profiles``.

    @ai-generated
    """
    cells = usable_cells(metrics, min_cell_runs)
    profiles = [p for p in profiles if p in cells]
    if len(profiles) < 2:
        return pl.DataFrame(), [], []
    shared = sorted(set.intersection(*(cells[p] for p in profiles)))
    if not shared:
        return pl.DataFrame(), [], []
    subset = metrics.filter(
        pl.col("profile").is_in(profiles) & pl.col("algorithm").is_in(shared)
    )
    return subset, profiles, shared


def widest_pair(
    metrics: pl.DataFrame, min_cell_runs: int
) -> tuple[pl.DataFrame, list[str], list[str]]:
    """Return the two profiles sharing the most evaluated algorithms, and their grid.

    The full sweep is incomplete, so the widest available comparison is usually a pair
    rather than the whole set of profiles.

    @ai-generated
    """
    cells = usable_cells(metrics, min_cell_runs)
    best: tuple[pl.DataFrame, list[str], list[str]] = (pl.DataFrame(), [], [])
    profiles = sorted(cells)
    for i, left in enumerate(profiles):
        for right in profiles[i + 1 :]:
            subset, pair, shared = common_grid(metrics, [left, right], min_cell_runs)
            if len(shared) > len(best[2]):
                best = (subset, pair, shared)
    return best


CERTIFICATE_IMPLICATIONS = (
    ("interdependent", "chained"),
    ("interdependent", "cooperative"),
    ("chained", "cooperative"),
    ("asymmetric", "cooperative"),
    ("convergent", "cooperative"),
    ("divergent", "cooperative"),
)
FAILURE_ALGORITHM_ORDER = ("qmix", "dqn", "vdn", "ippo", "mappo")


def scan_certificate_episodes(path: Path, endpoint_only: bool) -> pl.LazyFrame:
    """Read the episode columns needed for certificate validation."""
    schema = pl.scan_csv(path).collect_schema().names()
    use_test_num = (
        "test_num" in schema
        and pl.scan_csv(path).select(pl.col("test_num").null_count()).collect().item()
        == 0
    )
    columns = [
        "time_step",
        *(["test_num"] if "test_num" in schema else []),
        *AGENT_EXIT_COLUMNS,
        *AGENT_ALIVE_COLUMNS,
        *(f"{name}-trajectory" for name in PREDICATES),
    ]
    frame = pl.scan_csv(path).select(columns)
    if endpoint_only:
        frame = frame.filter(pl.col("time_step") == FINAL_STEP)
    if use_test_num:
        frame = frame.unique(
            ["time_step", "test_num"], keep="last", maintain_order=True
        )
    return frame.with_columns(
        pl.sum_horizontal(AGENT_EXIT_COLUMNS).alias("n_exited"),
        (N_AGENTS - pl.sum_horizontal(AGENT_ALIVE_COLUMNS)).alias("n_dead"),
    ).rename({f"{name}-trajectory": name for name in PREDICATES})


def certificate_outcome_counts(frame: pl.LazyFrame, profile: str) -> pl.LazyFrame:
    """Count predicate hits in each episode-outcome class."""
    target = PROFILE_PREDICATES[profile]
    return frame.group_by("n_exited").agg(
        pl.len().alias("n_episodes"),
        pl.col("n_dead").sum().alias("n_deaths"),
        *(pl.col(name).sum().alias(f"n_{name}") for name in PREDICATES),
        pl.col(target).sum().alias("n_target"),
    )


def certificate_violation_counts(frame: pl.LazyFrame, profile: str) -> pl.LazyFrame:
    """Count certificate, theorem, and profile-inclusion violations."""
    target = PROFILE_PREDICATES[profile]
    winning = pl.col("n_exited") == N_AGENTS
    theorem = pl.col("cooperative") != (pl.col("asymmetric") | pl.col("chained"))
    aggregations = [
        pl.len().alias("n_episodes"),
        winning.sum().alias("n_winning"),
        (winning & ~pl.col(target).cast(pl.Boolean)).sum().alias("n_certificate"),
        (winning & ~pl.col("cooperative").cast(pl.Boolean))
        .sum()
        .alias("n_winning_not_cooperative"),
        theorem.sum().alias("n_theorem"),
    ]
    aggregations += [
        (pl.col(left).cast(pl.Boolean) & ~pl.col(right).cast(pl.Boolean))
        .sum()
        .alias(f"n_{left}_implies_{right}")
        for left, right in CERTIFICATE_IMPLICATIONS
    ]
    return frame.select(aggregations)


def collect_certificate_data(
    log_root: Path, endpoint_only: bool
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Collect outcome and violation tables from the profile sweep."""
    outcomes: list[pl.DataFrame] = []
    violations: list[pl.DataFrame] = []
    for profile, algorithm, directory in discover_experiments(log_root):
        for run in sorted(directory.glob("run-*")):
            seed = int(run.name.removeprefix("run-"))
            context = {
                "profile": pl.lit(profile),
                "algorithm": pl.lit(algorithm),
                "seed": pl.lit(seed),
            }
            for pool, filename in POOL_FILES.items():
                if not has_data(run / filename):
                    continue
                frame = scan_certificate_episodes(run / filename, endpoint_only)
                tagged = {"pool": pl.lit(pool), **context}
                outcomes.append(
                    certificate_outcome_counts(frame, profile)
                    .collect()
                    .with_columns(**tagged)
                )
                violations.append(
                    certificate_violation_counts(frame, profile)
                    .collect()
                    .with_columns(**tagged)
                )
        print(f"  scanned {profile}/{algorithm}", flush=True)
    return pl.concat(outcomes), pl.concat(violations)


def summarize_failure_episodes(path: Path) -> dict[str, object] | None:
    """Summarize one run's held-out final-checkpoint outcomes."""
    episodes = (
        pl.read_csv(
            path,
            columns=[
                "time_step",
                "test_num",
                "exit_rate",
                "episode_len",
                *AGENT_EXIT_COLUMNS,
                *AGENT_ALIVE_COLUMNS,
            ],
        )
        .filter(pl.col("time_step") == FINAL_STEP)
        .unique("test_num", keep="last", maintain_order=True)
    )
    if episodes.is_empty():
        return None
    episodes = episodes.with_columns(
        pl.sum_horizontal(AGENT_EXIT_COLUMNS).alias("n_exited"),
        (N_AGENTS - pl.sum_horizontal(AGENT_ALIVE_COLUMNS)).alias("n_dead"),
    )
    return episodes.select(
        pl.col("exit_rate").mean(),
        pl.col("n_dead").mean().alias("deaths"),
        (pl.col("n_dead") > 0).mean().alias("death_share"),
        (pl.col("n_exited") == N_AGENTS).sum().alias("joint_count"),
        pl.col("episode_len").mean().alias("length"),
        pl.len().alias("n_episodes"),
    ).row(0, named=True)


def collect_failure_runs(log_root: Path) -> pl.DataFrame:
    """Collect final-checkpoint failure outcomes for the profile sweep."""
    rows: list[dict[str, object]] = []
    for profile, algorithm, directory in discover_experiments(log_root):
        for run in sorted(directory.glob("run-*")):
            path = run / POOL_FILES["test"]
            if not has_data(path):
                continue
            outcomes = summarize_failure_episodes(path)
            if outcomes is not None:
                rows.append(
                    {"profile": profile, "algorithm": algorithm, "seed": run.name}
                    | outcomes
                )
    if not rows:
        raise FileNotFoundError(
            f"No held-out final-checkpoint episodes found under {log_root}"
        )
    return pl.DataFrame(rows)


def aggregate_failures(runs: pl.DataFrame) -> pl.DataFrame:
    """Average failure shares by profile and then equally across profiles."""
    metrics = ("exit_rate", "deaths", "death_share", "length")
    shares = (
        runs.group_by("algorithm", "profile")
        .agg(pl.col(*metrics).mean())
        .group_by("algorithm")
        .agg(pl.col(*metrics).mean())
    )
    counts = runs.group_by("algorithm").agg(
        pl.col("joint_count").sum(), pl.col("n_episodes").sum()
    )
    return shares.join(counts, on="algorithm")


def format_failure_table(summary: pl.DataFrame, *, show_count: bool) -> str:
    """Render the death/timeout/joint-success LaTeX table."""
    total_episodes = int(summary["n_episodes"][0])
    lines = [
        r"    \begin{tabular}{lcccccc}",
        r"        \toprule",
        r"        \multirow{2}{*}{\textbf{Algorithm}} & \multirow{2}{*}{\textbf{Exit rate}} & \multirow{2}{*}{\textbf{Deaths/episode}} & \multicolumn{3}{c}{\textbf{Episode outcome}} & \multirow{2}{*}{\textbf{Length}}\\",
        r"        \cmidrule(lr){4-6}",
        r"        & & & Death & Timeout & Joint success & \\",
        r"        \midrule",
    ]
    for algorithm in FAILURE_ALGORITHM_ORDER:
        row = summary.filter(pl.col("algorithm") == algorithm).row(0, named=True)
        death_pct = round(row["death_share"] * 100)
        joint_count = int(row["joint_count"])
        joint = f"{100 * joint_count / total_episodes:.3f}\\%"
        if show_count:
            joint += f" ({joint_count}/\\num{{{total_episodes}}})"
        lines.append(
            f"        {ALGORITHM_LABELS[algorithm]:<9} & {row['exit_rate']:.3f} "
            f"& {row['deaths']:.2f} & {death_pct}\\% & {100 - death_pct}\\% "
            f"& {joint} & {row['length']:.1f}\\\\"
        )
    lines += [r"        \bottomrule", r"    \end{tabular}"]
    return "\n".join(lines) + "\n"


def collect_laser_layout_scores(log_root: Path, layout_data: Path) -> pl.DataFrame:
    """Average endpoint outcomes by profile, algorithm, layout, and agent."""
    frames: list[pl.DataFrame] = []
    for profile, algorithm, directory in discover_experiments(log_root):
        runs: list[pl.DataFrame] = []
        for run in sorted(directory.glob("run-*")):
            path = run / POOL_FILES["test"]
            if not path.exists():
                continue
            episodes = (
                pl.read_csv(
                    path, columns=["time_step", "test_num", *AGENT_EXIT_COLUMNS]
                )
                .filter(pl.col("time_step") == FINAL_STEP)
                .unique("test_num", keep="last", maintain_order=True)
            )
            if episodes.height != EVAL_EPISODES:
                continue
            for agent, column in enumerate(AGENT_EXIT_COLUMNS):
                runs.append(
                    episodes.select(
                        (pl.col("test_num") + EVAL_EPISODES).alias("layout_index"),
                        pl.lit(agent).alias("agent"),
                        pl.col(column).cast(pl.Float64).alias("exited"),
                    )
                )
        if not runs:
            continue
        scores = (
            pl.concat(runs)
            .group_by("layout_index", "agent")
            .agg(pl.col("exited").mean(), pl.len().alias("n_seeds"))
            .with_columns(
                pl.lit(profile).alias("profile"),
                pl.lit(algorithm).alias("algorithm"),
            )
        )
        colours = pl.read_csv(
            layout_data / f"{LAYOUT_FAMILIES[profile]}-laser-colours.csv"
        ).rename({"index": "layout_index"})
        for agent in range(N_AGENTS):
            colours = colours.with_columns(
                (pl.col(f"laser-{agent}") > 0).alias(f"matched-{agent}")
            )
        frames.append(
            scores.join(colours, on="layout_index", how="left")
            .with_columns(
                pl.when(pl.col("agent") == agent)
                .then(pl.col(f"matched-{agent}"))
                .otherwise(None)
                .alias(f"is-matched-{agent}")
                for agent in range(N_AGENTS)
            )
            .with_columns(
                pl.coalesce(
                    *(f"is-matched-{agent}" for agent in range(N_AGENTS))
                ).alias("matched")
            )
            .select(
                "profile",
                "algorithm",
                "layout_index",
                "agent",
                "n_seeds",
                "matched",
                "exited",
            )
        )
    if not frames:
        raise FileNotFoundError("No complete endpoint evaluations were found.")
    result = pl.concat(frames)
    if set(result["n_seeds"].unique()) != {30}:
        raise ValueError("Every layout score must average exactly 30 seeds.")
    return result


def laser_layout_contrasts(
    scores: pl.DataFrame, group_columns: list[str]
) -> pl.DataFrame:
    keys = list(dict.fromkeys([*group_columns, "profile", "layout_index"]))
    matched = (
        scores.filter("matched")
        .group_by(keys)
        .agg(pl.col("exited").mean().alias("matched_rate"))
    )
    absent = (
        scores.filter(~pl.col("matched"))
        .group_by(keys)
        .agg(pl.col("exited").mean().alias("absent_rate"))
    )
    return matched.join(absent, on=keys).with_columns(
        (pl.col("matched_rate") - pl.col("absent_rate")).alias("difference")
    )


def summarize_laser_bootstrap(
    contrasts: pl.DataFrame,
    group_columns: list[str],
    rng: np.random.Generator,
) -> pl.DataFrame:
    groups = (
        contrasts.partition_by(group_columns, as_dict=True)
        if group_columns
        else {("overall",): contrasts}
    )
    rows: list[dict[str, object]] = []
    for key, group in groups.items():
        key = key if isinstance(key, tuple) else (key,)
        arrays = [
            profile.sort("layout_index")
            .select("matched_rate", "absent_rate", "difference")
            .to_numpy()
            for profile in group.partition_by("profile")
        ]
        samples = np.empty((N_BOOTSTRAPS, 3), dtype=float)
        for index in range(N_BOOTSTRAPS):
            samples[index] = np.mean(
                [
                    values[rng.integers(0, len(values), len(values))].mean(axis=0)
                    for values in arrays
                ],
                axis=0,
            )
        estimate = np.mean([values.mean(axis=0) for values in arrays], axis=0)
        low, high = np.quantile(samples[:, 2], [0.025, 0.975])
        row = dict(zip(group_columns, key, strict=True)) if group_columns else {}
        rows.append(
            row
            | {
                "matched_rate": estimate[0],
                "absent_rate": estimate[1],
                "difference": estimate[2],
                "ci_low": low,
                "ci_high": high,
                "n_layouts": group.select("profile", "layout_index").unique().height,
                "n_bootstraps": N_BOOTSTRAPS,
            }
        )
    return pl.DataFrame(rows)


def laser_permutation_p_value(scores: pl.DataFrame, rng: np.random.Generator) -> float:
    averaged = scores.group_by("profile", "layout_index", "agent").agg(
        pl.col("exited").mean(), pl.col("matched").first()
    )
    strata: list[tuple[np.ndarray, np.ndarray]] = []
    for profile in averaged.partition_by("profile"):
        outcomes = (
            profile.pivot(on="agent", index="layout_index", values="exited")
            .sort("layout_index")
            .select(*(str(agent) for agent in range(N_AGENTS)))
            .to_numpy()
        )
        missing = (
            profile.filter(~pl.col("matched")).sort("layout_index")["agent"].to_numpy()
        )
        strata.append((outcomes, missing))

    def statistic(labels: list[np.ndarray]) -> float:
        differences = []
        for (outcomes, _), missing in zip(strata, labels, strict=True):
            absent = outcomes[np.arange(len(outcomes)), missing]
            matched = (outcomes.sum(axis=1) - absent) / (N_AGENTS - 1)
            differences.append(float(np.mean(matched - absent)))
        return float(np.mean(differences))

    observed = abs(statistic([missing for _, missing in strata]))
    extreme = sum(
        abs(statistic([rng.permutation(missing) for _, missing in strata]))
        >= observed - 1e-12
        for _ in range(N_PERMUTATIONS)
    )
    return (extreme + 1) / (N_PERMUTATIONS + 1)


def summarize_agent_exits(scores: pl.DataFrame) -> pl.DataFrame:
    return scores.group_by("agent").agg(pl.col("exited").mean()).sort("agent")
