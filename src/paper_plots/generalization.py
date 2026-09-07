"""Pool-size generalization analysis and presentation."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Literal, cast

import matplotlib.pyplot as plt
import polars as pl
from matplotlib.lines import Line2D

SETTINGS = ("independent", "cooperative")
ALGORITHMS = ("dqn", "ippo", "mappo", "qmix", "vdn")
POOL_SIZES = (1, 10, 20, 50, 100, 150, 200, 300, 400, 500)
FINAL_STEP = 1_000_000
TRAIN_GRANULARITY = 50_000
EVALUATION_TRAIN_FILE = "test-policy-on-train-envs.csv"
EVALUATION_TEST_FILE = "test-policy-on-test-envs.csv"
LineStyle = Literal["-", "--", ":"]

# \textwidth of latex/main.tex is 31pc (see sn-jnl.cls); the figure is
# included at \linewidth, so rendering it at that physical width makes the
# rcParams font sizes below match their final size in the compiled PDF.
TEXT_WIDTH_IN = 31 * 12 / 72.27

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 12,
        "legend.fontsize": 12,
        "legend.title_fontsize": 14,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.axisbelow": True,
        "grid.linewidth": 0.4,
        "grid.alpha": 0.4,
    }
)

ALGORITHM_COLORS = {
    "dqn": "C0",
    "ippo": "C1",
    "mappo": "C2",
    "qmix": "C3",
    "vdn": "C4",
}
AVERAGE_COLOR = "black"

OUTCOME_COLUMNS = {0.0: "none_exited", 0.5: "one_exited", 1.0: "both_exited"}
TRAJECTORY_PROFILES = (
    "cooperative",
    "asymmetric",
    "chained",
    "convergent",
    "divergent",
    "interdependent",
)

OUTCOME_ALGORITHM_ORDER = ("dqn", "vdn", "qmix", "ippo", "mappo")
ALGORITHM_LABELS = {
    "dqn": "DQN",
    "vdn": "VDN",
    "qmix": "QMIX",
    "ippo": "IPPO",
    "mappo": "MAPPO",
}
SETTING_LABELS = {"independent": "Independent", "cooperative": "Cooperative"}


def final_mean(csv_path: Path) -> float:
    """Read a CSV with Polars and return its exit-rate mean at the final logged step.

    @ai-generated
    """
    data = pl.read_csv(csv_path)
    required = {"time_step", "exit_rate"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"{csv_path} is missing columns: {', '.join(sorted(missing))}")
    final_step = data["time_step"].max()
    if final_step != FINAL_STEP:
        raise ValueError(f"{csv_path} ends at step {final_step}, expected {FINAL_STEP}")
    value = data.filter(pl.col("time_step") == final_step)["exit_rate"].mean()
    if value is None:
        raise ValueError(f"{csv_path} has no exit-rate values at step {FINAL_STEP}")
    return float(cast(int | float, value))


def joint_final_mean(csv_path: Path) -> float:
    """Read a CSV and return the fraction of final-step episodes where all agents exited.

    `exit_rate` is the fraction of agents that exited an episode, so a value of 1.0 marks a
    joint exit; this differs from `final_mean`, which averages that per-agent fraction instead.

    @ai-generated
    """
    data = pl.read_csv(csv_path)
    required = {"time_step", "exit_rate"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"{csv_path} is missing columns: {', '.join(sorted(missing))}")
    final_step = data["time_step"].max()
    if final_step != FINAL_STEP:
        raise ValueError(f"{csv_path} ends at step {final_step}, expected {FINAL_STEP}")
    episodes = data.filter(pl.col("time_step") == final_step)
    if episodes.height == 0:
        raise ValueError(f"{csv_path} has no exit-rate values at step {FINAL_STEP}")
    return float((episodes["exit_rate"] == 1.0).mean())


def late_training_mean(csv_path: Path) -> float | None:
    """Reproduce the old 50k-binned late-training metric, if its endpoint bin exists.

    @ai-generated
    """
    data = pl.read_csv(csv_path)
    endpoint = data.filter(
        ((pl.col("time_step") / TRAIN_GRANULARITY).round(0) * TRAIN_GRANULARITY).cast(
            pl.Int64
        )
        == FINAL_STEP
    )
    value = endpoint["exit_rate"].mean()
    if value is None:
        return None
    return float(cast(int | float, value))


def _iter_paired_runs(log_root: Path) -> Iterator[tuple[str, str, int, int, Path]]:
    """Yield (setting, algorithm, pool_size, seed, run_dir) for runs with both final-policy files.

    @ai-generated
    """
    for setting in SETTINGS:
        for algorithm in ALGORITHMS:
            for pool_size in POOL_SIZES:
                experiment = (
                    log_root / f"5x5_2agents_1laser-{setting}-{algorithm}-{pool_size}"
                )
                for seed in range(16):
                    run = experiment / f"run-{seed}"
                    if (run / EVALUATION_TRAIN_FILE).exists() and (
                        run / EVALUATION_TEST_FILE
                    ).exists():
                        yield setting, algorithm, pool_size, seed, run


def collect_runs(log_root: Path, *, include_training_policy: bool) -> pl.DataFrame:
    """Collect paired final-policy evaluations, reading the custom train-pool CSV directly with Polars.

    A run is included only when both final-policy files are available, which preserves pairing for the gap.

    @ai-generated
    """
    rows: list[dict[str, object]] = []
    for setting, algorithm, pool_size, seed, run in _iter_paired_runs(log_root):
        row: dict[str, object] = {
            "setting": setting,
            "algorithm": algorithm,
            "pool_size": pool_size,
            "seed": seed,
            "evaluation_train": final_mean(run / EVALUATION_TRAIN_FILE),
            "evaluation_test": final_mean(run / EVALUATION_TEST_FILE),
        }
        if include_training_policy:
            row["training_train"] = late_training_mean(run / "train.csv")
        rows.append(row)
    if not rows:
        raise FileNotFoundError(f"No paired generalization runs found under {log_root}")
    return pl.DataFrame(rows).with_columns(
        (pl.col("evaluation_train") - pl.col("evaluation_test")).alias("gap")
    )


def collect_joint_runs(log_root: Path) -> pl.DataFrame:
    """Collect paired final-policy joint exit rates (all agents exit) across train/test pools.

    A run is included only when both final-policy files are available, which preserves pairing for the gap.

    @ai-generated
    """
    rows: list[dict[str, object]] = [
        {
            "setting": setting,
            "algorithm": algorithm,
            "pool_size": pool_size,
            "seed": seed,
            "evaluation_train": joint_final_mean(run / EVALUATION_TRAIN_FILE),
            "evaluation_test": joint_final_mean(run / EVALUATION_TEST_FILE),
        }
        for setting, algorithm, pool_size, seed, run in _iter_paired_runs(log_root)
    ]
    if not rows:
        raise FileNotFoundError(f"No paired generalization runs found under {log_root}")
    return pl.DataFrame(rows).with_columns(
        (pl.col("evaluation_train") - pl.col("evaluation_test")).alias("gap")
    )


def aggregate_runs(runs: pl.DataFrame) -> pl.DataFrame:
    """Aggregate run-level means and 95% normal CIs across paired seeds.

    @ai-generated
    """
    metrics = ["evaluation_train", "evaluation_test", "gap"]
    if "training_train" in runs.columns:
        metrics.append("training_train")
    aggregations: list[pl.Expr] = [pl.len().alias("n")]
    for metric in metrics:
        aggregations.extend(
            [
                pl.col(metric).mean().alias(metric),
                (1.96 * pl.col(metric).std() / pl.col(metric).count().sqrt()).alias(
                    f"{metric}_ci"
                ),
            ]
        )
    return (
        runs.group_by("setting", "algorithm", "pool_size")
        .agg(aggregations)
        .sort("setting", "algorithm", "pool_size")
    )


def _draw_panel(
    axis: plt.Axes,
    summary: pl.DataFrame,
    setting: str,
    styles: list[tuple[str, str, LineStyle]],
    *,
    show_average: bool,
) -> None:
    """Draw per-algorithm lines (and optionally their average) for one setting onto an axis.

    @ai-generated
    """
    x = list(POOL_SIZES)
    for algorithm in ALGORITHMS:
        values = summary.filter(
            (pl.col("setting") == setting) & (pl.col("algorithm") == algorithm)
        ).sort("pool_size")
        color = ALGORITHM_COLORS[algorithm]
        for metric, _label, line_style in styles:
            mean = values[metric].to_numpy()
            ci = values[f"{metric}_ci"].to_numpy()
            axis.plot(
                x,
                mean,
                color=color,
                linestyle=line_style,
                marker="o",
                markersize=2,
                linewidth=1,
            )
            axis.fill_between(
                x, mean - ci, mean + ci, color=color, alpha=0.10, linewidth=0
            )
    if show_average:
        setting_values = summary.filter(pl.col("setting") == setting)
        for metric, _label, line_style in styles:
            average = (
                setting_values.group_by("pool_size")
                .agg(pl.col(metric).mean())
                .sort("pool_size")[metric]
                .to_numpy()
            )
            axis.plot(
                x,
                average,
                color=AVERAGE_COLOR,
                linestyle=line_style,
                marker="o",
                markersize=2,
                linewidth=1,
            )
    axis.set_xticks(
        x,
        labels=["" if size == 10 else str(size) for size in POOL_SIZES],
        rotation=30,
    )
    axis.set_ylim(0, 1.03)
    axis.grid(True, axis="y")
    axis.set_xmargin(0.02)


def _legend_handles(
    styles: list[tuple[str, str, LineStyle]], *, show_average: bool
) -> tuple[list[Line2D], list[Line2D]]:
    """Build the figure-level algorithm and evaluation legend handles.

    @ai-generated
    """
    algorithm_handles = [
        Line2D(
            [0],
            [0],
            color=ALGORITHM_COLORS[algorithm],
            linewidth=2,
            label=algorithm.upper(),
        )
        for algorithm in ALGORITHMS
    ]
    if show_average:
        algorithm_handles.append(
            Line2D([0], [0], color=AVERAGE_COLOR, linewidth=2, label="Average")
        )
    policy_handles = [
        Line2D([0], [0], color="black", linestyle=line_style, linewidth=2, label=label)
        for _metric, label, line_style in styles
    ]
    return algorithm_handles, policy_handles


def _add_figure_legends(
    fig: plt.Figure, algorithm_handles: list[Line2D], policy_handles: list[Line2D]
) -> None:
    """Attach the algorithm and evaluation legends once, above the whole figure.

    @ai-generated
    """
    algorithm_legend = fig.legend(
        handles=algorithm_handles,
        title="Algorithm",
        loc="outside upper center",
        ncols=len(algorithm_handles),
        columnspacing=0.8,
        handletextpad=0.4,
        bbox_to_anchor=(0.36, 1.09),
    )
    fig.add_artist(algorithm_legend)
    fig.legend(
        handles=policy_handles,
        title="Evaluation",
        loc="outside upper center",
        ncols=len(policy_handles),
        columnspacing=0.8,
        handletextpad=0.4,
        bbox_to_anchor=(0.77, 1.09),
    )


def _save_figure(fig: plt.Figure, output: Path) -> None:
    """Write the figure to disk, creating the destination directory if needed.

    @ai-generated
    """
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def plot(
    summary: pl.DataFrame,
    output: Path,
    *,
    include_training_policy: bool,
    show_average: bool,
) -> None:
    """Plot matched train-layout and evaluation-layout exit rates for the article.

    @ai-generated
    """
    styles: list[tuple[str, str, LineStyle]] = [
        ("evaluation_train", "Train pool", "-"),
        ("evaluation_test", "Test pool", "--"),
    ]
    if include_training_policy:
        styles.append(("training_train", "Training policy on train layouts", ":"))

    fig, axes = plt.subplots(
        1, 2, figsize=(10, 3.5), sharey=True, constrained_layout=True
    )
    for axis, setting in zip(axes, SETTINGS, strict=True):
        _draw_panel(axis, summary, setting, styles, show_average=show_average)
        axis.set_title(f"{setting.capitalize()} layouts")
        axis.set_xlabel("Pool size")
    axes[0].set_ylabel("Exit rate")

    algorithm_handles, policy_handles = _legend_handles(
        styles, show_average=show_average
    )
    _add_figure_legends(fig, algorithm_handles, policy_handles)
    _save_figure(fig, output)


def plot_joint(
    summary_exit: pl.DataFrame,
    summary_joint: pl.DataFrame,
    output: Path,
    *,
    show_average: bool,
) -> None:
    """Plot a 2x2 grid: exit rate on top, joint exit rate (all agents exit) below, by setting.

    @ai-generated
    """
    styles: list[tuple[str, str, LineStyle]] = [
        ("evaluation_train", "Train pool", "--"),
        ("evaluation_test", "Test pool", "-"),
    ]
    rows: list[tuple[pl.DataFrame, str]] = [
        (summary_exit, "Exit rate"),
        (summary_joint, "Joint success rate"),
    ]

    fig, axes = plt.subplots(
        2, 2, figsize=(10, 7), sharex=True, sharey=True, constrained_layout=True
    )
    for row_index, (row_summary, row_ylabel) in enumerate(rows):
        for col_index, setting in enumerate(SETTINGS):
            axis = axes[row_index, col_index]
            _draw_panel(axis, row_summary, setting, styles, show_average=show_average)
            if row_index == 0:
                axis.set_title(f"{setting.capitalize()} layouts")
            if row_index == len(rows) - 1:
                axis.set_xlabel("Pool size")
        axes[row_index, 0].set_ylabel(row_ylabel)

    algorithm_handles, policy_handles = _legend_handles(
        styles, show_average=show_average
    )
    _add_figure_legends(fig, algorithm_handles, policy_handles)
    _save_figure(fig, output)


def episode_outcome_shares(csv_path: Path) -> dict[str, float]:
    """Measure none/single/joint exit shares at the final checkpoint."""
    data = pl.read_csv(csv_path)
    final_step = data["time_step"].max()
    if final_step != FINAL_STEP:
        raise ValueError(f"{csv_path} ends at step {final_step}, expected {FINAL_STEP}")
    episodes = data.filter(pl.col("time_step") == FINAL_STEP)
    unexpected = set(episodes["exit_rate"].unique().to_list()) - set(OUTCOME_COLUMNS)
    if unexpected:
        raise ValueError(
            f"{csv_path} holds unexpected exit rates: {sorted(unexpected)}"
        )
    shares = dict.fromkeys(OUTCOME_COLUMNS.values(), 0.0)
    for outcome, count in episodes["exit_rate"].value_counts().iter_rows():
        shares[OUTCOME_COLUMNS[outcome]] = count / episodes.height
    shares["episodes"] = float(episodes.height)
    shares["exit_rate"] = float(episodes["exit_rate"].mean() or 0.0)
    return shares


def collect_outcome_runs(log_root: Path) -> pl.DataFrame:
    """Collect paired train/test outcome distributions for the 5x5 sweep."""
    rows: list[dict[str, object]] = []
    for setting, algorithm, pool_size, seed, run in _iter_paired_runs(log_root):
        for pool, filename in (
            ("train", EVALUATION_TRAIN_FILE),
            ("test", EVALUATION_TEST_FILE),
        ):
            rows.append(
                {
                    "setting": setting,
                    "algorithm": algorithm,
                    "pool_size": pool_size,
                    "pool": pool,
                    "seed": seed,
                }
                | episode_outcome_shares(run / filename)
            )
    if not rows:
        raise FileNotFoundError(f"No paired generalization runs found under {log_root}")
    return pl.DataFrame(rows)


def aggregate_outcomes(runs: pl.DataFrame) -> pl.DataFrame:
    """Aggregate seed-level exit-outcome shares with normal 95% intervals."""
    metrics = (*OUTCOME_COLUMNS.values(), "exit_rate")
    aggregations: list[pl.Expr] = [pl.len().alias("seeds")]
    for metric in metrics:
        aggregations += [
            pl.col(metric).mean().alias(metric),
            pl.col(metric).std().alias(f"{metric}_std"),
        ]
    summary = runs.group_by("setting", "algorithm", "pool_size", "pool").agg(
        aggregations
    )
    return (
        summary.with_columns(
            (1.96 * pl.col(f"{metric}_std") / pl.col("seeds").sqrt()).alias(
                f"{metric}_ci"
            )
            for metric in metrics
        )
        .drop(f"{metric}_std" for metric in metrics)
        .sort(
            "setting",
            "algorithm",
            "pool_size",
            "pool",
            descending=[False, False, False, True],
        )
    )


def print_exit_summary(summary: pl.DataFrame, pool_size: int) -> None:
    """Print the held-out exit-outcome split at one training-pool size."""
    selection = summary.filter(
        (pl.col("pool") == "test") & (pl.col("pool_size") == pool_size)
    )
    if selection.is_empty():
        raise ValueError(f"No held-out rows at pool size {pool_size}")
    for setting in SETTINGS:
        rows = selection.filter(pl.col("setting") == setting).sort("algorithm")
        print(f"\n{setting} layouts, held-out pool, {pool_size} training layouts")
        print(
            f"{'algorithm':<10}{'exit rate':>12}{'none exited':>14}"
            f"{'one exited':>13}{'both exited':>14}"
        )
        for row in rows.iter_rows(named=True):
            print(
                f"{row['algorithm']:<10}{row['exit_rate']:>12.3f}"
                f"{row['none_exited']:>14.3f}{row['one_exited']:>13.3f}"
                f"{row['both_exited']:>14.3f}"
            )
        means = rows.select(
            pl.col("exit_rate", "none_exited", "one_exited", "both_exited").mean()
        ).row(0)
        print(
            f"{'mean':<10}"
            + "".join(
                f"{value:>{width}.3f}"
                for value, width in zip(means, (12, 14, 13, 14), strict=True)
            )
        )


def format_exit_outcomes_table(selection: pl.DataFrame) -> str:
    """Render the held-out exit-outcome split as a LaTeX table."""
    lines = [
        r"    \begin{tabular}{llcccc}",
        r"        \toprule",
        r"        & & & \multicolumn{3}{c}{Exit outcome} \\",
        r"        \cmidrule(lr){4-6}",
        r"        Layouts & Algorithm & Exit rate & None & Single & Joint \\",
        r"        \midrule",
    ]
    for setting in SETTINGS:
        rows = selection.filter(pl.col("setting") == setting)
        lines.append(
            rf"        \multirow{{{len(OUTCOME_ALGORITHM_ORDER) + 1}}}{{*}}{{{SETTING_LABELS[setting]}}}"
        )
        for algorithm in OUTCOME_ALGORITHM_ORDER:
            row = rows.filter(pl.col("algorithm") == algorithm).row(0, named=True)
            lines.append(
                f"          & {ALGORITHM_LABELS[algorithm]} & {row['exit_rate']:.3f} "
                f"& {row['none_exited']:.3f} & {row['one_exited']:.3f} "
                f"& {row['both_exited']:.3f} \\\\"
            )
        mean = rows.select(
            pl.col("exit_rate", "none_exited", "one_exited", "both_exited").mean()
        ).row(0, named=True)
        lines += [
            r"        \cmidrule(lr){2-6}",
            (
                f"          & Mean & {mean['exit_rate']:.3f} "
                f"& {mean['none_exited']:.3f} & {mean['one_exited']:.3f} "
                f"& {mean['both_exited']:.3f} \\\\"
            ),
        ]
        if setting != SETTINGS[-1]:
            lines.append(r"        \midrule")
    lines += [r"        \bottomrule", r"    \end{tabular}"]
    return "\n".join(lines) + "\n"


def summarize_trajectory_run(csv_path: Path) -> dict[str, float | None]:
    """Summarize final evaluation episodes by outcome and trajectory predicate."""
    data = pl.read_csv(csv_path)
    final_step = data["time_step"].max()
    if final_step != FINAL_STEP:
        raise ValueError(f"{csv_path} ends at step {final_step}, expected {FINAL_STEP}")
    episodes = data.filter(pl.col("time_step") == FINAL_STEP)
    zero_exits = episodes.filter(pl.col("exit_rate") == 0.0)
    one_exits = episodes.filter(pl.col("exit_rate") == 0.5)
    successes = episodes.filter(pl.col("exit_rate") == 1.0)
    abandoned_alive = (
        pl.when(pl.col("agent-0-exited"))
        .then(pl.col("agent-1-alive"))
        .otherwise(pl.col("agent-0-alive"))
    )
    abandoned_mean = one_exits.select(abandoned_alive).to_series().mean()
    result: dict[str, float | None] = {
        "episodes": float(episodes.height),
        "zero_exits": float(zero_exits.height),
        "one_exits": float(one_exits.height),
        "successes": float(successes.height),
        "joint_success_rate": successes.height / episodes.height,
        "abandoned_alive_rate": abandoned_mean if one_exits.height else None,
    }
    for profile in TRAJECTORY_PROFILES:
        column = f"{profile}-trajectory"
        result[f"{profile}_all"] = episodes[column].mean() or 0.0
        result[f"{profile}_zero_exit"] = (
            zero_exits[column].mean() or 0.0 if zero_exits.height else None
        )
        result[f"{profile}_success"] = (
            successes[column].mean() or 0.0 if successes.height else None
        )
    return result


def collect_trajectory_runs(log_root: Path) -> pl.DataFrame:
    """Collect held-out trajectory-predicate summaries for the 5x5 sweep."""
    rows: list[dict[str, object]] = []
    for setting in SETTINGS:
        for algorithm in ALGORITHMS:
            for pool_size in POOL_SIZES:
                experiment = log_root / (
                    f"5x5_2agents_1laser-{setting}-{algorithm}-{pool_size}"
                )
                for seed in range(16):
                    path = experiment / f"run-{seed}" / EVALUATION_TEST_FILE
                    if path.exists():
                        rows.append(
                            {
                                "setting": setting,
                                "algorithm": algorithm,
                                "pool_size": pool_size,
                                "seed": seed,
                            }
                            | summarize_trajectory_run(path)
                        )
    if not rows:
        raise FileNotFoundError(f"No trajectory-profile data found under {log_root}")
    return pl.DataFrame(rows)


def aggregate_trajectory_profiles(runs: pl.DataFrame) -> pl.DataFrame:
    """Aggregate trajectory-predicate rates across seeds."""
    metrics = ["joint_success_rate", "abandoned_alive_rate"]
    metrics += [
        f"{profile}_{suffix}"
        for profile in TRAJECTORY_PROFILES
        for suffix in ("all", "zero_exit", "success")
    ]
    aggregations: list[pl.Expr] = [
        pl.len().alias("seeds"),
        pl.col("cooperative_success").is_not_null().sum().alias("seeds_with_success"),
        *(
            pl.col(name).sum().alias(name)
            for name in ("episodes", "zero_exits", "one_exits", "successes")
        ),
    ]
    for metric in metrics:
        aggregations += [
            pl.col(metric).mean().alias(metric),
            (1.96 * pl.col(metric).std() / pl.col(metric).count().sqrt()).alias(
                f"{metric}_ci"
            ),
        ]
    return (
        runs.group_by("setting", "algorithm", "pool_size")
        .agg(aggregations)
        .sort("setting", "algorithm", "pool_size")
    )


def print_trajectory_report(runs: pl.DataFrame) -> None:
    """Print the headline pooled numbers that back the §5.2 Results paragraph.

    Pooling episodes rather than averaging seeds is deliberate throughout: the
    success-conditional rate on cooperative layouts rests on very few successful
    episodes, so the pooled count is the honest way to report it. All figures are
    computed from the held-out (test-pool) evaluation, i.e. what the manuscript
    calls "test episodes".

    @ai-generated
    """
    unreachable = [
        profile
        for profile in TRAJECTORY_PROFILES
        if (runs[f"{profile}_all"].max() or 0.0) == 0.0
    ]
    print(
        f"Predicates never satisfied in any episode (structurally impossible with two agents and one laser): {', '.join(unreachable) if unreachable else 'none'}"
    )

    pooled = runs.group_by("setting").agg(
        pl.col("episodes").sum(),
        pl.col("zero_exits").sum(),
        pl.col("successes").sum(),
        (pl.col("cooperative_all") * pl.col("episodes")).sum().alias("coop_episodes"),
        (pl.col("cooperative_zero_exit").fill_null(0.0) * pl.col("zero_exits"))
        .sum()
        .alias("coop_zero_exits"),
        (pl.col("cooperative_success").fill_null(0.0) * pl.col("successes"))
        .sum()
        .alias("coop_successes"),
    )
    print(
        f"\n{'layouts':<14}{'episodes':>10}{'successes':>11}{'coop rate':>11}"
        f"{'coop | success':>16}{'coop | 0 exits':>16}"
    )
    for row in pooled.sort("setting", descending=True).iter_rows(named=True):
        print(
            f"{row['setting']:<14}"
            f"{row['episodes']:>10.0f}"
            f"{row['successes']:>11.0f}"
            f"{row['coop_episodes'] / row['episodes']:>11.4f}"
            f"{row['coop_successes'] / row['successes']:>16.4f}"
            f"{row['coop_zero_exits'] / row['zero_exits']:>16.4f}"
        )
        if row["setting"] == "independent":
            print(
                f"  -> {row['successes']:.0f} joint successes on independent layouts, "
                f"{row['coop_successes']:.0f} of them ({row['coop_successes'] / row['successes']:.4%}) "
                "contain a cooperative event"
            )

    print(
        "\ncooperative-event rate vs. pool size (cooperative layouts, all algorithms pooled):"
    )
    by_pool = (
        runs.filter(pl.col("setting") == "cooperative")
        .group_by("pool_size")
        .agg(
            pl.col("episodes").sum(),
            (pl.col("cooperative_all") * pl.col("episodes"))
            .sum()
            .alias("coop_episodes"),
        )
        .with_columns((pl.col("coop_episodes") / pl.col("episodes")).alias("coop_rate"))
        .sort("pool_size")
    )
    for row in by_pool.iter_rows(named=True):
        print(f"  n={row['pool_size']:<4} coop rate = {row['coop_rate']:.4f}")

    print(
        "\nabandoned-agent-alive rate among single-exit episodes, cooperative layouts:"
    )
    by_pool_alive = (
        runs.filter(pl.col("setting") == "cooperative")
        .group_by("pool_size")
        .agg(
            pl.col("one_exits").sum(),
            (pl.col("abandoned_alive_rate").fill_null(0.0) * pl.col("one_exits"))
            .sum()
            .alias("alive_episodes"),
        )
        .with_columns(
            (pl.col("alive_episodes") / pl.col("one_exits")).alias("alive_rate")
        )
        .sort("pool_size")
    )
    for row in by_pool_alive.iter_rows(named=True):
        print(
            f"  n={row['pool_size']:<4} abandoned-alive rate = {row['alive_rate']:.4f}"
        )
