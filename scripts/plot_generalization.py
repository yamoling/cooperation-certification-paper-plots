from __future__ import annotations

import argparse
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


def parse_args() -> argparse.Namespace:
    """Parse plot inputs and outputs.

    @ai-generated
    """
    parser = argparse.ArgumentParser(
        description="Plot the final-policy generalization sweep."
    )
    parser.add_argument(
        "--logs", type=Path, default=Path("logs"), help="Experiment log root."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Destination PDF.",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=None,
        help="Destination for aggregated values.",
    )
    parser.add_argument(
        "--include-training-policy",
        action="store_true",
        help="Also plot the old on-policy train.csv endpoint metric on the training layouts.",
    )
    parser.add_argument(
        "--joint",
        action="store_true",
        help="Also plot the joint exit rate (all agents exit) in a second row below the exit rate.",
    )
    parser.add_argument(
        "--avg",
        action="store_true",
        help="Draw the across-algorithm average line. Off by default.",
    )
    return parser.parse_args()


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


def main() -> None:
    """Generate the article figure and its auditable aggregate table.

    @ai-generated
    """
    args = parse_args()
    runs = collect_runs(args.logs, include_training_policy=args.include_training_policy)
    summary = aggregate_runs(runs)
    summary_path = args.summary or Path("data/generalization_summary.csv")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.write_csv(summary_path)

    if args.joint:
        joint_runs = collect_joint_runs(args.logs)
        joint_summary = aggregate_runs(joint_runs)
        joint_summary_path = Path("data/generalization_joint_summary.csv")
        joint_summary_path.parent.mkdir(parents=True, exist_ok=True)
        joint_summary.write_csv(joint_summary_path)
        output = args.output or Path("latex/plots/generalization-joint-5x5.pdf")
        plot_joint(summary, joint_summary, output, show_average=args.avg)
        print(
            f"Wrote {output}, {summary_path} and {joint_summary_path} "
            f"from {runs.height} paired runs"
        )
    else:
        output = args.output or Path("plots/generalization-5x5.pdf")
        plot(
            summary,
            output,
            include_training_policy=args.include_training_policy,
            show_average=args.avg,
        )
        print(f"Wrote {output} and {summary_path} from {runs.height} paired runs")
    counts = summary.group_by("n").len().sort("n")
    print(counts)


if __name__ == "__main__":
    main()
