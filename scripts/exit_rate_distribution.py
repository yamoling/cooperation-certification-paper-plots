from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

SETTINGS = ("independent", "cooperative")
ALGORITHMS = ("dqn", "ippo", "mappo", "qmix", "vdn")
POOL_SIZES = (1, 10, 20, 50, 100, 150, 200, 300, 400, 500)
SEEDS = 16
FINAL_STEP = 1_000_000
EVALUATION_TRAIN_FILE = "test-policy-on-train-envs.csv"
EVALUATION_TEST_FILE = "test-policy-on-test-envs.csv"
# With two agents, a per-episode exit rate can only be 0, 0.5 or 1.
OUTCOMES = (0.0, 0.5, 1.0)
OUTCOME_COLUMNS = {0.0: "none_exited", 0.5: "one_exited", 1.0: "both_exited"}


def parse_args() -> argparse.Namespace:
    """Parse the log root and the destination of the aggregated table.

    @ai-generated
    """
    parser = argparse.ArgumentParser(
        description=(
            "Measure how final-policy episodes split between zero, one and two "
            "agents reaching an exit on the 5x5 generalization sweep."
        )
    )
    parser.add_argument(
        "--logs", type=Path, default=Path("logs"), help="Experiment log root."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/exit_rate_distribution.csv"),
        help="Destination for the aggregated per-configuration table.",
    )
    parser.add_argument(
        "--pool-size",
        type=int,
        default=500,
        help="Training-pool size summarized on standard output.",
    )
    return parser.parse_args()


def episode_outcomes(csv_path: Path) -> dict[str, float]:
    """Return the share of final-policy episodes in which zero, one or both agents exited.

    The per-episode ``exit_rate`` column is the fraction of agents that reached an
    exit, so with two agents it is restricted to ``OUTCOMES``. Any other value would
    mean the metric is not what this analysis assumes, hence the explicit check.

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
    unexpected = set(episodes["exit_rate"].unique().to_list()) - set(OUTCOMES)
    if unexpected:
        raise ValueError(
            f"{csv_path} holds unexpected exit rates: {sorted(unexpected)}"
        )

    counts = episodes["exit_rate"].value_counts()
    total = episodes.height
    shares = dict.fromkeys(OUTCOME_COLUMNS.values(), 0.0)
    for outcome, count in counts.iter_rows():
        shares[OUTCOME_COLUMNS[outcome]] = count / total
    shares["episodes"] = float(total)
    shares["exit_rate"] = float(episodes["exit_rate"].mean() or 0.0)
    return shares


def collect_runs(log_root: Path) -> pl.DataFrame:
    """Collect the per-seed outcome distribution of the final policy on both layout pools.

    A run contributes only when both evaluation files exist, which keeps the training
    and held-out rows backed by the same set of seeds.

    @ai-generated
    """
    rows: list[dict[str, object]] = []
    for setting in SETTINGS:
        for algorithm in ALGORITHMS:
            for pool_size in POOL_SIZES:
                experiment = (
                    log_root / f"5x5_2agents_1laser-{setting}-{algorithm}-{pool_size}"
                )
                for seed in range(SEEDS):
                    run = experiment / f"run-{seed}"
                    files = {
                        "train": run / EVALUATION_TRAIN_FILE,
                        "test": run / EVALUATION_TEST_FILE,
                    }
                    if not all(path.exists() for path in files.values()):
                        continue
                    for pool, path in files.items():
                        rows.append(
                            {
                                "setting": setting,
                                "algorithm": algorithm,
                                "pool_size": pool_size,
                                "pool": pool,
                                "seed": seed,
                            }
                            | episode_outcomes(path)
                        )
    if not rows:
        raise FileNotFoundError(f"No paired generalization runs found under {log_root}")
    return pl.DataFrame(rows)


def aggregate(runs: pl.DataFrame) -> pl.DataFrame:
    """Average the per-seed shares within each configuration and attach 95% confidence intervals.

    Aggregating the per-seed shares rather than pooling all episodes keeps the seed
    as the unit of analysis, as in the paired generalization-gap analysis.

    @ai-generated
    """
    metrics = (*OUTCOME_COLUMNS.values(), "exit_rate")
    aggregations = [pl.len().alias("seeds")]
    for metric in metrics:
        aggregations.append(pl.col(metric).mean().alias(metric))
        aggregations.append(pl.col(metric).std().alias(f"{metric}_std"))
    summary = runs.group_by("setting", "algorithm", "pool_size", "pool").agg(
        aggregations
    )
    half_widths = [
        (1.96 * pl.col(f"{metric}_std") / pl.col("seeds").sqrt()).alias(f"{metric}_ci")
        for metric in metrics
    ]
    return (
        summary.with_columns(half_widths)
        .drop([f"{metric}_std" for metric in metrics])
        .sort(
            "setting", "algorithm", "pool_size", "pool", descending=[False] * 3 + [True]
        )
    )


def print_summary(summary: pl.DataFrame, pool_size: int) -> None:
    """Print the held-out outcome split at one training-pool size, per setting and algorithm.

    @ai-generated
    """
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
                f"{row['algorithm']:<10}"
                f"{row['exit_rate']:>12.3f}"
                f"{row['none_exited']:>14.3f}"
                f"{row['one_exited']:>13.3f}"
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


def main() -> None:
    """Aggregate the exit-outcome distribution of the 5x5 sweep and write it to disk.

    @ai-generated
    """
    args = parse_args()
    runs = collect_runs(args.logs)
    summary = aggregate(runs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary.write_csv(args.output)
    print(
        f"Aggregated {runs.height} run evaluations "
        f"({runs.height // 2} paired runs) into {args.output}"
    )
    print_summary(summary, args.pool_size)


if __name__ == "__main__":
    main()
