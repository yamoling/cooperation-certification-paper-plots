from __future__ import annotations

import argparse
from pathlib import Path
from typing import cast

import polars as pl

SETTINGS = ("independent", "cooperative")
ALGORITHMS = ("dqn", "ippo", "mappo", "qmix", "vdn")
POOL_SIZES = (1, 10, 20, 50, 100, 150, 200, 300, 400, 500)
SEEDS = 16
FINAL_STEP = 1_000_000
EVALUATION_FILE = "test-policy-on-test-envs.csv"
PROFILES = (
    "cooperative",
    "asymmetric",
    "chained",
    "convergent",
    "divergent",
    "interdependent",
)
ZERO = 0.0
ONE = 0.5
SUCCESS = 1.0


def parse_args() -> argparse.Namespace:
    """Parse the log root and the destination of the aggregated table.

    @ai-generated
    """
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate the per-episode cooperation predicates recorded during the "
            "final-policy evaluation on the training layouts."
        )
    )
    parser.add_argument(
        "--logs", type=Path, default=Path("logs"), help="Experiment log root."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/trajectory_profiles.csv"),
        help="Destination for the aggregated per-configuration table.",
    )
    return parser.parse_args()


def run_profiles(csv_path: Path) -> dict[str, float | None]:
    """Summarize one run's evaluation episodes by outcome and by cooperation predicate.

    Three rates are reported per predicate: over all episodes, over the episodes in
    which both agents exit (joint success), and over the episodes in which neither
    agent exits (zero exit). Conditioning on the outcome matters because a failed
    episode can avoid cooperation simply by failing early.

    @ai-generated
    """
    data = pl.read_csv(csv_path)
    required = {
        "time_step",
        "exit_rate",
        "agent-0-exited",
        "agent-0-alive",
        "agent-1-alive",
    }
    missing = required.union(
        f"{profile}-trajectory" for profile in PROFILES
    ).difference(data.columns)
    if missing:
        raise ValueError(f"{csv_path} is missing columns: {', '.join(sorted(missing))}")
    final_step = data["time_step"].max()
    if final_step != FINAL_STEP:
        raise ValueError(f"{csv_path} ends at step {final_step}, expected {FINAL_STEP}")

    episodes = data.filter(pl.col("time_step") == final_step)
    zero_exits = episodes.filter(pl.col("exit_rate") == ZERO)
    one_exits = episodes.filter(pl.col("exit_rate") == ONE)
    successes = episodes.filter(pl.col("exit_rate") == SUCCESS)
    # The abandoned agent is whichever of the two did not exit.
    abandoned_alive = (
        pl.when(pl.col("agent-0-exited"))
        .then(pl.col("agent-1-alive"))
        .otherwise(pl.col("agent-0-alive"))
    )
    abandoned_alive_mean = one_exits.select(abandoned_alive).to_series().mean()
    summary = {
        "episodes": float(episodes.height),
        "zero_exits": float(zero_exits.height),
        "one_exits": float(one_exits.height),
        "successes": float(successes.height),
        "joint_success_rate": successes.height / episodes.height,
        "abandoned_alive_rate": (
            abandoned_alive_mean
            if one_exits.height and abandoned_alive_mean is not None
            else None
        ),
    }
    for profile in PROFILES:
        column = f"{profile}-trajectory"
        summary[f"{profile}_all"] = episodes[column].mean() or 0.0
        # Undefined rather than zero when the outcome never occurs for this seed.
        summary[f"{profile}_zero_exit"] = (
            zero_exits[column].mean() or 0.0 if zero_exits.height else None
        )
        summary[f"{profile}_success"] = (
            successes[column].mean() or 0.0 if successes.height else None
        )
    return summary


def collect_runs(log_root: Path) -> pl.DataFrame:
    """Collect the per-run cooperation-predicate summaries of the whole 5x5 sweep.

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
                    evaluation = experiment / f"run-{seed}" / EVALUATION_FILE
                    row = {
                        "setting": setting,
                        "algorithm": algorithm,
                        "pool_size": pool_size,
                        "seed": seed,
                    } | run_profiles(evaluation)
                    rows.append(cast(dict[str, object], row))
    if not rows:
        raise FileNotFoundError(f"No evaluation files found under {log_root}")
    return pl.DataFrame(rows)


def aggregate(runs: pl.DataFrame) -> pl.DataFrame:
    """Average the per-seed rates within each configuration, with 95% confidence intervals.

    Seeds whose policy never completes an episode contribute no success-conditional
    rate, so ``seeds_with_success`` records how many seeds back that column.

    @ai-generated
    """
    metrics = ["joint_success_rate", "abandoned_alive_rate"]
    metrics += [f"{profile}_all" for profile in PROFILES]
    metrics += [f"{profile}_zero_exit" for profile in PROFILES]
    metrics += [f"{profile}_success" for profile in PROFILES]
    aggregations: list[pl.Expr] = [
        pl.len().alias("seeds"),
        pl.col("cooperative_success").is_not_null().sum().alias("seeds_with_success"),
        pl.col("episodes").sum().alias("episodes"),
        pl.col("zero_exits").sum().alias("zero_exits"),
        pl.col("one_exits").sum().alias("one_exits"),
        pl.col("successes").sum().alias("successes"),
    ]
    for metric in metrics:
        aggregations.append(pl.col(metric).mean().alias(metric))
        aggregations.append(
            (1.96 * pl.col(metric).std() / pl.col(metric).count().sqrt()).alias(
                f"{metric}_ci"
            )
        )
    return (
        runs.group_by("setting", "algorithm", "pool_size")
        .agg(aggregations)
        .sort("setting", "algorithm", "pool_size")
    )


def print_report(runs: pl.DataFrame) -> None:
    """Print the headline pooled numbers that back the §5.2 Results paragraph.

    Pooling episodes rather than averaging seeds is deliberate throughout: the
    success-conditional rate on cooperative layouts rests on very few successful
    episodes, so the pooled count is the honest way to report it. All figures are
    computed from the held-out (test-pool) evaluation, i.e. what the manuscript
    calls "test episodes".

    @ai-generated
    """
    unreachable = [
        profile for profile in PROFILES if (runs[f"{profile}_all"].max() or 0.0) == 0.0
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


def main() -> None:
    """Aggregate the cooperation predicates of the 5x5 sweep and write them to disk.

    @ai-generated
    """
    args = parse_args()
    runs = collect_runs(args.logs)
    summary = aggregate(runs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary.write_csv(args.output)
    print(f"Aggregated {runs.height} runs into {args.output}")
    print_report(runs)


if __name__ == "__main__":
    main()
