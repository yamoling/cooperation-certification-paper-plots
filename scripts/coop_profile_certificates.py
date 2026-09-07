"""Episode-level checks that the cooperation-profile sweep could not support before.

Three questions that the aggregated curves cannot answer, all of which need the raw
evaluation episodes rather than per-checkpoint means:

1. **Certificate validation.** Every layout in a pool is certified so that *every*
   winning trajectory requires the target profile. An evaluation episode in which all
   agents exit is a winning trajectory, so its target predicate must hold. A single
   counter-example would refute either the certificate or the predicate detector.
2. **Theorem validation.** The completeness theorem states that a trajectory is
   cooperative if and only if it is asymmetric or sequential. That is an identity on
   every logged episode, and the inclusion structure of the profile family gives further
   implications to check.
3. **Cooperation conditioned on outcome.** Predicate incidence pooled over episodes
   confounds the rare successful episodes with the many failures; splitting by the
   number of agents that exit separates them.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl
from coop_profiles import (
    AGENT_ALIVE_COLUMNS,
    AGENT_EXIT_COLUMNS,
    FINAL_STEP,
    N_AGENTS,
    POOL_FILES,
    PREDICATES,
    PROFILE_PREDICATES,
    discover_experiments,
    has_data,
)

# `cooperative` is the disjunction the completeness theorem characterises; the others
# are the inclusions stated alongside the profile definitions. Each entry is read as
# "left implies right", and `chained` is the logged name of the sequential predicate.
IMPLICATIONS = (
    ("interdependent", "chained"),
    ("interdependent", "cooperative"),
    ("chained", "cooperative"),
    ("asymmetric", "cooperative"),
    ("convergent", "cooperative"),
    ("divergent", "cooperative"),
)


def parse_args() -> argparse.Namespace:
    """Parse the log root and the destination directory.

    @ai-generated
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs", type=Path, default=Path("logs"))
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument(
        "--endpoint-only",
        action="store_true",
        help="Restrict the scan to the final checkpoint instead of all 21.",
    )
    return parser.parse_args()


def scan_episodes(path: Path, endpoint_only: bool) -> pl.LazyFrame:
    """Lazily read the episode columns needed by the three checks.

    The ``test_num`` column is used to dedup restarted logging, but some runs were
    retrofitted with a partially populated ``test_num`` (null on the checkpoints logged
    before the column existed); nulls compare equal under ``unique``, so deduping on such
    a column would collapse 500 genuine episodes per null checkpoint down to one. Only
    dedup when every row has a real ``test_num``.

    @ai-generated
    """
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


def outcome_counts(frame: pl.LazyFrame, profile: str) -> pl.LazyFrame:
    """Count episodes and predicate hits per outcome class, for one run and pool.

    Grouping by ``n_exited`` keeps the winning episodes -- the ones the certificate
    constrains -- separable from the failures they are otherwise swamped by.

    @ai-generated
    """
    target = PROFILE_PREDICATES[profile]
    return frame.group_by("n_exited").agg(
        pl.len().alias("n_episodes"),
        pl.col("n_dead").sum().alias("n_deaths"),
        *(pl.col(name).sum().alias(f"n_{name}") for name in PREDICATES),
        pl.col(target).sum().alias("n_target"),
    )


def violation_counts(frame: pl.LazyFrame, profile: str) -> pl.LazyFrame:
    """Count the episodes that contradict the certificate, the theorem or an inclusion.

    @ai-generated
    """
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
        for left, right in IMPLICATIONS
    ]
    return frame.select(aggregations)


def collect(log_root: Path, endpoint_only: bool) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Scan every run of the sweep and return the outcome and violation tables.

    @ai-generated
    """
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
                frame = scan_episodes(run / filename, endpoint_only)
                tagged = {"pool": pl.lit(pool), **context}
                outcomes.append(
                    outcome_counts(frame, profile).collect().with_columns(**tagged)
                )
                violations.append(
                    violation_counts(frame, profile).collect().with_columns(**tagged)
                )
        print(f"  scanned {profile}/{algorithm}", flush=True)
    return pl.concat(outcomes), pl.concat(violations)


def main() -> None:
    """Write the certificate, theorem and outcome-conditioned predicate tables.

    @ai-generated
    """
    args = parse_args()
    args.data.mkdir(parents=True, exist_ok=True)
    outcomes, violations = collect(args.logs, args.endpoint_only)

    suffix = "-endpoint" if args.endpoint_only else ""
    outcomes.write_csv(args.data / f"episode_outcomes{suffix}.csv")
    violations.write_csv(args.data / f"certificate_violations{suffix}.csv")

    totals = violations.select(pl.exclude("profile", "algorithm", "pool", "seed")).sum()
    print(f"\nTotals over {totals['n_episodes'][0]:,} episodes:")
    for column in totals.columns:
        print(f"  {column}: {totals[column][0]:,}")

    by_profile = (
        outcomes.group_by("profile", "pool", "n_exited")
        .agg(pl.col(pl.Int64).sum(), pl.col(pl.UInt32).sum())
        .sort("profile", "pool", "n_exited")
    )
    by_profile.write_csv(args.data / f"episode_outcomes_by_profile{suffix}.csv")
    print(f"\nWrote tables to {args.data}")


if __name__ == "__main__":
    main()
