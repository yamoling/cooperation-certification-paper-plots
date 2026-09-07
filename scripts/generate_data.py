#!/usr/bin/env python3
"""Generate every intermediate dataset used by the paper's plots and tables.

With no target options this performs the complete data pass.  Select one or
more datasets either positionally or with their ``--<target>`` options::

    uv run python scripts/generate_data.py
    uv run python scripts/generate_data.py generalization sat-duration
    uv run python scripts/generate_data.py --generalization --sat-duration
    uv run python scripts/generate_data.py --list

This entry point deliberately does not render figures or LaTeX.  Rendering is
handled by :mod:`plot_data`, which reads only the files written under ``data``
(apart from the two layout montages, whose inputs are layout files).
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Callable
from pathlib import Path

import numpy as np
import polars as pl

from paper_plots import (
    acceptance,
    availability,
    cooperation,
    generalization,
    layouts,
    sat,
)

CANONICAL_POOLS = (
    "asymmetric",
    "convergent-2",
    "divergent-2",
    "sequential-2",
    "interdependent-2",
)

TARGET_HELP = {
    "coop-profiles": "Cooperation-profile aggregates and statistical tests.",
    "coop-profile-certificates": "Certificate and theorem validation datasets.",
    "coop-profile-failures": "Death, timeout, and joint-success aggregate.",
    "laser-colour": "Laser-colour per-layout data and inferential summaries.",
    "pool-characteristics": "Cooperation-predicate incidence in canonical pools.",
    "generalization": "5x5 pool-size exit-rate and joint-success aggregates.",
    "sat-duration": "Aggregated SAT construction/solving durations.",
    "acceptance-rates": "Certification acceptance-rate and timing summary.",
    "exit-outcomes": "5x5 none/single/joint exit-rate distribution.",
    "trajectory-profiles": "5x5 per-episode cooperation-predicate rates.",
    "acceptance-raw": "Rebuild the raw acceptance-rate experiment (slow).",
    "sat-ablation": "Run the SAT solve-mode measurement sweep (slow).",
    "generate-layouts": "Generate an independent-layout pool.",
    "laser-counts": "Extract laser-colour counts from a layout directory.",
    "availability": "Write the experiment-data availability report.",
}


def ensure_canonical_laser_counts(args: argparse.Namespace) -> None:
    """Create the per-layout laser-count tables required by laser analyses."""
    for family in dict.fromkeys(cooperation.LAYOUT_FAMILIES.values()):
        output = args.data / f"{family}-laser-colours.csv"
        if output.exists() and not args.overwrite:
            continue
        layout_dir = args.layouts / "canonical" / family
        rows = layouts.extract_laser_counts(layout_dir)
        layouts.write_laser_counts(rows, output)
        print(f"Wrote {len(rows)} laser-count rows to {output}")


def generate_cooperation(args: argparse.Namespace) -> None:
    """Extract the cooperation sweep and persist all reusable analysis results."""
    data_dir: Path = args.data
    data_dir.mkdir(parents=True, exist_ok=True)
    curves = cooperation.load_curves(args.logs, data_dir, refresh=args.refresh)
    cooperation.coverage(curves).write_csv(data_dir / "coverage.csv")

    curves, common_seeds = cooperation.restrict_to_common_seeds(curves)
    print(f"Using {len(common_seeds)} common seeds")
    plottable = cooperation.drop_thin_cells(curves, cooperation.MIN_CELL_RUNS)
    test_curves = plottable.filter(pl.col("pool") == "test")
    train_curves = plottable.filter(pl.col("pool") == "train")
    cooperation.aggregate_curve(test_curves).write_csv(data_dir / "test_curves.csv")
    cooperation.aggregate_curve(train_curves).write_csv(data_dir / "train_curves.csv")

    metrics = cooperation.run_level_metrics(curves)
    metrics.write_csv(data_dir / "run_metrics.csv")
    test_metrics = metrics.filter(pl.col("pool") == "test")
    algorithm_profile = cooperation.summarise(test_metrics, "profile", "algorithm")
    algorithm_profile.write_csv(data_dir / "profile_algorithm_summary.csv")
    cooperation.average_final_across_profiles(test_metrics).write_csv(
        data_dir / "algorithm_average_summary.csv"
    )
    cooperation.average_final_across_algorithms(test_metrics).write_csv(
        data_dir / "profile_average_summary.csv"
    )
    cooperation.summarise(test_metrics, "profile").write_csv(
        data_dir / "profile_summary.csv"
    )
    cooperation.summarise(metrics, "profile", "algorithm", "pool").write_csv(
        data_dir / "pool_summary.csv"
    )

    trajectory = cooperation.gap_trajectory(curves)
    if not trajectory.is_empty():
        trajectory.write_csv(data_dir / "gap_trajectory.csv")
        (
            trajectory.group_by("profile", "algorithm", "time_step")
            .agg(
                pl.len().alias("n"),
                pl.col("train_exit_rate").mean(),
                pl.col("test_exit_rate").mean(),
                pl.col("gap").mean(),
                (1.96 * pl.col("gap").std() / pl.len().sqrt()).alias("gap_ci"),
            )
            .sort("profile", "algorithm", "time_step")
            .write_csv(data_dir / "gap_trajectory_summary.csv")
        )

    all_profiles = sorted(test_metrics["profile"].unique().to_list())
    balanced, profiles, shared = cooperation.common_grid(
        test_metrics, all_profiles, cooperation.MIN_CELL_RUNS
    )
    pair_metrics, pair_profiles, pair_shared = cooperation.widest_pair(
        test_metrics, cooperation.MIN_CELL_RUNS
    )
    rng = np.random.default_rng(cooperation.RNG_SEED)
    if not balanced.is_empty():
        layout_curves = cooperation.load_test_layout_curves(
            args.logs,
            data_dir,
            refresh=args.refresh,
            selected_seeds=set(common_seeds),
        )
        layout_curve_summary = cooperation.layout_curve_block_bootstrap(
            layout_curves,
            profiles,
            shared,
            np.random.default_rng(cooperation.RNG_SEED),
        )
        if not layout_curve_summary.is_empty():
            layout_curve_summary.write_csv(data_dir / "layout_bootstrap_curves.csv")

        layout_scores = cooperation.load_final_layout_scores(
            args.logs, data_dir, refresh=args.refresh
        ).filter(pl.col("seed").is_in(common_seeds))
        layout_summary, layout_pairwise = cooperation.layout_block_bootstrap(
            layout_scores, profiles, shared, rng
        )
        if not layout_summary.is_empty():
            layout_summary.write_csv(data_dir / "layout_bootstrap_summary.csv")
        if not layout_pairwise.is_empty():
            layout_pairwise.write_csv(data_dir / "layout_bootstrap_pairwise.csv")

        cooperation.compare_profiles(balanced, profiles, rng).write_csv(
            data_dir / "profile_pairwise_balanced.csv"
        )
        cooperation.compare_profiles(balanced, profiles, rng, metric="gain").write_csv(
            data_dir / "profile_pairwise_gain.csv"
        )
        cooperation.compare_profiles_blocked(balanced, profiles, rng).write_csv(
            data_dir / "profile_pairwise_blocked.csv"
        )
        cooperation.compare_profiles_blocked(
            balanced, profiles, rng, metric="gain"
        ).write_csv(data_dir / "profile_pairwise_blocked_gain.csv")
        cooperation.blocked_profile_summary(balanced).write_csv(
            data_dir / "profile_summary_blocked.csv"
        )
        cooperation.summarise(balanced, "profile").write_csv(
            data_dir / "profile_summary_grid.csv"
        )
        cooperation.interaction_test(balanced, rng).write_csv(
            data_dir / "algorithm_interaction.csv"
        )

    if not pair_metrics.is_empty():
        cooperation.compare_profiles(pair_metrics, pair_profiles, rng).write_csv(
            data_dir / "profile_pairwise_widest.csv"
        )
        cooperation.summarise(pair_metrics, "profile").write_csv(
            data_dir / "profile_summary_widest.csv"
        )
        cooperation.interaction_test(pair_metrics, rng).write_csv(
            data_dir / "algorithm_interaction_widest.csv"
        )

    cooperation.compare_algorithms_within_profile(test_metrics, rng).write_csv(
        data_dir / "algorithm_pairwise.csv"
    )
    cooperation.compare_algorithms_within_profile(
        test_metrics, rng, metric="gain"
    ).write_csv(data_dir / "algorithm_pairwise_gain.csv")
    cooperation.learning_gain_tests(test_metrics, rng).write_csv(
        data_dir / "learning_gain.csv"
    )
    widening = cooperation.gap_widening_tests(trajectory, rng)
    if not widening.is_empty():
        widening.write_csv(data_dir / "gap_widening.csv")
    ensure_canonical_laser_counts(args)
    lasers = cooperation.laser_colour_analysis(args.logs, data_dir, curves)
    if not lasers.is_empty():
        lasers.write_csv(data_dir / "laser_colour_effect.csv")

    print(f"Wrote cooperation-profile datasets to {data_dir}")
    print(f"Common grid: {', '.join(profiles)} x {', '.join(shared)}")
    print(f"Widest pair: {', '.join(pair_profiles)} x {', '.join(pair_shared)}")


def generate_certificates(args: argparse.Namespace) -> None:
    """Generate full-trajectory and endpoint-only certificate datasets."""
    args.data.mkdir(parents=True, exist_ok=True)
    for endpoint_only in (False, True):
        outcomes, violations = cooperation.collect_certificate_data(
            args.logs, endpoint_only
        )
        suffix = "-endpoint" if endpoint_only else ""
        outcomes.write_csv(args.data / f"episode_outcomes{suffix}.csv")
        violations.write_csv(args.data / f"certificate_violations{suffix}.csv")
        (
            outcomes.group_by("profile", "pool", "n_exited")
            .agg(pl.col(pl.Int64).sum(), pl.col(pl.UInt32).sum())
            .sort("profile", "pool", "n_exited")
            .write_csv(args.data / f"episode_outcomes_by_profile{suffix}.csv")
        )
    print(f"Wrote certificate datasets to {args.data}")


def generate_failures(args: argparse.Namespace) -> None:
    runs = cooperation.collect_failure_runs(args.logs)
    summary = cooperation.aggregate_failures(runs)
    args.data.mkdir(parents=True, exist_ok=True)
    summary.write_csv(args.data / "coop_profile_failures_summary.csv")
    print(f"Wrote {args.data / 'coop_profile_failures_summary.csv'}")


def generate_laser_colour(args: argparse.Namespace) -> None:
    args.data.mkdir(parents=True, exist_ok=True)
    ensure_canonical_laser_counts(args)
    scores = cooperation.collect_laser_layout_scores(args.logs, args.data)
    scores.write_csv(args.data / "laser_colour_layout_scores.csv")
    rng = np.random.default_rng(cooperation.RNG_SEED)
    overall = cooperation.summarize_laser_bootstrap(
        cooperation.laser_layout_contrasts(scores, []), [], rng
    ).with_columns(
        pl.lit(cooperation.laser_permutation_p_value(scores, rng)).alias("p_value")
    )
    overall.write_csv(args.data / "laser_colour_summary.csv")
    for columns, filename in (
        (["algorithm"], "laser_colour_by_algorithm.csv"),
        (["profile"], "laser_colour_by_profile.csv"),
        (["profile", "algorithm"], "laser_colour_by_profile_algorithm.csv"),
    ):
        cooperation.summarize_laser_bootstrap(
            cooperation.laser_layout_contrasts(scores, columns), columns, rng
        ).write_csv(args.data / filename)
    cooperation.summarize_agent_exits(scores).write_csv(
        args.data / "agent_exit_summary.csv"
    )
    print(f"Wrote laser-colour datasets to {args.data}")


def generate_pool_characteristics(args: argparse.Namespace) -> None:
    output = args.data / "layout-characteristics.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    for index, pool in enumerate(CANONICAL_POOLS):
        paths = layouts.list_layouts(args.layouts / "canonical" / pool)
        if args.layout_limit > 0:
            paths = paths[: args.layout_limit]
        layouts.characterize_pool(paths, output, args.n_workers, overwrite=index == 0)
    print(f"Wrote {output}")


def generate_generalization(args: argparse.Namespace) -> None:
    args.data.mkdir(parents=True, exist_ok=True)
    runs = generalization.collect_runs(
        args.logs, include_training_policy=args.include_training_policy
    )
    generalization.aggregate_runs(runs).write_csv(
        args.data / "generalization_summary.csv"
    )
    joint_runs = generalization.collect_joint_runs(args.logs)
    generalization.aggregate_runs(joint_runs).write_csv(
        args.data / "generalization_joint_summary.csv"
    )
    print(f"Wrote generalization datasets to {args.data}")


def generate_sat_duration(args: argparse.Namespace) -> None:
    source = args.sat_source or args.data / "sat-measurements.csv"
    if not source.exists():
        print(
            f"Skipped sat-duration: source measurements not found at {source}. "
            "Restore that precomputed horizon-sweep CSV or pass --sat-source."
        )
        return
    summary = sat.summarize_measurements(pl.read_csv(source, infer_schema_length=None))
    summary.write_csv(args.data / "sat_measurements_summary.csv")
    print(f"Wrote {args.data / 'sat_measurements_summary.csv'}")


def generate_acceptance_rates(args: argparse.Namespace) -> None:
    raw_path = args.acceptance_source or args.data / "acceptance-rates.csv"
    negative_path = (
        args.negative_query_source or args.data / "no-profile-x-end-to-end-timings.csv"
    )
    missing = [path for path in (raw_path, negative_path) if not path.exists()]
    if missing:
        print(
            "Skipped acceptance-rates: source data not found: "
            + ", ".join(map(str, missing))
        )
        return
    raw = pl.read_csv(raw_path)
    negative = pl.read_csv(negative_path)
    summary = acceptance.summarize(
        raw, negative, [*acceptance.PROFILES, acceptance.FULLY_COUPLED]
    )
    summary.write_csv(args.data / "acceptance-rates-summary.csv")
    generation_time = acceptance.measure_generation_time(args.generation_samples)
    pl.DataFrame(
        {"generation_time_s": [generation_time], "samples": [args.generation_samples]}
    ).write_csv(args.data / "acceptance-generation-time.csv")
    print(f"Wrote acceptance-rate datasets to {args.data}")


def generate_exit_outcomes(args: argparse.Namespace) -> None:
    runs = generalization.collect_outcome_runs(args.logs)
    summary = generalization.aggregate_outcomes(runs)
    args.data.mkdir(parents=True, exist_ok=True)
    summary.write_csv(args.data / "exit_rate_distribution.csv")
    print(f"Wrote {args.data / 'exit_rate_distribution.csv'}")


def generate_trajectory_profiles(args: argparse.Namespace) -> None:
    runs = generalization.collect_trajectory_runs(args.logs)
    summary = generalization.aggregate_trajectory_profiles(runs)
    args.data.mkdir(parents=True, exist_ok=True)
    runs.write_csv(args.data / "trajectory_profile_runs.csv")
    summary.write_csv(args.data / "trajectory_profiles.csv")
    print(f"Wrote trajectory-profile datasets to {args.data}")


def generate_raw_acceptance(args: argparse.Namespace) -> None:
    acceptance.run_experiment(
        args.n_layouts,
        args.acceptance_workers,
        args.t_max,
        args.seed,
        args.data / "acceptance-rates.csv",
        overwrite=args.overwrite,
    )


def generate_sat_ablation(args: argparse.Namespace) -> None:
    sat.run_measurements(
        args.data / "sat-measurements-mode.csv",
        levels=args.levels,
        modes=tuple(args.modes),
        shuffle=args.shuffle,
        repetitions=args.repetitions,
        overwrite=args.overwrite,
    )


def generate_layout_pool(args: argparse.Namespace) -> None:
    if args.layout_output is None:
        raise ValueError("--layout-output is required for --generate-layouts")
    layouts.generate_independent_layouts(
        args.layout_output,
        t_max=args.t_max,
        n_jobs=args.layout_workers,
        seed=args.seed,
        total=args.layout_count,
    )


def generate_laser_counts(args: argparse.Namespace) -> None:
    if args.layout_source is None:
        raise ValueError("--layout-source is required for --laser-counts")
    output = args.data / f"{args.layout_source.name}-laser-colours.csv"
    rows = layouts.extract_laser_counts(args.layout_source)
    layouts.write_laser_counts(rows, output)
    print(f"Wrote {len(rows)} laser-count rows to {output}")


def generate_availability_report(args: argparse.Namespace) -> None:
    availability.write_report(
        args.logs,
        args.availability_output,
        expected_seeds=args.expected_seeds,
        json_output=args.availability_json,
    )


GENERATORS: dict[str, Callable[[argparse.Namespace], None]] = {
    "coop-profiles": generate_cooperation,
    "coop-profile-certificates": generate_certificates,
    "coop-profile-failures": generate_failures,
    "laser-colour": generate_laser_colour,
    "pool-characteristics": generate_pool_characteristics,
    "generalization": generate_generalization,
    "sat-duration": generate_sat_duration,
    "acceptance-rates": generate_acceptance_rates,
    "exit-outcomes": generate_exit_outcomes,
    "trajectory-profiles": generate_trajectory_profiles,
    "acceptance-raw": generate_raw_acceptance,
    "sat-ablation": generate_sat_ablation,
    "generate-layouts": generate_layout_pool,
    "laser-counts": generate_laser_counts,
    "availability": generate_availability_report,
}

DEFAULT_GENERATORS = (
    "coop-profiles",
    "coop-profile-certificates",
    "coop-profile-failures",
    "laser-colour",
    "pool-characteristics",
    "generalization",
    "sat-duration",
    "acceptance-rates",
    "exit-outcomes",
    "trajectory-profiles",
)


def target_output(name: str, args: argparse.Namespace) -> Path | None:
    """Return the file that marks a target as previously completed."""
    data_outputs = {
        "coop-profiles": "learning_gain.csv",
        "coop-profile-certificates": "episode_outcomes_by_profile-endpoint.csv",
        "coop-profile-failures": "coop_profile_failures_summary.csv",
        "laser-colour": "agent_exit_summary.csv",
        "pool-characteristics": "layout-characteristics.csv",
        "generalization": "generalization_joint_summary.csv",
        "sat-duration": "sat_measurements_summary.csv",
        "acceptance-rates": "acceptance-generation-time.csv",
        "exit-outcomes": "exit_rate_distribution.csv",
        "trajectory-profiles": "trajectory_profiles.csv",
        "acceptance-raw": "acceptance-rates.csv",
        "sat-ablation": "sat-measurements-mode.csv",
    }
    if name in data_outputs:
        return args.data / data_outputs[name]
    if name == "generate-layouts":
        return args.layout_output
    if name == "laser-counts" and args.layout_source is not None:
        return args.data / f"{args.layout_source.name}-laser-colours.csv"
    if name == "availability":
        return args.availability_output
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("targets", nargs="*", choices=GENERATORS, metavar="TARGET")
    parser.add_argument("--list", action="store_true", help="List targets and exit.")
    for name in GENERATORS:
        parser.add_argument(
            f"--{name}",
            dest="target_options",
            action="append_const",
            const=name,
            help=f"Generate only {name} data (repeatable).",
        )
    parser.add_argument("--logs", type=Path, default=Path("logs"))
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--layouts", type=Path, default=Path("layouts"))
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate selected targets even when their output already exists.",
    )
    parser.add_argument("--layout-limit", type=int, default=1000)
    parser.add_argument("--n-workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--acceptance-workers", type=int, default=10)
    parser.add_argument("--layout-workers", type=int, default=1)
    parser.add_argument("--generation-samples", type=int, default=100)
    parser.add_argument("--include-training-policy", action="store_true")
    parser.add_argument("--n-layouts", type=int, default=100_000)
    parser.add_argument("--layout-count", type=int, default=10_000)
    parser.add_argument("--t-max", type=int, default=81)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--levels", type=int, nargs="+", default=sat.DEFAULT_LEVELS)
    parser.add_argument("--modes", nargs="+", default=sat.DEFAULT_MODES)
    parser.add_argument("--repetitions", type=int, default=sat.SAT_REPETITIONS)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--layout-source", type=Path)
    parser.add_argument("--layout-output", type=Path)
    parser.add_argument("--expected-seeds", type=int, default=30)
    parser.add_argument(
        "--availability-output", type=Path, default=Path("data-summary.md")
    )
    parser.add_argument("--availability-json", type=Path)
    parser.add_argument(
        "--sat-source",
        type=Path,
        help="Precomputed SAT horizon-sweep CSV (default: DATA/sat-measurements.csv).",
    )
    parser.add_argument(
        "--acceptance-source",
        type=Path,
        help="Raw acceptance-rate CSV (default: DATA/acceptance-rates.csv).",
    )
    parser.add_argument(
        "--negative-query-source",
        type=Path,
        help="Negative-query timing CSV used by the acceptance summary.",
    )
    args = parser.parse_args()
    args.refresh = args.refresh or args.overwrite
    requested = [*args.targets, *(args.target_options or [])]
    args.selected = list(dict.fromkeys(requested)) or list(DEFAULT_GENERATORS)
    return args


def main() -> None:
    args = parse_args()
    if args.list:
        width = max(map(len, GENERATORS))
        for name in GENERATORS:
            print(f"{name:<{width}}  {TARGET_HELP[name]}")
        return
    for index, name in enumerate(args.selected, start=1):
        print(f"\n=== [{index}/{len(args.selected)}] {name} ===")
        output = target_output(name, args)
        if output is not None and output.exists() and not args.overwrite:
            print(f"Skipped {name}: {output} already exists (use --overwrite).")
            continue
        GENERATORS[name](args)
    print(f"\nDone: {', '.join(args.selected)}")


if __name__ == "__main__":
    main()
