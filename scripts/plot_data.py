#!/usr/bin/env python3
"""Render the paper's figures, tables, and reports from generated data.

With no target options this renders everything.  Select outputs positionally or
with repeatable keyword options::

    uv run python scripts/plot_data.py
    uv run python scripts/plot_data.py generalization sat-duration
    uv run python scripts/plot_data.py --generalization --sat-duration
    uv run python scripts/plot_data.py --coop-gap --coop-heatmap
    uv run python scripts/plot_data.py --list

Run ``scripts/generate_data.py`` first.  This script never scans experiment
logs; the only non-CSV inputs are the layout files used by the two montages.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path

import polars as pl

from paper_plots import acceptance, cooperation, generalization, layouts, sat

TARGET_HELP = {
    "coop-profiles": "All cooperation-profile figures and its LaTeX table.",
    "coop-curves": "Cooperation-profile train/test learning curves.",
    "coop-gap": "Cooperation-profile generalization-gap figure.",
    "coop-difficulty": "Cooperation-profile layout-difficulty figure.",
    "coop-heatmap": "Algorithm-by-profile heatmap.",
    "coop-profile-table": "Algorithm-by-profile LaTeX table.",
    "coop-profile-failures": "Death/timeout/joint-success LaTeX table.",
    "pool-characteristics": "Canonical-pool characteristics LaTeX table.",
    "layouts-picture": "Canonical profile layout montage.",
    "generalization-layouts-picture": "5x5 independent/cooperative montage.",
    "generalization": "5x5 pool-size generalization figure.",
    "sat-duration": "SAT duration figure.",
    "acceptance-rates": "Certification figures and yield LaTeX table.",
    "exit-outcomes": "5x5 exit-outcome LaTeX table.",
    "trajectory-profiles": "Print the trajectory-profile report.",
    "exit-rate-distribution": "Print the exit-outcome distribution report.",
    "trajectory": "Draw a solved or supplied trajectory on one layout.",
}

DEFAULT_TARGETS = (
    "coop-profiles",
    "coop-profile-failures",
    "pool-characteristics",
    "layouts-picture",
    "generalization-layouts-picture",
    "generalization",
    "sat-duration",
    "acceptance-rates",
    "exit-outcomes",
    "trajectory-profiles",
    "exit-rate-distribution",
)


def read_csv(path: Path) -> pl.DataFrame:
    """Read a generated dataset with an actionable error when it is absent."""
    if not path.exists():
        raise FileNotFoundError(
            f"Missing generated dataset: {path}. Run scripts/generate_data.py first."
        )
    return pl.read_csv(path)


def plot_coop_curves(args: argparse.Namespace) -> None:
    cooperation.plot_curves(
        read_csv(args.data / "test_curves.csv"), args.plots / "coop-profiles-test"
    )
    cooperation.plot_curves(
        read_csv(args.data / "train_curves.csv"), args.plots / "coop-profiles-train"
    )


def plot_coop_gap(args: argparse.Namespace) -> None:
    cooperation.plot_gap_trajectory(
        read_csv(args.data / "gap_trajectory.csv"), args.plots / "coop-profiles-gap"
    )


def plot_coop_difficulty(args: argparse.Namespace) -> None:
    cooperation.plot_difficulty(
        read_csv(args.data / "layout_bootstrap_curves.csv"),
        read_csv(args.data / "layout_bootstrap_summary.csv"),
        args.plots / "coop-profiles-difficulty",
    )


def plot_coop_heatmap(args: argparse.Namespace) -> None:
    metrics = read_csv(args.data / "run_metrics.csv").filter(pl.col("pool") == "test")
    cooperation.plot_algorithm_heatmap(
        cooperation.drop_thin_cells(metrics, cooperation.MIN_CELL_RUNS),
        args.plots / "coop-profiles-heatmaps",
    )


def write_coop_table(args: argparse.Namespace) -> None:
    table = cooperation.format_algorithm_profile_table(
        read_csv(args.data / "profile_algorithm_summary.csv"),
        read_csv(args.data / "algorithm_average_summary.csv"),
        read_csv(args.data / "profile_average_summary.csv"),
    )
    output = args.tables / "profile_algorithm_table.tex"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(table)
    print(f"Wrote {output}")


def plot_all_cooperation(args: argparse.Namespace) -> None:
    for action in (
        plot_coop_curves,
        plot_coop_gap,
        plot_coop_difficulty,
        plot_coop_heatmap,
        write_coop_table,
    ):
        action(args)


def write_failure_table(args: argparse.Namespace) -> None:
    summary = read_csv(args.data / "coop_profile_failures_summary.csv")
    table = cooperation.format_failure_table(summary, show_count=args.count)
    output = args.tables / "coop_profile_failures_table.tex"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(table)
    print(f"Wrote {output}")


def write_pool_table(args: argparse.Namespace) -> None:
    summary = layouts.summarize_pools(args.data / "layout-characteristics.csv")
    table = layouts.format_table(summary)
    output = args.tables / "pool_characteristics_table.tex"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(table)
    print(f"Wrote {output}")


def plot_layouts(args: argparse.Namespace) -> None:
    output = args.plots / "layouts.pdf"
    layouts.render_canonical_layouts(
        args.layouts / "canonical",
        output,
        seed=args.layout_seed,
        transpose=args.transpose,
    )
    print(f"Wrote {output}")


def plot_generalization_layouts(args: argparse.Namespace) -> None:
    output = args.plots / "generalization-layouts-5x5.pdf"
    layouts.render_generalization_layouts(
        args.layouts / "tuning", output, seed=args.generalization_layout_seed
    )
    print(f"Wrote {output}")


def plot_generalization(args: argparse.Namespace) -> None:
    output = args.plots / "generalization-joint-5x5.pdf"
    generalization.plot_joint(
        read_csv(args.data / "generalization_summary.csv"),
        read_csv(args.data / "generalization_joint_summary.csv"),
        output,
        show_average=args.avg,
    )
    print(f"Wrote {output}")


def plot_sat_duration(args: argparse.Namespace) -> None:
    sat.make_duration_plot(
        read_csv(args.data / "sat_measurements_summary.csv"),
        compare_shuffled=True,
        log_y=args.log_y,
        output_dir=args.plots,
    )
    print(f"Wrote {args.plots / 'solving_duration-comparison.pdf'}")


def plot_acceptance_rates(args: argparse.Namespace) -> None:
    full_summary = read_csv(args.data / "acceptance-rates-summary.csv")
    plotted = (
        full_summary
        if args.fully
        else full_summary.filter(pl.col("profile") != acceptance.FULLY_COUPLED)
    )
    acceptance.plot_timing(plotted, args.plots)
    acceptance.plot_acceptance(plotted, args.plots, y_ellipsis=args.y_ellipsis)
    timing = read_csv(args.data / "acceptance-generation-time.csv")
    generation_time = float(timing["generation_time_s"][0])
    table = acceptance.format_table(full_summary, generation_time, include_ci=args.ci)
    output = args.tables / "certification-yield-table.tex"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(table)
    print(f"Wrote acceptance-rate plots and {output}")


def write_exit_outcomes(args: argparse.Namespace) -> None:
    summary = read_csv(args.data / "exit_rate_distribution.csv")
    selection = summary.filter(
        (pl.col("pool") == "test") & (pl.col("pool_size") == args.pool_size)
    )
    if selection.is_empty():
        raise ValueError(f"No held-out rows at pool size {args.pool_size}")
    table = generalization.format_exit_outcomes_table(selection)
    output = args.tables / "exit_outcomes_table.tex"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(table)
    print(f"Wrote {output}")


def report_trajectory_profiles(args: argparse.Namespace) -> None:
    generalization.print_trajectory_report(
        read_csv(args.data / "trajectory_profile_runs.csv")
    )


def report_exit_distribution(args: argparse.Namespace) -> None:
    generalization.print_exit_summary(
        read_csv(args.data / "exit_rate_distribution.csv"), args.pool_size
    )


def plot_trajectory(args: argparse.Namespace) -> None:
    if args.layout is None:
        raise ValueError("--layout is required for --trajectory")
    filename = args.output.name if args.output else f"{args.layout.stem}-trajectory.pdf"
    output = (args.plots / filename).with_suffix(".pdf")
    layouts.render_trajectory(
        args.layout,
        output,
        trajectories=args.trajectories,
        t_max=args.t_max,
        labelled=args.labelled,
        first_label=args.first_label,
        hide_path=args.no_path,
        bullet=args.bullet,
        final_arrow=args.final_arrow,
    )
    print(f"Wrote {output}")


PLOTTERS: dict[str, Callable[[argparse.Namespace], None]] = {
    "coop-profiles": plot_all_cooperation,
    "coop-curves": plot_coop_curves,
    "coop-gap": plot_coop_gap,
    "coop-difficulty": plot_coop_difficulty,
    "coop-heatmap": plot_coop_heatmap,
    "coop-profile-table": write_coop_table,
    "coop-profile-failures": write_failure_table,
    "pool-characteristics": write_pool_table,
    "layouts-picture": plot_layouts,
    "generalization-layouts-picture": plot_generalization_layouts,
    "generalization": plot_generalization,
    "sat-duration": plot_sat_duration,
    "acceptance-rates": plot_acceptance_rates,
    "exit-outcomes": write_exit_outcomes,
    "trajectory-profiles": report_trajectory_profiles,
    "exit-rate-distribution": report_exit_distribution,
    "trajectory": plot_trajectory,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("targets", nargs="*", choices=PLOTTERS, metavar="TARGET")
    parser.add_argument("--list", action="store_true", help="List targets and exit.")
    for name in PLOTTERS:
        parser.add_argument(
            f"--{name}",
            dest="target_options",
            action="append_const",
            const=name,
            help=f"Render only {name} (repeatable).",
        )
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--plots", type=Path, default=Path("plots"))
    parser.add_argument("--tables", type=Path, default=Path("tables"))
    parser.add_argument("--layouts", type=Path, default=Path("layouts"))
    parser.add_argument("--pool-size", type=int, default=500)
    parser.add_argument("--layout-seed", type=int, default=None)
    parser.add_argument("--generalization-layout-seed", type=int, default=42)
    parser.add_argument("--transpose", action="store_true")
    parser.add_argument("--avg", action="store_true")
    parser.add_argument("--log-y", action="store_true")
    parser.add_argument("--fully", action="store_true")
    parser.add_argument("--ci", action="store_true")
    parser.add_argument("--count", action="store_true")
    parser.add_argument("--y-ellipsis", action="store_true")
    parser.add_argument("--layout", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--trajectories")
    parser.add_argument("--t-max", type=int)
    parser.add_argument("--labelled", action="store_true")
    parser.add_argument("--first-label", action="store_true")
    parser.add_argument("--no-path", action="store_true")
    parser.add_argument("--bullet", action="store_true")
    parser.add_argument("--final-arrow", action="store_true")
    args = parser.parse_args()
    requested = [*args.targets, *(args.target_options or [])]
    args.explicit_selection = bool(requested)
    args.selected = list(dict.fromkeys(requested)) or list(DEFAULT_TARGETS)
    return args


def main() -> None:
    args = parse_args()
    if args.list:
        width = max(map(len, PLOTTERS))
        for name in PLOTTERS:
            print(f"{name:<{width}}  {TARGET_HELP[name]}")
        return
    for index, name in enumerate(args.selected, start=1):
        print(f"\n=== [{index}/{len(args.selected)}] {name} ===")
        try:
            PLOTTERS[name](args)
        except FileNotFoundError as error:
            if args.explicit_selection:
                raise
            print(f"Skipped {name}: {error}")
    print(f"\nDone: {', '.join(args.selected)}")


if __name__ == "__main__":
    main()
