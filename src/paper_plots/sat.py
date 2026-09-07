"""SAT measurements, aggregation, and plotting."""

from __future__ import annotations

import csv
import random
import time
import typing
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
from lle import World
from lle.solver import solve_model
from lle.solver.clauses import ClauseGenerator
from lle.solver.types import SolveModeLiteral
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "mathtext.fontset": "cm",
        "axes.unicode_minus": False,
        "legend.frameon": True,
        "axes.labelsize": 14,
        "legend.fontsize": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.axisbelow": True,
        "grid.linewidth": 0.4,
        "grid.alpha": 0.4,
    }
)

DATA_DIR = Path("data")
PLOTS_DIR = Path("plots")
SAT_T_MAX = 156
SAT_REPETITIONS = 30
DEFAULT_LEVELS = list(range(1, 7))
DEFAULT_MODES = typing.get_args(SolveModeLiteral)
MEASUREMENT_FIELDS = [
    "level",
    "mode",
    "shuffled",
    "repetition",
    "t_max",
    "lower_bound",
    "width",
    "height",
    "n_agents",
    "n_gems",
    "n_laser_colours",
    "clause_count",
    "variable_count",
    "build_duration_s",
    "solve_duration_s",
    "total_duration_s",
    "sat",
]


def _format_duration(value: float, _pos: int) -> str:
    return f"{value:.1f}"


DURATION_FORMATTER = FuncFormatter(_format_duration)


def summarize_measurements(measurements: pl.DataFrame | None = None) -> pl.DataFrame:
    """Aggregate duration repetitions into one row per level and horizon."""

    if measurements is None:
        measurements = pl.read_csv(
            DATA_DIR / "sat-measurements.csv", infer_schema_length=None
        )

    return (
        measurements.group_by("level", "t", "max_t")
        .agg(pl.col("duration_s").mean().alias("duration_mean_s"))
        .with_columns(
            pl.col("level")
            .cast(pl.Utf8)
            .str.extract(r"^(\d+)")
            .cast(pl.Int64)
            .alias("level_number")
        )
        .sort("level_number", "t", "level")
    )


def selected_levels(summary: pl.DataFrame, *, shuffled: bool) -> pl.DataFrame:
    """Return standard or shuffled experiment rows."""

    pattern = r"-shuffled$" if shuffled else r"^\d+$"
    return summary.filter(pl.col("level").cast(pl.Utf8).str.contains(pattern))


def make_duration_plot(
    summary: pl.DataFrame | None = None,
    *,
    shuffled: bool = False,
    compare_shuffled: bool = False,
    log_y: bool = False,
    output_dir: Path = PLOTS_DIR,
) -> None:
    """Generate the SAT solving-duration plot."""

    if summary is None:
        summary = summarize_measurements()

    standard = selected_levels(summary, shuffled=False)
    plotted = (
        standard if compare_shuffled else selected_levels(summary, shuffled=shuffled)
    )
    variants = [(plotted, "--" if compare_shuffled else "-", True)]
    if compare_shuffled:
        variants.append((selected_levels(summary, shuffled=True), "-", False))

    levels = plotted.get_column("level_number").unique().sort().to_list()
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    level_colors = {
        level: colors[index % len(colors)] for index, level in enumerate(levels)
    }

    fig, ax = plt.subplots(figsize=(9, 3), constrained_layout=True)
    for variant, linestyle, show_labels in variants:
        for level in levels:
            rows = variant.filter(pl.col("level_number") == level)
            ax.plot(
                rows.get_column("t").to_list(),
                rows.get_column("duration_mean_s").to_list(),
                color=level_colors[level],
                linewidth=1,
                linestyle=linestyle,
                label=f"Level {level}" if show_labels else "_nolegend_",
            )

    ax.set(
        xlabel="Time horizon [t]",
        ylabel="Duration [s]",
        xlim=(0, plotted["max_t"].max()),
    )
    if log_y:
        ax.set_yscale("log")
    ax.yaxis.set_major_formatter(DURATION_FORMATTER)
    ax.grid(True, which="both")
    ax.margins(x=0.01, y=0.05)

    if compare_shuffled:
        ax.legend(
            handles=[
                Line2D([0], [0], color="black", linestyle="-", label="Shuffled"),
                Line2D([0], [0], color="black", linestyle="--", label="Non-shuffled"),
            ],
            loc="upper left",
        )
        fig.legend(
            handles=[
                Line2D([0], [0], color=level_colors[level], label=f"Level {level}")
                for level in levels
            ],
            loc="lower center",
            bbox_to_anchor=(0.5, 1.0),
            ncols=len(levels),
        )
    else:
        ax.legend(loc="best")

    suffix = "-comparison" if compare_shuffled else "-shuffled" if shuffled else ""
    output = output_dir / f"solving_duration{suffix}.pdf"
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def measure_once(world: World, mode: str, *, shuffle: bool) -> dict[str, object]:
    """Build and solve one SAT instance while measuring both phases."""
    generator = ClauseGenerator(world, SAT_T_MAX)
    build_start = time.perf_counter()
    clauses, assumptions = generator.generate(SAT_T_MAX, mode=mode)
    if shuffle:
        random.shuffle(clauses)
        random.shuffle(assumptions)
    build_duration = time.perf_counter() - build_start
    solve_start = time.perf_counter()
    model = solve_model(clauses, assumptions=assumptions)
    solve_duration = time.perf_counter() - solve_start
    return {
        "lower_bound": generator.solution_lower_bound,
        "clause_count": len(clauses),
        "variable_count": generator.n_vars,
        "build_duration_s": build_duration,
        "solve_duration_s": solve_duration,
        "total_duration_s": build_duration + solve_duration,
        "sat": model is not None,
    }


def migrate_measurements(output: Path) -> None:
    """Add ``shuffled=False`` to measurement files written before that field existed."""
    if not output.exists() or output.stat().st_size == 0:
        return
    with output.open("r", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None or "shuffled" in reader.fieldnames:
            return
        rows = list(reader)
    for row in rows:
        row["shuffled"] = False
    with output.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=MEASUREMENT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def run_measurements(
    output: Path,
    *,
    levels: list[int] = DEFAULT_LEVELS,
    modes: tuple[str, ...] = DEFAULT_MODES,
    shuffle: bool = False,
    repetitions: int = SAT_REPETITIONS,
    overwrite: bool = False,
) -> None:
    """Append a level/profile SAT measurement sweep to ``output``."""
    output.parent.mkdir(parents=True, exist_ok=True)
    if not overwrite:
        migrate_measurements(output)
    write_header = overwrite or not output.exists() or output.stat().st_size == 0
    with output.open("w" if overwrite else "a", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=MEASUREMENT_FIELDS)
        if write_header:
            writer.writeheader()
        for level in levels:
            world = World.level(level)
            for mode in modes:
                for repetition in range(repetitions):
                    stats = measure_once(world, mode, shuffle=shuffle)
                    row = {
                        "level": level,
                        "mode": mode,
                        "shuffled": shuffle,
                        "repetition": repetition,
                        "t_max": SAT_T_MAX,
                        "width": world.width,
                        "height": world.height,
                        "n_agents": world.n_agents,
                        "n_gems": world.n_gems,
                        "n_laser_colours": world.n_laser_colours,
                        **stats,
                    }
                    writer.writerow(row)
                    csv_file.flush()
                    print(
                        f"level={level} mode={mode} repetition={repetition} "
                        f"clauses={row['clause_count']} vars={row['variable_count']}"
                    )
