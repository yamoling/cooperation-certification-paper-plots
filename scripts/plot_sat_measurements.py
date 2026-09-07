from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shuffled",
        action="store_true",
        help="Plot the shuffled-experiment rows instead of the standard rows.",
    )
    parser.add_argument(
        "--compare-shuffled",
        action="store_true",
        help="Overlay shuffled and non-shuffled duration lines.",
    )
    parser.add_argument(
        "--log-y", action="store_true", help="Use a logarithmic duration axis."
    )
    return parser.parse_args()


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
) -> None:
    """Generate the SAT solving-duration plot."""

    if summary is None:
        summary = summarize_measurements()

    standard = selected_levels(summary, shuffled=False)
    plotted = standard if compare_shuffled else selected_levels(summary, shuffled=shuffled)
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
    output = PLOTS_DIR / f"solving_duration{suffix}.png"
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    make_duration_plot(
        shuffled=args.shuffled,
        compare_shuffled=args.compare_shuffled,
        log_y=args.log_y,
    )


if __name__ == "__main__":
    main()
