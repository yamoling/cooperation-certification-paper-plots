from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl
from exit_rate_distribution import SETTINGS, aggregate, collect_runs

ALGORITHM_LABELS = {
    "dqn": "DQN",
    "vdn": "VDN",
    "qmix": "QMIX",
    "ippo": "IPPO",
    "mappo": "MAPPO",
}
# Report order: value-based methods, then policy-gradient methods.
ALGORITHM_ORDER = ("dqn", "vdn", "qmix", "ippo", "mappo")
SETTING_LABELS = {"independent": "Independent", "cooperative": "Cooperative"}


def parse_args() -> argparse.Namespace:
    """Parse the log root, the held-out pool size, and the LaTeX destination.

    @ai-generated
    """
    parser = argparse.ArgumentParser(
        description=(
            "Generate the LaTeX table decomposing held-out exit rate into the "
            "none/one/both-exited episode shares, with a per-class average row."
        )
    )
    parser.add_argument(
        "--logs", type=Path, default=Path("logs"), help="Experiment log root."
    )
    parser.add_argument(
        "--pool-size",
        type=int,
        default=500,
        help="Training-pool size tabulated.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tables/exit_outcomes_table.tex"),
        help="Destination for the generated LaTeX table.",
    )
    return parser.parse_args()


def average_row(rows: pl.DataFrame) -> dict[str, float]:
    """Average exit-rate outcome shares across algorithms within a layout class.

    @ai-generated
    """
    means = rows.select(
        pl.col("exit_rate", "none_exited", "one_exited", "both_exited").mean()
    ).row(0, named=True)
    return means


def format_table(selection: pl.DataFrame) -> str:
    """Render the held-out outcome split as a LaTeX table with a mean row per layout class.

    @ai-generated
    """
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
            rf"        \multirow{{{len(ALGORITHM_ORDER) + 1}}}{{*}}{{{SETTING_LABELS[setting]}}}"
        )
        for algorithm in ALGORITHM_ORDER:
            row = rows.filter(pl.col("algorithm") == algorithm).row(0, named=True)
            lines.append(
                f"          & {ALGORITHM_LABELS[algorithm]} "
                f"& {row['exit_rate']:.3f} & {row['none_exited']:.3f} "
                f"& {row['one_exited']:.3f} & {row['both_exited']:.3f} \\\\"
            )
        mean = average_row(rows)
        lines.append(r"        \cmidrule(lr){2-6}")
        lines.append(
            f"          & Mean "
            f"& {mean['exit_rate']:.3f} & {mean['none_exited']:.3f} "
            f"& {mean['one_exited']:.3f} & {mean['both_exited']:.3f} \\\\"
        )
        if setting != SETTINGS[-1]:
            lines.append(r"        \midrule")
    lines += [
        r"        \bottomrule",
        r"    \end{tabular}",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    """Generate the exit-outcome LaTeX table from the 5x5 sweep logs.

    @ai-generated
    """
    args = parse_args()
    runs = collect_runs(args.logs)
    summary = aggregate(runs)
    selection = summary.filter(
        (pl.col("pool") == "test") & (pl.col("pool_size") == args.pool_size)
    )
    if selection.is_empty():
        raise ValueError(f"No held-out rows at pool size {args.pool_size}")

    table = format_table(selection)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(table)
    print(f"Wrote {args.output}")
    print(table)


if __name__ == "__main__":
    main()
