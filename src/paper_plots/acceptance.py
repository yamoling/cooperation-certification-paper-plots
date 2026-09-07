"""Acceptance-rate experiments, summaries, plots, and tables."""

from __future__ import annotations

import csv
import time
from collections.abc import Callable
from concurrent.futures import (
    FIRST_COMPLETED,
    Future,
    ProcessPoolExecutor,
    as_completed,
    wait,
)
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
from lle import World, characterize
from lle.characterization.world_characterization import WorldCharacterizer
from lle.generator.generator import WorldGenerator
from tqdm import tqdm

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "text.latex.preamble": r"\usepackage{amsmath}",
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

DATA_CSV = Path("data/acceptance-rates.csv")
NEGATIVE_QUERY_DATA_CSV = Path("data/no-profile-x-end-to-end-timings.csv")
SUMMARY_CSV = Path("data/acceptance-rates-summary.csv")
PLOTS_DIR = Path("plots")
TABLE_OUTPUT = Path("tables/certification-yield-table.tex")

WIDTH = 9
HEIGHT = 9
N_AGENTS = 3
N_LASERS = 2

ACCEPTANCE_PREDICATES: dict[str, Callable[[WorldCharacterizer], bool]] = {
    "cooperative": lambda c: c.is_cooperative(),
    "asymmetric": lambda c: c.is_asymmetric(),
    "sequential-2": lambda c: c.is_sequential(length=2),
    "divergent-2": lambda c: c.is_divergent(k=2),
    "convergent-2": lambda c: c.is_convergent(k=2),
    "interdependent-2": lambda c: c.is_interdependent(n_agents=2),
    "fully-coupled": lambda c: c.is_fully_coupled(),
    "solvable": lambda c: c.is_solvable(),
}
ACCEPTANCE_TIME_FIELDS = [f"{name}-time" for name in ACCEPTANCE_PREDICATES]
ACCEPTANCE_FIELDNAMES = [
    "layout",
    *ACCEPTANCE_PREDICATES,
    *ACCEPTANCE_TIME_FIELDS,
]

PROFILES = [
    "solvable",
    "cooperative",
    "asymmetric",
    "sequential-2",
    "divergent-2",
    "convergent-2",
    "interdependent-2",
]
FULLY_COUPLED = "fully-coupled"

LABELS = {
    "cooperative": r"$\mathrm{Coop}$",
    "asymmetric": r"$\mathrm{Asym}$",
    "sequential-2": r"$\mathrm{Seq}_2$",
    "divergent-2": r"$\mathrm{Div}_2$",
    "convergent-2": r"$\mathrm{Conv}_2$",
    "interdependent-2": r"$\mathrm{Inter}_2$",
    "fully-coupled": r"$\mathrm{Fully}$",
    "solvable": "Solvable",
}
WILSON_Z = 1.96


def build_generator() -> WorldGenerator:
    """Build an unbiased world generator without an acceptance predicate."""
    return WorldGenerator(
        width=WIDTH, height=HEIGHT, n_agents=N_AGENTS, n_lasers=N_LASERS
    )


def characterize_layout(layout: str, t_max: int) -> dict[str, object]:
    """Characterize one layout against every acceptance predicate."""
    row: dict[str, object] = {"layout": layout}
    for name, predicate in ACCEPTANCE_PREDICATES.items():
        characterizer = characterize(World(layout), t_max)
        start = time.perf_counter()
        row[f"{name}-time"] = time.perf_counter() - start
        row[name] = predicate(characterizer)
    return row


def count_existing_rows(output: Path) -> int:
    """Count completed rows so an interrupted experiment can resume."""
    if not output.exists() or output.stat().st_size == 0:
        return 0
    with output.open(newline="") as existing_file:
        reader = csv.reader(existing_file)
        next(reader, None)
        return sum(1 for _ in reader)


def run_experiment(
    n_layouts: int,
    n_workers: int,
    t_max: int,
    seed: int | None,
    output: Path,
    *,
    overwrite: bool = False,
) -> None:
    """Generate and characterize random layouts, resuming an existing CSV."""
    output.parent.mkdir(parents=True, exist_ok=True)
    existing_rows = 0 if overwrite else count_existing_rows(output)
    remaining = n_layouts - existing_rows
    if remaining <= 0:
        print(f"{output} already has {existing_rows} rows (>= {n_layouts} requested)")
        return

    generator = build_generator()
    pending: set[Future] = set()
    mode = "a" if existing_rows else "w"
    with output.open(mode, newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=ACCEPTANCE_FIELDNAMES)
        if not existing_rows:
            writer.writeheader()
            output_file.flush()
        with (
            ProcessPoolExecutor(max_workers=n_workers) as executor,
            tqdm(total=remaining, unit="layout", smoothing=0.01) as progress,
        ):
            for world in generator.generate_n(
                remaining, n_jobs=1, seed=seed, quiet=True
            ):
                pending.add(
                    executor.submit(characterize_layout, world.world_string, t_max)
                )
                if len(pending) >= n_workers * 4:
                    done, pending = wait(pending, return_when=FIRST_COMPLETED)
                    for future in done:
                        writer.writerow(future.result())
                        output_file.flush()
                        progress.update()
            for future in as_completed(pending):
                writer.writerow(future.result())
                progress.update()


def wilson_interval(
    n_accepted: int, n: int, z: float = WILSON_Z
) -> tuple[float, float]:
    """Compute the Wilson score interval for a binomial proportion.

    Unlike the normal-approximation (Wald) interval, this stays within
    [0, 1] and remains well-calibrated when the proportion is close to 0 or
    1 or `n` is small — both true of the low-acceptance profiles here.

    @ai-generated
    """
    p = n_accepted / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = z * (p * (1 - p) / n + z**2 / (4 * n**2)) ** 0.5 / denom
    return center - margin, center + margin


def summarize(
    data: pl.DataFrame,
    negative_query_data: pl.DataFrame,
    profiles: list[str],
) -> pl.DataFrame:
    """Compute yield and timing statistics for the certification table.

    `solvable-time` isolates the increasing-horizon search shared by every
    certification call. Negative-query means come from fresh, end-to-end
    calls at the maximal horizon and include formula construction and solving.
    These isolated phase means are descriptive benchmarks: the negative query
    is conditional, so they are not added to obtain complete call durations.

    The 95% confidence interval on the mean certification time uses the
    normal approximation `1.96 * std / sqrt(n)`, matching the convention used
    elsewhere in this project. The acceptance rate, being a proportion
    rather than a continuous quantity, instead gets a Wilson score interval.

    @ai-generated
    """
    required_negative_profiles = set(profiles).difference({"solvable"})
    available_negative_profiles = set(negative_query_data["profile"].unique())
    missing_negative_profiles = required_negative_profiles - available_negative_profiles
    if missing_negative_profiles:
        raise ValueError(
            "Missing negative-query timings for profiles: "
            f"{sorted(missing_negative_profiles)}"
        )

    negative_query_means = {
        row["profile"]: row["negative_query_mean_time_s"]
        for row in (
            negative_query_data.group_by("profile")
            .agg(
                pl.len().alias("negative_query_n"),
                pl.col("total_duration_s").mean().alias("negative_query_mean_time_s"),
            )
            .iter_rows(named=True)
        )
    }
    horizon_search_mean_time_s = data["solvable-time"].mean()
    assert horizon_search_mean_time_s is not None

    rows = []
    for profile in profiles:
        times = data[f"{profile}-time"]
        n = times.len()
        std = times.std()
        assert std is not None
        n_accepted = int(data[profile].sum())
        wilson_low, wilson_high = wilson_interval(n_accepted, n)
        rows.append(
            {
                "profile": profile,
                "label": LABELS[profile],
                "n": n,
                "n_accepted": n_accepted,
                "acceptance_rate": data[profile].mean(),
                "acceptance_rate_wilson_low": wilson_low,
                "acceptance_rate_wilson_high": wilson_high,
                "horizon_search_mean_time_s": horizon_search_mean_time_s,
                "negative_query_mean_time_s": negative_query_means.get(profile),
                "mean_time_s": times.mean(),
                "std_time_s": std,
                "ci95_time_s": 1.96 * std / (n**0.5),
            }
        )
    return pl.DataFrame(rows)


def plot_timing(summary: pl.DataFrame, plots_dir: Path) -> None:
    """Plot mean per-profile certification time with 95% confidence intervals.

    Certification times are measured from independent, freshly constructed
    `WorldCharacterizer` instances per profile, so the comparison is not
    distorted by cross-profile caching shortcuts.

    @ai-generated
    """
    labels = summary.get_column("label").to_list()
    means_ms = (summary.get_column("mean_time_s") * 1000).to_list()
    ci_ms = (summary.get_column("ci95_time_s") * 1000).to_list()

    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    x = range(len(labels))
    colors = plt.cm.viridis([i / max(len(labels) - 1, 1) for i in x])
    bars = ax.bar(
        x,
        means_ms,
        yerr=ci_ms,
        capsize=4,
        color=colors,
        edgecolor="black",
        linewidth=0.6,
    )
    for bar, mean, ci in zip(bars, means_ms, ci_ms):
        ax.annotate(
            f"{mean:.1f} ms",
            xy=(bar.get_x() + bar.get_width() / 2, mean + ci),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            fontsize=10,
        )
    # ax.set_yscale("log")
    # ax.set_ylim(top=max(m + c for m, c in zip(means_ms, ci_ms)) * 1.4)
    ax.set_ylabel("Mean certification time [ms]")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.grid(True, axis="y", which="both")

    plots_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(plots_dir / "acceptance-timing.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_acceptance(
    summary: pl.DataFrame, plots_dir: Path, y_ellipsis: bool = False
) -> None:
    """Plot the acceptance rate of each profile among the sampled layouts.

    When a profile (e.g. "solvable") sits an order of magnitude above the
    rest, a linear axis flattens the smaller bars into illegibility. Passing
    `y_ellipsis=True` switches to a broken y-axis (two stacked subplots with
    a diagonal cut mark), giving the low bars their own scale while still
    showing the tall bar's height.

    @ai-generated
    """
    labels = summary.get_column("label").to_list()
    rates_pct = (summary.get_column("acceptance_rate") * 100).to_list()

    if not y_ellipsis:
        fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
        x = range(len(labels))
        colors = plt.cm.viridis([i / max(len(labels) - 1, 1) for i in x])
        bars = ax.bar(x, rates_pct, color=colors, edgecolor="black", linewidth=0.6)
        for bar, rate in zip(bars, rates_pct):
            ax.annotate(
                f"{rate:.2f}\\%",
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                fontsize=10,
            )
        ax.set_ylabel("Acceptance rate [\\%]")
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels)
        ax.set_ymargin(0.12)
        ax.grid(True, axis="y")

        plots_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(plots_dir / "acceptance-rates.pdf", bbox_inches="tight")
        plt.close(fig)
        return

    low_rates = [r for r in rates_pct if r < 20]
    low_top = (max(low_rates) if low_rates else 10) * 1.05
    high_top = max(rates_pct) * 1.05

    fig, (ax_high, ax_low) = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=(8, 5),
        constrained_layout=True,
        gridspec_kw={"height_ratios": [1, 3]},
    )
    x = range(len(labels))
    colors = plt.cm.viridis([i / max(len(labels) - 1, 1) for i in x])
    for ax in (ax_high, ax_low):
        bars = ax.bar(x, rates_pct, color=colors, edgecolor="black", linewidth=0.6)
        for bar, rate in zip(bars, rates_pct):
            ax.annotate(
                f"{rate:.2f}\\%",
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                fontsize=10,
            )
        ax.grid(True, axis="y")

    ax_high.set_ylim(bottom=low_top, top=high_top)
    ax_low.set_ylim(bottom=0, top=low_top)
    ax_high.spines["bottom"].set_visible(False)
    ax_low.spines["top"].set_visible(False)
    ax_high.tick_params(bottom=False)

    d = 0.012
    kwargs = {
        "transform": ax_high.transAxes,
        "color": "black",
        "clip_on": False,
        "lw": 1,
    }
    ax_high.plot((-d, +d), (-d, +d), **kwargs)
    ax_high.plot((1 - d, 1 + d), (-d, +d), **kwargs)
    kwargs.update(transform=ax_low.transAxes)
    scale = ax_high.get_position().height / ax_low.get_position().height
    ax_low.plot((-d, +d), (1 - d * scale, 1 + d * scale), **kwargs)
    ax_low.plot((1 - d, 1 + d), (1 - d * scale, 1 + d * scale), **kwargs)

    ax_low.set_ylabel("Acceptance rate [\\%]")
    ax_low.set_xticks(list(x))
    ax_low.set_xticklabels(labels)

    plots_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(plots_dir / "acceptance-rates.pdf", bbox_inches="tight")
    plt.close(fig)


def measure_generation_time(n: int) -> float:
    """Benchmark the mean single-core layout-generation time.

    Uses the exact same plain, unfiltered generator as the acceptance-rate
    experiment (`build_generator`, no acceptance predicate), with `n_jobs=1`,
    so the result is directly comparable to (and addable with) the
    single-core certification times.

    @ai-generated
    """
    generator = build_generator()
    start = time.perf_counter()
    list(generator.generate_n(n, n_jobs=1, quiet=True))
    return (time.perf_counter() - start) / n


def format_table(
    summary: pl.DataFrame,
    generation_time_s: float,
    *,
    include_ci: bool = False,
) -> str:
    """Render the profile-certification cost table as booktabs LaTeX.

    The isolated negative-query duration is shown beside the three aggregate
    `RequiresProfile` metrics: acceptance rate, complete-call duration, and
    estimated time per accepted layout. The fully-coupled target remains in
    the table to expose its negative-query cost, but its undefined aggregate
    metrics are rendered as `---`. Wilson intervals are appended to acceptance
    rates only when `include_ci` is true.

    @ai-generated
    """
    lines = [
        r"    \begin{tabular}{lrrrr}",
        r"        \toprule",
        r"        \multirow{2}{*}{\textbf{Target}} & \multirow{2}{*}{\textbf{\shortstack{Mean $\phi_\p^-$\\duration}}} & \multicolumn{3}{c}{\textbf{RequiresProfile($L, \p$)}} \\",
        r"        \cmidrule(lr){3-5}",
        r"        & & Acceptance rate & Mean duration & Mean Time/accepted\\",
        r"        \midrule",
    ]
    for row in summary.iter_rows(named=True):
        rate = row["acceptance_rate"]
        rate_pct = rate * 100
        wilson_low_pct = row["acceptance_rate_wilson_low"] * 100
        wilson_high_pct = row["acceptance_rate_wilson_high"] * 100
        acceptance_str = f"{rate_pct:.3f}\\%"
        if include_ci:
            acceptance_str += (
                f" {{\\tiny [{wilson_low_pct:.2f}, {wilson_high_pct:.2f}]}}"
            )
        negative_query_time_s = row["negative_query_mean_time_s"]
        negative_query_str = (
            f"{negative_query_time_s * 1000:,.1f}~ms"
            if negative_query_time_s is not None
            else "---"
        )
        per_attempt_s = generation_time_s + row["mean_time_s"]
        call_time_str = f"{row['mean_time_s'] * 1000:,.1f}~ms"

        if row["profile"] == FULLY_COUPLED:
            acceptance_str = "---"
            call_time_str = "---"
            total_str = "---"
        else:
            total_s = per_attempt_s / rate
            total_str = f"{total_s:,.2f}~s"

        lines.append(
            f"        {row['label']} & {negative_query_str} "
            f"& {acceptance_str} & {call_time_str} & {total_str} \\\\"
        )
    lines += [r"        \bottomrule", r"    \end{tabular}"]
    return "\n".join(lines) + "\n"
