from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl
from coop_profiles import discover_experiments

ALGORITHMS = ("dqn", "vdn", "qmix", "ippo", "mappo")
ALGORITHM_LABELS = {
    "dqn": "DQN",
    "vdn": "VDN",
    "qmix": "QMIX",
    "ippo": "IPPO",
    "mappo": "MAPPO",
}
# Report order matches the existing tab:coop-profile-failures table in main.tex.
ALGORITHM_ORDER = ("qmix", "dqn", "vdn", "ippo", "mappo")
PROFILES = ("asymmetric", "convergent", "divergent", "sequential", "interdependent")
N_AGENTS = 3
FINAL_STEP = 1_000_000
EVAL_FILE = "test-policy-on-test-envs.csv"
AGENT_EXIT_COLUMNS = tuple(f"agent-{i}-exited" for i in range(N_AGENTS))
AGENT_ALIVE_COLUMNS = tuple(f"agent-{i}-alive" for i in range(N_AGENTS))


def parse_args() -> argparse.Namespace:
    """Parse the log root and the LaTeX destination.

    @ai-generated
    """
    parser = argparse.ArgumentParser(
        description=(
            "Generate the LaTeX table decomposing final-policy episodes on the "
            "cooperation-profile sweep into death, timeout and joint-success shares, "
            "averaged over the five cooperation profiles."
        )
    )
    parser.add_argument(
        "--logs", type=Path, default=Path("logs"), help="Experiment log root."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tables/coop_profile_failures_table.tex"),
        help="Destination for the generated LaTeX table.",
    )
    parser.add_argument(
        "--count",
        action="store_true",
        help="Show the raw joint-success count in parentheses next to its share.",
    )
    return parser.parse_args()


def episode_outcomes(path: Path) -> dict[str, object] | None:
    """Summarise one run's held-out episodes at the final checkpoint into outcome shares.

    Returns ``None`` when the file has no complete final-checkpoint block yet.

    @ai-generated
    """
    columns = [
        "time_step",
        "test_num",
        "exit_rate",
        "episode_len",
        *AGENT_EXIT_COLUMNS,
        *AGENT_ALIVE_COLUMNS,
    ]
    episodes = (
        pl.read_csv(path, columns=columns)
        .filter(pl.col("time_step") == FINAL_STEP)
        .unique("test_num", keep="last", maintain_order=True)
    )
    if episodes.is_empty():
        return None
    episodes = episodes.with_columns(
        pl.sum_horizontal(AGENT_EXIT_COLUMNS).alias("n_exited"),
        (N_AGENTS - pl.sum_horizontal(AGENT_ALIVE_COLUMNS)).alias("n_dead"),
    )
    return episodes.select(
        pl.col("exit_rate").mean(),
        pl.col("n_dead").mean().alias("deaths"),
        (pl.col("n_dead") > 0).mean().alias("death_share"),
        (pl.col("n_exited") == N_AGENTS).sum().alias("joint_count"),
        pl.col("episode_len").mean().alias("length"),
        pl.len().alias("n_episodes"),
    ).row(0, named=True)


def collect_runs(log_root: Path) -> pl.DataFrame:
    """Read every seed's final-checkpoint outcome shares across the cooperation-profile sweep.

    @ai-generated
    """
    rows: list[dict[str, object]] = []
    for profile, algorithm, directory in discover_experiments(log_root):
        for run in sorted(directory.glob("run-*")):
            path = run / EVAL_FILE
            if not path.exists() or path.stat().st_size == 0:
                continue
            outcomes = episode_outcomes(path)
            if outcomes is None:
                continue
            rows.append(
                {"profile": profile, "algorithm": algorithm, "seed": run.name}
                | outcomes
            )
    if not rows:
        raise FileNotFoundError(
            f"No held-out final-checkpoint episodes found under {log_root}"
        )
    return pl.DataFrame(rows)


def aggregate(runs: pl.DataFrame) -> pl.DataFrame:
    """Average seed-level shares within each profile, then average the five profiles.

    Averaging in two stages weights every profile equally regardless of its seed count,
    matching how the rest of the cooperation-profile analysis blocks on profile. Joint
    success is pooled as a raw count over every profile and seed instead: it is rare
    enough (single-digit occurrences per algorithm) that a profile-weighted share would
    round to zero everywhere and hide the count that actually produced it.

    @ai-generated
    """
    share_metrics = ("exit_rate", "deaths", "death_share", "length")
    per_profile = runs.group_by("algorithm", "profile").agg(
        pl.col(*share_metrics).mean()
    )
    shares = per_profile.group_by("algorithm").agg(pl.col(*share_metrics).mean())
    counts = runs.group_by("algorithm").agg(
        pl.col("joint_count").sum(), pl.col("n_episodes").sum()
    )
    return shares.join(counts, on="algorithm")


def format_table(summary: pl.DataFrame, *, show_count: bool) -> str:
    """Render the death/timeout/joint-success decomposition as a LaTeX table.

    Death and timeout are complementary by construction (the episode ends one way or the
    other), so rounding one and subtracting from 100 keeps them summing to 100\\%. Joint
    success is reported as a share of the same pooled episodes at 3-decimal precision,
    since at whole-percent precision every algorithm's share rounds to 0\\%; the raw count
    can be appended in parentheses via ``show_count`` so the reader can see how thin that
    share is.

    @ai-generated
    """
    total_episodes = int(summary["n_episodes"][0])
    lines = [
        r"    \begin{tabular}{lcccccc}",
        r"        \toprule",
        r"        \multirow{2}{*}{\textbf{Algorithm}} & \multirow{2}{*}{\textbf{Exit rate}} & \multirow{2}{*}{\textbf{Deaths/episode}} & \multicolumn{3}{c}{\textbf{Episode outcome}} & \multirow{2}{*}{\textbf{Length}}\\",
        r"        \cmidrule(lr){4-6}",
        r"        & & & Death & Timeout & Joint success & \\",
        r"        \midrule",
    ]
    for algorithm in ALGORITHM_ORDER:
        row = summary.filter(pl.col("algorithm") == algorithm).row(0, named=True)
        death_pct = round(row["death_share"] * 100)
        timeout_pct = 100 - death_pct
        joint_count = int(row["joint_count"])
        joint_pct = 100 * joint_count / total_episodes
        joint_cell = f"{joint_pct:.3f}\\%"
        if show_count:
            joint_cell += f" ({joint_count}/\\num{{{total_episodes}}})"
        lines.append(
            f"        {ALGORITHM_LABELS[algorithm]:<9} & {row['exit_rate']:.3f}     "
            f"& {row['deaths']:.2f}             & {death_pct}\\%             "
            f"& {timeout_pct}\\%     & {joint_cell}"
            f"     & {row['length']:.1f}\\\\"
        )
    lines += [
        r"        \bottomrule",
        r"    \end{tabular}",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    """Generate the death/timeout/joint-success LaTeX table from the cooperation-profile sweep.

    @ai-generated
    """
    args = parse_args()
    runs = collect_runs(args.logs)
    summary = aggregate(runs)

    table = format_table(summary, show_count=args.count)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(table)
    print(f"Wrote {args.output}")
    print(table)


if __name__ == "__main__":
    main()
