"""Analyse whether an agent exits more often when its own laser colour is present."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl
from coop_profiles import (
    AGENT_EXIT_COLUMNS,
    FINAL_STEP,
    LAYOUT_FAMILIES,
    N_AGENTS,
    N_BOOTSTRAPS,
    N_PERMUTATIONS,
    POOL_FILES,
    RNG_SEED,
    discover_experiments,
)


def parse_args() -> argparse.Namespace:
    """Parse input and output paths.

    @ai-generated
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs", type=Path, default=Path("logs"))
    parser.add_argument("--layout-data", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("data"))
    return parser.parse_args()


def collect_layout_scores(log_root: Path, layout_data: Path) -> pl.DataFrame:
    """Average final test outcomes by profile, algorithm, layout, and agent.

    The evaluation replays test layouts 500--999 in ``test_num`` order. Duplicate rows
    from restarted logging are removed before averaging the 30 seeds.

    @ai-generated
    """
    frames: list[pl.DataFrame] = []
    for profile, algorithm, directory in discover_experiments(log_root):
        runs: list[pl.DataFrame] = []
        for run in sorted(directory.glob("run-*")):
            path = run / POOL_FILES["test"]
            if not path.exists():
                continue
            episodes = (
                pl.read_csv(
                    path, columns=["time_step", "test_num", *AGENT_EXIT_COLUMNS]
                )
                .filter(pl.col("time_step") == FINAL_STEP)
                .unique("test_num", keep="last", maintain_order=True)
            )
            if episodes.height != 500:
                continue
            for agent, column in enumerate(AGENT_EXIT_COLUMNS):
                runs.append(
                    episodes.select(
                        (pl.col("test_num") + 500).alias("layout_index"),
                        pl.lit(agent).alias("agent"),
                        pl.col(column).cast(pl.Float64).alias("exited"),
                    )
                )
        if not runs:
            continue
        scores = (
            pl.concat(runs)
            .group_by("layout_index", "agent")
            .agg(pl.col("exited").mean(), pl.len().alias("n_seeds"))
            .with_columns(
                pl.lit(profile).alias("profile"),
                pl.lit(algorithm).alias("algorithm"),
            )
        )
        colours = pl.read_csv(
            layout_data / f"{LAYOUT_FAMILIES[profile]}-laser-colours.csv"
        ).rename({"index": "layout_index"})
        for agent in range(N_AGENTS):
            colours = colours.with_columns(
                (pl.col(f"laser-{agent}") > 0).alias(f"matched-{agent}")
            )
        scores = (
            scores.join(colours, on="layout_index", how="left")
            .with_columns(
                pl.when(pl.col("agent") == agent)
                .then(pl.col(f"matched-{agent}"))
                .otherwise(None)
                .alias(f"is-matched-{agent}")
                for agent in range(N_AGENTS)
            )
            .with_columns(
                pl.coalesce(
                    *(f"is-matched-{agent}" for agent in range(N_AGENTS))
                ).alias("matched")
            )
        )
        frames.append(
            scores.select(
                "profile",
                "algorithm",
                "layout_index",
                "agent",
                "n_seeds",
                "matched",
                "exited",
            )
        )
    if not frames:
        raise FileNotFoundError("No complete endpoint evaluations were found.")
    result = pl.concat(frames)
    if set(result["n_seeds"].unique()) != {30}:
        raise ValueError("Every layout score must average exactly 30 seeds.")
    return result


def layout_contrasts(scores: pl.DataFrame, group_columns: list[str]) -> pl.DataFrame:
    """Collapse agent outcomes to one matched-minus-absent contrast per layout.

    @ai-generated
    """
    keys = list(dict.fromkeys([*group_columns, "profile", "layout_index"]))
    matched = (
        scores.filter("matched")
        .group_by(keys)
        .agg(pl.col("exited").mean().alias("matched_rate"))
    )
    absent = (
        scores.filter(~pl.col("matched"))
        .group_by(keys)
        .agg(pl.col("exited").mean().alias("absent_rate"))
    )
    return matched.join(absent, on=keys).with_columns(
        (pl.col("matched_rate") - pl.col("absent_rate")).alias("difference")
    )


def bootstrap_summary(
    contrasts: pl.DataFrame,
    group_columns: list[str],
    rng: np.random.Generator,
) -> pl.DataFrame:
    """Summarize rates and profile-stratified layout-bootstrap intervals.

    @ai-generated
    """
    groups = (
        contrasts.partition_by(group_columns, as_dict=True)
        if group_columns
        else {("overall",): contrasts}
    )
    rows: list[dict[str, object]] = []
    for key, group in groups.items():
        key = key if isinstance(key, tuple) else (key,)
        profile_arrays = [
            profile.sort("layout_index")
            .select("matched_rate", "absent_rate", "difference")
            .to_numpy()
            for profile in group.partition_by("profile")
        ]
        samples = np.empty((N_BOOTSTRAPS, 3), dtype=float)
        for sample_index in range(N_BOOTSTRAPS):
            profile_means = []
            for values in profile_arrays:
                indices = rng.integers(0, len(values), size=len(values))
                profile_means.append(values[indices].mean(axis=0))
            samples[sample_index] = np.mean(profile_means, axis=0)
        estimate = np.mean([values.mean(axis=0) for values in profile_arrays], axis=0)
        low, high = np.quantile(samples[:, 2], [0.025, 0.975])
        row = dict(zip(group_columns, key, strict=True)) if group_columns else {}
        rows.append(
            row
            | {
                "matched_rate": estimate[0],
                "absent_rate": estimate[1],
                "difference": estimate[2],
                "ci_low": low,
                "ci_high": high,
                "n_layouts": group.select("profile", "layout_index").unique().height,
                "n_bootstraps": N_BOOTSTRAPS,
            }
        )
    return pl.DataFrame(rows)


def permutation_p_value(scores: pl.DataFrame, rng: np.random.Generator) -> float:
    """Permute missing-colour labels among layouts within each profile.

    Outcomes are first averaged over seeds and algorithms. Each permutation preserves
    the observed number of layouts missing each colour in every profile.

    @ai-generated
    """
    averaged = scores.group_by("profile", "layout_index", "agent").agg(
        pl.col("exited").mean(), pl.col("matched").first()
    )
    strata: list[tuple[np.ndarray, np.ndarray]] = []
    for profile in averaged.partition_by("profile"):
        outcomes = (
            profile.pivot(on="agent", index="layout_index", values="exited")
            .sort("layout_index")
            .select(*(str(agent) for agent in range(N_AGENTS)))
            .to_numpy()
        )
        missing = (
            profile.filter(~pl.col("matched")).sort("layout_index")["agent"].to_numpy()
        )
        strata.append((outcomes, missing))

    def statistic(labels: list[np.ndarray]) -> float:
        profile_differences = []
        for (outcomes, _), missing in zip(strata, labels, strict=True):
            absent = outcomes[np.arange(len(outcomes)), missing]
            matched = (outcomes.sum(axis=1) - absent) / (N_AGENTS - 1)
            profile_differences.append(float(np.mean(matched - absent)))
        return float(np.mean(profile_differences))

    observed_labels = [missing for _, missing in strata]
    observed = abs(statistic(observed_labels))
    extreme = 0
    for _ in range(N_PERMUTATIONS):
        permuted = [rng.permutation(missing) for _, missing in strata]
        if abs(statistic(permuted)) >= observed - 1e-12:
            extreme += 1
    return (extreme + 1) / (N_PERMUTATIONS + 1)


def agent_summary(scores: pl.DataFrame) -> pl.DataFrame:
    """Average endpoint exit probabilities by agent identity.

    @ai-generated
    """
    return scores.group_by("agent").agg(pl.col("exited").mean()).sort("agent")


def main() -> None:
    """Write reusable per-layout data and inferential summaries.

    @ai-generated
    """
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    scores = collect_layout_scores(args.logs, args.layout_data)
    scores.write_csv(args.output / "laser_colour_layout_scores.csv")

    rng = np.random.default_rng(RNG_SEED)
    overall = bootstrap_summary(layout_contrasts(scores, []), [], rng).with_columns(
        pl.lit(permutation_p_value(scores, rng)).alias("p_value")
    )
    overall.write_csv(args.output / "laser_colour_summary.csv")
    bootstrap_summary(
        layout_contrasts(scores, ["algorithm"]), ["algorithm"], rng
    ).write_csv(args.output / "laser_colour_by_algorithm.csv")
    bootstrap_summary(
        layout_contrasts(scores, ["profile"]), ["profile"], rng
    ).write_csv(args.output / "laser_colour_by_profile.csv")
    bootstrap_summary(
        layout_contrasts(scores, ["profile", "algorithm"]),
        ["profile", "algorithm"],
        rng,
    ).write_csv(args.output / "laser_colour_by_profile_algorithm.csv")
    agent_summary(scores).write_csv(args.output / "agent_exit_summary.csv")
    print(overall)


if __name__ == "__main__":
    main()
