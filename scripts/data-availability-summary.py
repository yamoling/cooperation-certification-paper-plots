"""Report what is missing from the cooperation-profile experiment.

Run it with no arguments to get the current state of the sweep:

    uv run python scripts/data-availability-summary.py

The evaluation sweep writes files continuously, so this is a snapshot. Anything it
reports as missing is missing *right now*.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path

import polars as pl

PROFILES = ("asymmetric", "convergent", "divergent", "sequential", "interdependent")
ALGORITHMS = ("dqn", "ippo", "mappo", "qmix", "vdn")
EXPECTED_SEEDS = 30
# Fallbacks for runs whose run.json is missing or truncated; every run.json seen so
# far agrees with these.
DEFAULT_N_STEPS = 1_000_000
DEFAULT_TEST_INTERVAL = 50_000
DEFAULT_N_TESTS = 500

POOL_FILES = {
    "train": "test-policy-on-train-envs.csv",
    "test": "test-policy-on-test-envs.csv",
}
LEGACY_FILES = ("test.csv", "train.csv")
# Stem of the per-layout laser-colour tables in `data/`; only asymmetric drops the `-2`.
LAYOUT_FAMILIES = {
    "asymmetric": "asymmetric",
    "convergent": "convergent-2",
    "divergent": "divergent-2",
    "sequential": "sequential-2",
    "interdependent": "interdependent-2",
}


@dataclass
class Run:
    """One seed of one configuration, with whatever its logs currently contain.

    @ai-generated
    """

    profile: str
    algorithm: str
    seed: int
    path: Path
    n_steps: int = DEFAULT_N_STEPS
    test_interval: int = DEFAULT_TEST_INTERVAL
    n_tests: int = DEFAULT_N_TESTS
    layout_dir: str | None = None
    # Highest layout index either pool indexes into, i.e. how many layout files the
    # run needs on disk.
    pool_span: int = 0
    # pool -> {checkpoint: episodes logged}; absent key means the file is absent.
    pools: dict[str, dict[int, int]] = field(default_factory=dict)
    legacy_last_step: dict[str, int] = field(default_factory=dict)
    empty_files: list[str] = field(default_factory=list)
    missing_artifacts: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def expected_checkpoints(self) -> list[int]:
        """The checkpoint grid this run should have evaluated.

        @ai-generated
        """
        return list(range(0, self.n_steps + 1, self.test_interval))

    @property
    def training_reached(self) -> int | None:
        """Latest step any log records, or ``None`` when the run produced no logs.

        @ai-generated
        """
        steps = [max(counts) for counts in self.pools.values() if counts] + list(
            self.legacy_last_step.values()
        )
        return max(steps) if steps else None


def parse_args() -> argparse.Namespace:
    """Parse the directories to inspect.

    @ai-generated
    """
    parser = argparse.ArgumentParser(
        description="Summarise missing data in the cooperation-profile experiment."
    )
    parser.add_argument("--logs", type=Path, default=Path("logs"))
    parser.add_argument(
        "--seeds",
        type=int,
        default=EXPECTED_SEEDS,
        help="Seeds expected per configuration.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data-summary.md"),
        help="Markdown report to write.",
    )
    parser.add_argument(
        "--json", type=Path, default=None, help="Also write the findings as JSON."
    )
    return parser.parse_args()


def human_step(step: int) -> str:
    """Render a time step as 0 / 50k / 1M.

    @ai-generated
    """
    if step == 0:
        return "0"
    if step % 1_000_000 == 0:
        return f"{step // 1_000_000}M"
    return f"{step // 1_000}k"


def compress(steps: Sequence[int]) -> str:
    """Collapse a sorted checkpoint list into ranges, e.g. ``0-200k, 950k-1M``.

    @ai-generated
    """
    if not steps:
        return "-"
    ordered = sorted(steps)
    spacing = min((b - a for a, b in pairwise(ordered)), default=0)
    groups: list[list[int]] = [[ordered[0]]]
    for step in ordered[1:]:
        if spacing and step - groups[-1][-1] == spacing:
            groups[-1].append(step)
        else:
            groups.append([step])
    return ", ".join(
        human_step(g[0]) if len(g) == 1 else f"{human_step(g[0])}-{human_step(g[-1])}"
        for g in groups
    )


def experiment_directory(log_root: Path, profile: str, algorithm: str) -> Path | None:
    """Find a profile/algorithm's configuration directory.

    @ai-generated
    """
    profile_stem = LAYOUT_FAMILIES[profile]
    return next(
        (
            directory
            for directory in sorted(
                log_root.glob(f"canonical-{profile_stem}-{algorithm}-*")
            )
            if directory.is_dir()
        ),
        None,
    )


def checkpoint_counts(path: Path) -> dict[int, int] | None:
    """Count logged episodes per checkpoint, reading only the ``time_step`` column.

    Returns ``None`` if the file cannot be read, which happens while the evaluation
    sweep is appending to it.

    @ai-generated
    """
    try:
        frame = pl.read_csv(path, columns=["time_step"])
    except pl.exceptions.PolarsError, OSError:
        return None
    counts = frame.group_by("time_step").len().sort("time_step")
    return dict(zip(counts["time_step"], counts["len"], strict=True))


def load_run(profile: str, algorithm: str, seed: int, path: Path) -> Run:
    """Read one run directory into a :class:`Run`.

    @ai-generated
    """
    run = Run(profile=profile, algorithm=algorithm, seed=seed, path=path)
    metadata = path / "run.json"
    if metadata.exists():
        try:
            payload = json.loads(metadata.read_text())
        except json.JSONDecodeError, OSError:
            run.notes.append("unreadable run.json")
            payload = {}
        run.n_steps = payload.get("n_steps", DEFAULT_N_STEPS)
        run.test_interval = payload.get("test_interval", DEFAULT_TEST_INTERVAL)
        run.n_tests = payload.get("n_tests", DEFAULT_N_TESTS)
        run.layout_dir = (payload.get("env") or {}).get("directory")
        run.pool_span = max(
            (
                (pool or {}).get("offset", 0) + (pool or {}).get("size", 0)
                for pool in (payload.get("env"), payload.get("test_env"))
            ),
            default=0,
        )
        # `save_weights` / `save_actions` promise artifacts that replay and qualitative
        # analysis depend on; check they are actually on disk.
        artifacts = {
            entry.name
            for entry in path.iterdir()
            if entry.suffix not in {".csv", ".json"} and entry.name != "pid"
        }
        for flag in ("save_weights", "save_actions"):
            if payload.get(flag) and not artifacts:
                run.missing_artifacts.append(flag)
    else:
        run.notes.append("no run.json")
    if (path / "run.pre-restore.json").exists():
        run.notes.append("restored from checkpoint")

    for pool, filename in POOL_FILES.items():
        target = path / filename
        if not target.exists():
            continue
        if target.stat().st_size == 0:
            run.empty_files.append(filename)
            continue
        counts = checkpoint_counts(target)
        if counts is None:
            run.notes.append(f"unreadable {filename}")
            continue
        run.pools[pool] = counts

    for filename in LEGACY_FILES:
        target = path / filename
        if not target.exists():
            continue
        if target.stat().st_size == 0:
            run.empty_files.append(filename)
            continue
        counts = checkpoint_counts(target)
        if counts:
            run.legacy_last_step[filename] = max(counts)
    return run


def collect(
    log_root: Path,
) -> tuple[list[Run], list[tuple[str, str]], list[str]]:
    """Load every run of the sweep, and list absent configurations and skipped dirs.

    @ai-generated
    """
    runs: list[Run] = []
    absent: list[tuple[str, str]] = []
    skipped = sorted(
        directory.name
        for directory in log_root.iterdir()
        if directory.is_dir()
        and directory.name.endswith("-failed")
        and directory.name.split("-")[0] in PROFILES
    )
    for profile in PROFILES:
        for algorithm in ALGORITHMS:
            directory = experiment_directory(log_root, profile, algorithm)
            if directory is None:
                absent.append((profile, algorithm))
                continue
            for run_dir in sorted(directory.glob("run-*")):
                try:
                    seed = int(run_dir.name.removeprefix("run-"))
                except ValueError:
                    continue
                runs.append(load_run(profile, algorithm, seed, run_dir))
    return runs, absent, skipped


def group_runs(runs: Iterable[Run]) -> dict[tuple[str, str], list[Run]]:
    """Index runs by configuration.

    @ai-generated
    """
    grouped: dict[tuple[str, str], list[Run]] = {}
    for run in runs:
        grouped.setdefault((run.profile, run.algorithm), []).append(run)
    return grouped


def training_progress(runs: Sequence[Run], expected_seeds: int) -> dict[str, object]:
    """Summarise finished, in-progress, and unstarted training seeds.

    A run with no readable training/evaluation log is counted as unstarted. Missing
    seed directories are also unstarted, so all three counts are out of the same
    expected seed total.

    @ai-generated
    """
    finished = [
        run
        for run in runs
        if run.training_reached is not None and run.training_reached >= run.n_steps
    ]
    partial = [
        run
        for run in runs
        if run.training_reached is not None and run.training_reached < run.n_steps
    ]
    partial_steps = sorted(
        run.training_reached for run in partial if run.training_reached is not None
    )
    partial_steps = [
        run.training_reached for run in partial if run.training_reached is not None
    ]
    partial_range = (
        human_step(min(partial_steps))
        if len(set(partial_steps)) == 1
        else f"{human_step(min(partial_steps))}-{human_step(max(partial_steps))}"
        if partial_steps
        else "-"
    )
    started = len(finished) + len(partial)
    return {
        "done": len(finished),
        "partial": len(partial),
        "partial_range": partial_range,
        "not_started": max(expected_seeds - started, 0),
        "partial_seeds": [run.seed for run in partial],
        "not_started_seeds": sorted(
            set(range(expected_seeds))
            - {run.seed for run in runs if run.training_reached is not None}
        ),
    }


def evaluation_progress(
    runs: Sequence[Run], pool: str, expected_seeds: int
) -> dict[str, object]:
    """Summarise CSV availability and full checkpoint-grid coverage for one pool.

    A complete grid contains every checkpoint from 0 through 1M at 50k intervals.
    The grid check deliberately tests checkpoint presence rather than the number of
    logged episodes at each checkpoint, which is a separate data-quality concern.

    @ai-generated
    """
    checkpoints = list(range(0, DEFAULT_N_STEPS + 1, DEFAULT_TEST_INTERVAL))
    with_file = [run for run in runs if pool in run.pools]
    complete = [
        run
        for run in with_file
        if all(checkpoint in run.pools[pool] for checkpoint in checkpoints)
    ]
    incomplete_steps = sorted(
        {
            checkpoint
            for run in with_file
            for checkpoint in checkpoints
            if checkpoint not in run.pools[pool]
        }
    )
    return {
        "files": len(with_file),
        "complete": len(complete),
        "missing_checkpoints": compress(incomplete_steps),
        "incomplete_seeds": sorted(
            set(range(expected_seeds)) - {run.seed for run in complete}
        ),
    }


def completion_cell(completed: object, expected: int) -> str:
    if completed == expected:
        return "✅"
    if completed == 0:
        icon = "❌"
    else:
        icon = "🟡"
    return f"{completed}/{expected} {icon}"


def render_markdown(
    grouped: dict[tuple[str, str], list[Run]], expected_seeds: int, log_root: Path
) -> tuple[str, dict[tuple[str, str], dict[str, object]]]:
    """Render the availability matrix and retain its structured per-cell summaries.

    @ai-generated
    """
    summaries: dict[tuple[str, str], dict[str, object]] = {}
    lines = [
        "# Data availability summary",
        "",
        f"Snapshot: {datetime.now(UTC).astimezone():%Y-%m-%d %H:%M %Z}",
        f"Log directory: `{log_root}`",
        "",
        (
            f"Each count is out of {expected_seeds} expected seeds. A checkpoint grid is complete when "
            "the CSV contains every checkpoint from 0 through 1M in 50k increments."
        ),
        "",
        "| Profile | Algorithm | Training done | `test-policy-on-test-envs.csv` complete | `test-policy-on-train-envs.csv` complete |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for profile in PROFILES:
        for algorithm in ALGORITHMS:
            runs = grouped.get((profile, algorithm), [])
            training = training_progress(runs, expected_seeds)
            test = evaluation_progress(runs, "test", expected_seeds)
            train = evaluation_progress(runs, "train", expected_seeds)
            summaries[(profile, algorithm)] = {
                "training": training,
                "test": test,
                "train": train,
            }
            lines.append(
                f"| {profile} | {algorithm} | "
                f"{completion_cell(training['done'], expected_seeds)} | "
                f"{completion_cell(test['complete'], expected_seeds)} | "
                f"{completion_cell(train['complete'], expected_seeds)} |"
            )

    lines.append("")
    return "\n".join(lines), summaries


def main() -> None:
    """Write the availability report for the cooperation-profile sweep.

    @ai-generated
    """
    args = parse_args()
    if not args.logs.is_dir():
        raise SystemExit(f"No log directory at {args.logs}")

    runs, _, _ = collect(args.logs)
    runs = [run for run in runs if 0 <= run.seed < args.seeds]
    report, summaries = render_markdown(group_runs(runs), args.seeds, args.logs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report)
    print(f"Markdown report written to {args.output}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(
                {
                    f"{profile}-{algorithm}": summary
                    for (profile, algorithm), summary in summaries.items()
                },
                indent=2,
                sort_keys=True,
            )
        )
        print(f"JSON written to {args.json}")


if __name__ == "__main__":
    main()
