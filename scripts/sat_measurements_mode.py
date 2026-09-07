"""Measure SAT clause/variable counts and timings for each standard level and solve mode."""

from __future__ import annotations

import argparse
import csv
import random
import time
import typing
from pathlib import Path

from lle import World
from lle.solver import solve_model
from lle.solver.clauses import ClauseGenerator
from lle.solver.types import SolveModeLiteral

T_MAX = 156
N_REPETITIONS = 30
DEFAULT_LEVELS = list(range(1, 7))
DEFAULT_MODES = typing.get_args(SolveModeLiteral)
OUTPUT_PATH = Path("data/sat-measurements-mode.csv")

FIELDNAMES = [
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


def measure_once(world: World, mode: str, *, shuffle: bool) -> dict[str, object]:
    """Build the SAT clauses for `mode` at `T_MAX` and solve them once, timing each step.

    A fresh `ClauseGenerator` is used for every call so the build duration reflects the full
    cost of a solve, independent of caching effects from previous modes on a shared generator.
    When `shuffle` is set, the clause and assumption order is randomized right after generation
    (before timing stops for the build phase) to measure the effect of clause order on the solver.

    @ai-generated
    """
    generator = ClauseGenerator(world, T_MAX)

    build_start = time.perf_counter()
    clauses, assumptions = generator.generate(T_MAX, mode=mode)
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


def migrate_add_shuffled_column() -> None:
    """Backfill `shuffled=False` on rows written before the --shuffle option existed.

    Older rows in `OUTPUT_PATH` predate the `shuffled` column; appending new rows with an extra
    column would otherwise leave the CSV with an inconsistent number of fields per row.

    @ai-generated
    """
    if not OUTPUT_PATH.exists() or OUTPUT_PATH.stat().st_size == 0:
        return
    with OUTPUT_PATH.open("r", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None or "shuffled" in reader.fieldnames:
            return
        rows = list(reader)
    for row in rows:
        row["shuffled"] = False
    with OUTPUT_PATH.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    """Parse the optional --levels and --profiles arguments.

    @ai-generated
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--levels",
        type=int,
        nargs="+",
        default=DEFAULT_LEVELS,
        help="Standard LLE levels to measure (default: 1-6).",
    )
    parser.add_argument(
        "--profiles",
        dest="modes",
        type=str,
        nargs="+",
        default=DEFAULT_MODES,
        help=(
            "Solve modes/profiles to measure, e.g. 'standard', 'no-mutual', "
            "'no-interdependence-3' (default: all canonical solve modes)."
        ),
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Randomize the clause and assumption order before solving each instance.",
    )
    return parser.parse_args()


def main() -> None:
    """Sweep the requested levels x solve profiles x 30 repetitions, appending each row to the CSV as obtained.

    @ai-generated
    """
    args = parse_args()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    migrate_add_shuffled_column()
    write_header = not OUTPUT_PATH.exists() or OUTPUT_PATH.stat().st_size == 0

    with OUTPUT_PATH.open("a", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()
            csv_file.flush()

        for level in args.levels:
            world = World.level(level)
            for mode in args.modes:
                for repetition in range(N_REPETITIONS):
                    stats = measure_once(world, mode, shuffle=args.shuffle)
                    row = {
                        "level": level,
                        "mode": mode,
                        "shuffled": args.shuffle,
                        "repetition": repetition,
                        "t_max": T_MAX,
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
                        f"clauses={row['clause_count']} vars={row['variable_count']} "
                        f"build={row['build_duration_s']:.3f}s solve={row['solve_duration_s']:.3f}s"
                    )


if __name__ == "__main__":
    main()
