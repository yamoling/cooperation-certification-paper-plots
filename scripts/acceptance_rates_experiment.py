"""Measure how often plain, unfiltered random layouts satisfy each cooperation profile (and solvability)."""

from __future__ import annotations

import argparse
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

from lle import World, characterize
from lle.characterization.world_characterization import WorldCharacterizer
from lle.generator.generator import WorldGenerator
from tqdm import tqdm

WIDTH = 9
HEIGHT = 9
N_AGENTS = 3
N_LASERS = 2

PROFILES: dict[str, Callable[[WorldCharacterizer], bool]] = {
    "cooperative": lambda c: c.is_cooperative(),
    "asymmetric": lambda c: c.is_asymmetric(),
    "sequential-2": lambda c: c.is_sequential(length=2),
    "divergent-2": lambda c: c.is_divergent(k=2),
    "convergent-2": lambda c: c.is_convergent(k=2),
    "interdependent-2": lambda c: c.is_interdependent(n_agents=2),
    "fully-coupled": lambda c: c.is_fully_coupled(),
    "solvable": lambda c: c.is_solvable(),
}


def build_generator() -> WorldGenerator:
    """Build a plain world generator with no acceptance predicate at all.

    `lle.generate(...)` goes through `GeneratorBuilder`, which always wraps
    generation in a `Constraint` that defaults to an implicit `Solvable()`
    predicate — so every layout it produces is silently pre-filtered to be
    solvable. Constructing `WorldGenerator` directly and leaving `constraint`
    at its own default (`None`) accepts every structurally valid candidate
    instead, which is what an unbiased acceptance-rate measurement needs.

    @ai-generated
    """
    return WorldGenerator(
        width=WIDTH, height=HEIGHT, n_agents=N_AGENTS, n_lasers=N_LASERS
    )


TIME_FIELDS = [f"{name}-time" for name in PROFILES]
FIELDNAMES = ["layout", *PROFILES, *TIME_FIELDS]


def parse_args() -> argparse.Namespace:
    """Parse CLI options for the acceptance-rate experiment.

    @ai-generated
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n-layouts",
        type=int,
        default=100_000,
        help="Number of layouts to sample (default: %(default)s)",
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        default=10,
        help="Number of parallel characterization workers (default: %(default)s)",
    )
    parser.add_argument(
        "--t-max",
        type=int,
        default=81,
        help="Solver horizon used for characterization (default: %(default)s)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional RNG seed for the (single-job) generator",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/acceptance-rates.csv"),
        help="CSV output path (default: %(default)s)",
    )
    return parser.parse_args()


def process_layout(layout: str, t_max: int) -> dict[str, object]:
    """Characterize one layout string against every profile, timing each check.

    Every profile gets its own brand new `World` and `characterize()` call,
    so no profile benefits from another profile's cached solve (e.g. a world
    already known not to be cooperative short-circuiting the
    asymmetric/interdependent/etc. checks) — each timing is the honest,
    standalone cost of certifying that profile from scratch.

    Runs as the unit of work submitted to the characterization pool, so it
    must take and return only picklable values.

    @ai-generated
    """
    row: dict[str, object] = {"layout": layout}
    for name, predicate in PROFILES.items():
        fresh_characterizer = characterize(World(layout), t_max)
        start = time.perf_counter()
        value = predicate(fresh_characterizer)
        row[f"{name}-time"] = time.perf_counter() - start
        row[name] = value
    return row


def count_existing_rows(output: Path) -> int:
    """Count data rows already written by a previous, interrupted run.

    @ai-generated
    """
    if not output.exists() or output.stat().st_size == 0:
        return 0
    with output.open(newline="") as existing_file:
        reader = csv.reader(existing_file)
        next(reader, None)
        return sum(1 for _ in reader)


def run(
    n_layouts: int, n_workers: int, t_max: int, seed: int | None, output: Path
) -> None:
    """Sample plain, unfiltered random layouts and stream their profile membership to a CSV.

    Layout generation (via `build_generator`, with no acceptance predicate)
    runs as a single job while characterization is spread over `n_workers`
    processes. In-flight characterization tasks are capped so that generation
    cannot race arbitrarily far ahead of writing, and each row is flushed to
    disk as soon as its characterization completes.

    If `output` already holds rows from a previous, interrupted run, this
    resumes by appending only the missing rows instead of overwriting it.

    @ai-generated
    """
    output.parent.mkdir(parents=True, exist_ok=True)
    existing_rows = count_existing_rows(output)
    remaining = n_layouts - existing_rows
    if remaining <= 0:
        print(f"{output} already has {existing_rows} rows (>= {n_layouts} requested)")
        return

    generator = build_generator()

    max_pending = n_workers * 4
    pending: set[Future] = set()

    mode = "a" if existing_rows > 0 else "w"
    with output.open(mode, newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=FIELDNAMES)
        if existing_rows == 0:
            writer.writeheader()
            output_file.flush()
        with (
            ProcessPoolExecutor(max_workers=n_workers) as executor,
            tqdm(total=remaining, unit="layout", smoothing=0.01) as progress,
        ):
            for world in generator.generate_n(
                remaining, n_jobs=1, seed=seed, quiet=True
            ):
                pending.add(executor.submit(process_layout, world.world_string, t_max))
                if len(pending) >= max_pending:
                    done, pending = wait(pending, return_when=FIRST_COMPLETED)
                    for future in done:
                        writer.writerow(future.result())
                        output_file.flush()
                        progress.update(1)
            for future in as_completed(pending):
                writer.writerow(future.result())
                progress.update(1)


def main() -> None:
    """Run the acceptance-rate experiment from CLI arguments.

    @ai-generated
    """
    args = parse_args()
    run(args.n_layouts, args.n_workers, args.t_max, args.seed, args.output)
    print(f"Wrote acceptance-rate results to {args.output}")


if __name__ == "__main__":
    main()
