from __future__ import annotations

import argparse
from pathlib import Path

import lle
from lle.generator import Independent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="Output directory")
    parser.add_argument(
        "--t-max",
        type=int,
        default=81,
        help="Solver horizon used to prove each property (default: 40)",
    )
    parser.add_argument(
        "--n-jobs", type=int, default=1, help="Number of generator workers (default: 1)"
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Optional RNG seed; requires --n-jobs 1"
    )
    return parser.parse_args()


def main() -> None:
    """Generate and save 1 000 layouts satisfying the requested properties."""
    args = parse_args()
    args.n_jobs = max(args.n_jobs, 1)

    args.output.mkdir(parents=True, exist_ok=True)
    existing_paths = [path for path in args.output.iterdir() if path.is_file()]
    layouts = set()
    for path in existing_paths:
        content = path.read_text(encoding="utf-8")
        if content in layouts:
            raise ValueError(f"Duplicate layout already exists: {path}")
        layouts.add(content)

    layouts_to_generate = 10_000 - len(existing_paths)
    if layouts_to_generate <= 0:
        print(
            f"Output directory already contains {len(existing_paths)} layouts: {args.output}"
        )
        return

    first_index = len(existing_paths)
    output_paths = [
        args.output / f"{index:04d}.txt"
        for index in range(first_index, first_index + layouts_to_generate)
    ]
    collision = next((path for path in output_paths if path.exists()), None)
    if collision is not None:
        raise FileExistsError(f"Refusing to overwrite existing layout: {collision}")

    predicate = Independent()
    generator = (
        lle.generate(width=9, height=9, n_agents=3, t_max=args.t_max)
        .lasers(2)
        .require(predicate)
    )
    worlds = generator.take(
        layouts_to_generate, n_jobs=args.n_jobs, seed=args.seed, progress=True
    )
    for output_path, world in zip(output_paths, worlds, strict=True):
        content = f"{world.world_string}\n"
        if content in layouts:
            raise ValueError("Generator produced a duplicate layout")
        layouts.add(content)
        output_path.write_text(content, encoding="utf-8")

    print(f"Wrote {layouts_to_generate} layouts to {args.output}")


if __name__ == "__main__":
    main()
