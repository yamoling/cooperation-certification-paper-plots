from __future__ import annotations

import argparse
import csv
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from pathlib import Path

import polars as pl
from lle import World, characterize
from tqdm import tqdm

PREDICATES = {
    "asymmetric": lambda characterizer: characterizer.is_asymmetric(),
    "sequential-2": lambda characterizer: characterizer.is_sequential(length=2),
    "divergent-2": lambda characterizer: characterizer.is_divergent(k=2),
    "convergent-2": lambda characterizer: characterizer.is_convergent(k=2),
    "interdependent-2": lambda characterizer: characterizer.is_interdependent(
        n_agents=2
    ),
}

# Predicate name -> LaTeX macro, as defined in latex/preamble.tex.
LATEX_MACROS = {
    "asymmetric": r"\asym",
    "sequential-2": r"\seq_2",
    "divergent-2": r"\div_2",
    "convergent-2": r"\conv_2",
    "interdependent-2": r"\inter_2",
}


def positive_integer(value: str) -> int:
    """Parse a strictly positive command-line integer.

    @ai-generated
    """
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def parse_args() -> argparse.Namespace:
    """Parse the layout directory, worker count and output path for predicate extraction.

    @ai-generated
    """
    parser = argparse.ArgumentParser(
        description="Check cooperation-predicate membership for every layout in a directory."
    )
    parser.add_argument(
        "layouts",
        type=Path,
        nargs="?",
        default=None,
        help="Directory containing layout files. Omit when using --table.",
    )
    parser.add_argument(
        "--n-workers",
        type=positive_integer,
        default=1,
        help="number of layouts characterized in parallel (default: 1)",
    )
    parser.add_argument(
        "--limit",
        type=positive_integer,
        default=None,
        help="maximum number of layouts to process (default: all)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/layout-characteristics.csv"),
        help=(
            "CSV written when characterizing, or read when using --table "
            "(default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="overwrite the output CSV instead of appending to it",
    )
    parser.add_argument(
        "--table",
        action="store_true",
        help=(
            "Skip characterization and instead aggregate the --output CSV into "
            "the pool x predicate 'required-profile' percentage table, one row "
            "per pool (the layout's parent directory) and one column per "
            "predicate."
        ),
    )
    parser.add_argument(
        "--table-output",
        type=Path,
        default=None,
        help="Destination for the generated LaTeX table (printed if omitted).",
    )
    args = parser.parse_args()
    if not args.table and args.layouts is None:
        parser.error("the 'layouts' argument is required unless --table is given")
    return args


def list_layouts(layout_dir: Path) -> list[Path]:
    """List every layout file in a directory.

    @ai-generated
    """
    layout_paths = sorted(path for path in layout_dir.glob("*") if path.is_file())
    if not layout_paths:
        raise FileNotFoundError(f"No layout files found in {layout_dir}")
    return layout_paths


def process_layout(layout_path: Path) -> dict[str, object]:
    """Evaluate every predicate for one layout, tagged with its path.

    Runs as the unit of work submitted to the executor pool, so it must take
    and return only picklable values.

    @ai-generated
    """
    row: dict[str, object] = {"path": str(layout_path)}
    world = World.from_file(str(layout_path))
    characterizer = characterize(world, world.width * world.height)
    for name, predicate in PREDICATES.items():
        row[name] = predicate(characterizer)
    return row


def characterize_pool(
    layout_paths: list[Path],
    output_path: Path,
    n_workers: int,
    overwrite: bool,
) -> int:
    """Characterize every layout in parallel, writing each row to CSV as it arrives.

    Layouts are dispatched to a process pool with a bounded number of pending
    tasks. Each result is written and flushed as soon as it completes, so
    partial progress survives an interruption.

    @ai-generated
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = (
        overwrite or not output_path.exists() or output_path.stat().st_size == 0
    )
    mode = "w" if overwrite else "a"
    n_written = 0
    with output_path.open(mode, newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=["path", *PREDICATES])
        if write_header:
            writer.writeheader()
            output_file.flush()
        max_pending = n_workers * 4
        layout_iter = iter(layout_paths)
        with (
            ProcessPoolExecutor(max_workers=n_workers) as executor,
            tqdm(total=len(layout_paths), unit="layout") as progress,
        ):
            pending: set[Future[dict[str, object]]] = set()
            while True:
                while len(pending) < max_pending:
                    try:
                        layout_path = next(layout_iter)
                    except StopIteration:
                        break
                    pending.add(executor.submit(process_layout, layout_path))
                if not pending:
                    break
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    writer.writerow(future.result())
                    n_written += 1
                    progress.update(1)
    return n_written


def summarize_pools(csv_path: Path) -> pl.DataFrame:
    """Aggregate a combined characteristics CSV into a pool x predicate percentage matrix.

    Each layout's pool is taken from its parent directory name (e.g. a layout at
    ``layouts/canonical/asymmetric/00001.txt`` belongs to the ``asymmetric`` pool),
    which is how `pool-characteristics.py` is invoked once per pool directory. Rows
    are ordered following `PREDICATES`, restricted to pools actually present in the
    CSV.

    @ai-generated
    """
    rows = pl.read_csv(csv_path)
    predicate_names = [name for name in PREDICATES if name in rows.columns]
    rows = rows.with_columns(
        pl.col("path").str.extract(r"([^/\\]+)[/\\][^/\\]+$", 1).alias("pool")
    )
    summary = rows.group_by("pool").agg(
        (pl.col(name).mean() * 100) for name in predicate_names
    )
    pool_order = [pool for pool in PREDICATES if pool in summary["pool"]]
    return pl.concat([summary.filter(pl.col("pool") == pool) for pool in pool_order])


def format_table(summary: pl.DataFrame) -> str:
    """Render a pool x predicate percentage matrix as a LaTeX tabular.

    Mirrors the layout of `apx:tab:pool-characteristics` in `latex/appendix.tex`:
    one row per pool, one column per predicate, both labelled with the predicate's
    LaTeX macro and ordered following `PREDICATES`.

    @ai-generated
    """
    predicate_names = [name for name in PREDICATES if name in summary.columns]
    n_cols = len(predicate_names)
    header_cells = " & ".join(f"${LATEX_MACROS[name]}$" for name in predicate_names)
    lines = [
        r"\begin{tabular}{l" + "r" * n_cols + "}",
        r"    \toprule",
        r"     & \multicolumn{"
        + str(n_cols)
        + r"}{c}{\textbf{Layouts requiring predicate (\%)}} \\",
        r"    \cmidrule(lr){2-" + str(n_cols + 1) + "}",
        f"    \\textbf{{Pool}} & {header_cells} \\\\",
        r"    \midrule",
    ]
    for row in summary.iter_rows(named=True):
        values = " & ".join(f"{row[name]:.1f}" for name in predicate_names)
        lines.append(f"    ${LATEX_MACROS[row['pool']]}$ & {values} \\\\")
    lines += [r"    \bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def main() -> None:
    """Check cooperation predicates for a layout directory, or aggregate them into a table.

    @ai-generated
    """
    args = parse_args()
    if args.table:
        table = format_table(summarize_pools(args.output))
        if args.table_output is not None:
            args.table_output.parent.mkdir(parents=True, exist_ok=True)
            args.table_output.write_text(table)
            print(f"Wrote {args.table_output}")
        print(table)
        return
    layout_paths = list_layouts(args.layouts)[: args.limit]
    n_written = characterize_pool(
        layout_paths, args.output, args.n_workers, args.overwrite
    )
    verb = "Overwrote" if args.overwrite else "Appended"
    print(f"{verb} {n_written} predicate rows to {args.output}")


if __name__ == "__main__":
    main()
