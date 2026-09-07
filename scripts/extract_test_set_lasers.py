from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

LASER_PATTERN = re.compile(r"\bL([0-9])\w*\b")
LAYOUT_INDEX_PATTERN = re.compile(r"(\d+)$")
CSV_FIELDS = ("index", "laser-0", "laser-1", "laser-2")


def parse_args() -> argparse.Namespace:
    """Parse the layout directory and output path for laser-count extraction.

    @ai-generated
    """
    parser = argparse.ArgumentParser(
        description="Count laser colours in the layouts in a directory."
    )
    parser.add_argument(
        "layouts",
        type=Path,
        help="Directory containing layout files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="CSV output path (default: data/<directory name>-laser-colours.csv).",
    )
    return parser.parse_args()


def get_layout_index(layout_path: Path) -> int:
    """Extract the trailing numeric index from a layout filename.

    @ai-generated
    """
    match = LAYOUT_INDEX_PATTERN.search(layout_path.stem)
    if match is None:
        raise ValueError(f"Layout filename has no numeric index: {layout_path}")
    return int(match.group(1))


def extract_laser_counts(layout_dir: Path) -> list[tuple[int, int, int, int]]:
    """Return laser-colour counts in the layout loader's sorted index order.

    The evaluation environment applies its train/test offsets to the sorted file list,
    so the CSV index must be the file's position rather than the number in its name.

    @ai-generated
    """
    layout_paths = sorted(
        (path for path in layout_dir.glob("*") if path.is_file()),
        key=get_layout_index,
    )
    if not layout_paths:
        raise FileNotFoundError(f"No layout files found in {layout_dir}")

    rows: list[tuple[int, int, int, int]] = []
    for layout_index, layout_path in enumerate(layout_paths):
        counts = [0, 0, 0]
        for colour in LASER_PATTERN.findall(layout_path.read_text()):
            colour_index = int(colour)
            if colour_index > 2:
                raise ValueError(
                    f"Unsupported laser colour L{colour_index} in {layout_path}"
                )
            counts[colour_index] += 1
        rows.append((layout_index, *counts))
    return rows


def write_counts(rows: list[tuple[int, int, int, int]], output_path: Path) -> None:
    """Write laser counts as CSV, creating the parent directory if needed.

    @ai-generated
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(CSV_FIELDS)
        writer.writerows(rows)


def main() -> None:
    """Extract laser counts from one layout directory and write them to CSV.

    @ai-generated
    """
    args = parse_args()
    output_path = args.output or Path("data", f"{args.layouts.name}-laser-colours.csv")
    rows = extract_laser_counts(args.layouts)
    write_counts(rows, output_path)
    print(f"Wrote {len(rows)} laser-count rows to {output_path}")


if __name__ == "__main__":
    main()
