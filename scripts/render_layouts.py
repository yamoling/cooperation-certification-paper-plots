#!/usr/bin/env python3
"""Render five random LLE layouts from each family in a 5×5 Matplotlib grid."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
from lle import World

LAYOUT_FAMILIES = (
    "asymmetric",
    "convergent-2",
    "divergent-2",
    "sequential-2",
    "interdependent-2",
)
N_LAYOUTS_PER_FAMILY = 5

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed", type=int, default=None, help="Optional seed for reproducible sampling"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("latex/pictures/layouts.pdf"),
        help="Output PDF path",
    )
    parser.add_argument(
        "--transpose",
        action="store_true",
        help="Place layout families in rows and write their titles vertically",
    )
    parser.add_argument(
        "--layouts",
        type=Path,
        default=Path("layouts/canonical"),
        help="Directory holding one subdirectory of layout files per family",
    )
    return parser.parse_args()


def main() -> None:
    """Load, randomly sample, and render layouts grouped by family.

    @ai-generated
    """
    args = parse_args()
    rng = random.Random(args.seed)
    selected_files = []
    for family in LAYOUT_FAMILIES:
        family_files = sorted((args.layouts / family).glob("*.txt"))
        if len(family_files) < N_LAYOUTS_PER_FAMILY:
            raise ValueError(
                f"{family!r} contains {len(family_files)} layouts; "
                f"{N_LAYOUTS_PER_FAMILY} are required"
            )
        selected_files.append(rng.sample(family_files, N_LAYOUTS_PER_FAMILY))

    figure, axes = plt.subplots(5, 5, figsize=(10, 10), squeeze=False)
    figure.subplots_adjust(
        left=0.04 if args.transpose else 0.005,
        right=0.995,
        bottom=0.005,
        top=0.95,
        wspace=0.01,
        hspace=0.01,
    )

    if args.transpose:
        for row, (family, family_layouts) in enumerate(
            zip(LAYOUT_FAMILIES, selected_files)
        ):
            header = family.replace("-", " ").title()
            axes[row, 0].text(
                -0.04,
                0.5,
                header,
                transform=axes[row, 0].transAxes,
                rotation=90,
                ha="right",
                va="center",
                fontsize=14,
                clip_on=False,
            )
            for column, layout_file in enumerate(family_layouts):
                axis = axes[row, column]
                world = World.from_file(str(layout_file))
                world.reset()
                axis.imshow(world.get_image())
                axis.set_axis_off()
    else:
        for column, (family, family_layouts) in enumerate(
            zip(LAYOUT_FAMILIES, selected_files)
        ):
            header = family.replace("-", " ").title()
            axes[0, column].set_title(header, fontsize=14, pad=5)
            for row, layout_file in enumerate(family_layouts):
                axis = axes[row, column]
                world = World.from_file(str(layout_file))
                world.reset()
                axis.imshow(world.get_image())
                axis.set_axis_off()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=200, bbox_inches="tight", pad_inches=0.02)
    plt.close(figure)


if __name__ == "__main__":
    main()
