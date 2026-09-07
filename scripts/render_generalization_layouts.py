#!/usr/bin/env python3
"""Render representative layouts from the independent and cooperative 5×5 pools."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
from lle import World

LAYOUT_CLASSES = ("independent", "cooperative")
N_LAYOUTS_PER_CLASS = 5

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
    }
)


def parse_args() -> argparse.Namespace:
    """Parse renderer inputs and output path.

    @ai-generated
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--layouts",
        type=Path,
        default=Path("layouts/tuning"),
        help="Directory containing the independent and cooperative pools.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("latex/pictures/generalization-layouts-5x5.pdf"),
        help="Output PDF path.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed used to sample five layouts from each pool.",
    )
    return parser.parse_args()


def main() -> None:
    """Sample and render five layouts from each 5×5 layout class.

    @ai-generated
    """
    args = parse_args()
    rng = random.Random(args.seed)
    selected: dict[str, list[Path]] = {}
    for layout_class in LAYOUT_CLASSES:
        files = sorted((args.layouts / layout_class).glob("*.txt"))
        if len(files) < N_LAYOUTS_PER_CLASS:
            raise ValueError(
                f"{layout_class!r} contains {len(files)} layouts; "
                f"{N_LAYOUTS_PER_CLASS} are required"
            )
        selected[layout_class] = rng.sample(files, N_LAYOUTS_PER_CLASS)

    figure, axes = plt.subplots(2, 5, figsize=(10, 4), squeeze=False)
    figure.subplots_adjust(
        left=0.06,
        right=0.995,
        bottom=0.01,
        top=0.99,
        wspace=0.02,
        hspace=0.02,
    )
    for row, layout_class in enumerate(LAYOUT_CLASSES):
        axes[row, 0].text(
            -0.08,
            0.5,
            layout_class.capitalize(),
            transform=axes[row, 0].transAxes,
            rotation=90,
            ha="right",
            va="center",
            fontsize=12,
            clip_on=False,
        )
        for column, layout_file in enumerate(selected[layout_class]):
            world = World.from_file(str(layout_file))
            world.reset()
            axes[row, column].imshow(world.get_image())
            axes[row, column].set_axis_off()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=200, bbox_inches="tight", pad_inches=0.02)
    plt.close(figure)


if __name__ == "__main__":
    main()
