"""Layout generation, characterization, extraction, and rendering."""

from __future__ import annotations

import csv
import json
import random
import re
from collections.abc import Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from pathlib import Path
from typing import Literal

import lle
import matplotlib.pyplot as plt
import polars as pl
from lle import Action, World, characterize
from lle.generator import Independent
from matplotlib.axes import Axes
from matplotlib.figure import Figure
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

LAYOUT_FAMILIES = (
    "asymmetric",
    "convergent-2",
    "divergent-2",
    "sequential-2",
    "interdependent-2",
)
LAYOUT_CLASSES = ("independent", "cooperative")
LASER_PATTERN = re.compile(r"\bL([0-9])\w*\b")
LAYOUT_INDEX_PATTERN = re.compile(r"(\d+)$")
LASER_CSV_FIELDS = ("index", "laser-0", "laser-1", "laser-2")

Position = tuple[int, int]
Trajectory = list[Position]
AGENT_COLOURS = ["#e41a1c", "#ffd92f", "#4daf4a", "#2454a6"]
TILE_SIZE = 32
GRID_SIZE = 1
TILE_STRIDE = TILE_SIZE
ARROW_TIP_OVERSHOOT = 4.0
LABEL_OFFSET = 7
LABEL_OFFSETS = [
    (-LABEL_OFFSET, -LABEL_OFFSET),
    (LABEL_OFFSET, -LABEL_OFFSET),
    (-LABEL_OFFSET, LABEL_OFFSET),
    (LABEL_OFFSET, LABEL_OFFSET),
]


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


def generate_independent_layouts(
    output: Path,
    *,
    t_max: int = 81,
    n_jobs: int = 1,
    seed: int | None = None,
    total: int = 10_000,
) -> None:
    """Fill a directory with unique layouts satisfying ``Independent``."""
    output.mkdir(parents=True, exist_ok=True)
    existing_paths = [path for path in output.iterdir() if path.is_file()]
    contents = {path.read_text(encoding="utf-8") for path in existing_paths}
    if len(contents) != len(existing_paths):
        raise ValueError(f"Duplicate layout already exists in {output}")
    remaining = total - len(existing_paths)
    if remaining <= 0:
        print(f"{output} already contains {len(existing_paths)} layouts")
        return
    first_index = len(existing_paths)
    output_paths = [
        output / f"{index:04d}.txt"
        for index in range(first_index, first_index + remaining)
    ]
    collision = next((path for path in output_paths if path.exists()), None)
    if collision is not None:
        raise FileExistsError(f"Refusing to overwrite existing layout: {collision}")
    generator = (
        lle.generate(width=9, height=9, n_agents=3, t_max=t_max)
        .lasers(2)
        .require(Independent())
    )
    worlds = generator.take(remaining, n_jobs=max(n_jobs, 1), seed=seed, progress=True)
    for output_path, world in zip(output_paths, worlds, strict=True):
        content = f"{world.world_string}\n"
        if content in contents:
            raise ValueError("Generator produced a duplicate layout")
        contents.add(content)
        output_path.write_text(content, encoding="utf-8")
    print(f"Wrote {remaining} layouts to {output}")


def get_layout_index(layout_path: Path) -> int:
    match = LAYOUT_INDEX_PATTERN.search(layout_path.stem)
    if match is None:
        raise ValueError(f"Layout filename has no numeric index: {layout_path}")
    return int(match.group(1))


def extract_laser_counts(layout_dir: Path) -> list[tuple[int, int, int, int]]:
    """Count laser colours in layout-loader order."""
    paths = sorted(
        (path for path in layout_dir.glob("*") if path.is_file()),
        key=get_layout_index,
    )
    if not paths:
        raise FileNotFoundError(f"No layout files found in {layout_dir}")
    rows: list[tuple[int, int, int, int]] = []
    for index, path in enumerate(paths):
        counts = [0, 0, 0]
        for colour in LASER_PATTERN.findall(path.read_text()):
            colour_index = int(colour)
            if colour_index >= len(counts):
                raise ValueError(f"Unsupported laser colour L{colour_index} in {path}")
            counts[colour_index] += 1
        rows.append((index, *counts))
    return rows


def write_laser_counts(rows: list[tuple[int, int, int, int]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(LASER_CSV_FIELDS)
        writer.writerows(rows)


def render_canonical_layouts(
    layouts: Path,
    output: Path,
    *,
    seed: int | None = None,
    transpose: bool = False,
) -> None:
    """Render five sampled layouts from each canonical family."""
    rng = random.Random(seed)
    selected: list[list[Path]] = []
    for family in LAYOUT_FAMILIES:
        files = sorted((layouts / family).glob("*.txt"))
        if len(files) < 5:
            raise ValueError(
                f"{family!r} contains {len(files)} layouts; 5 are required"
            )
        selected.append(rng.sample(files, 5))
    figure, axes = plt.subplots(5, 5, figsize=(10, 10), squeeze=False)
    figure.subplots_adjust(
        left=0.04 if transpose else 0.005,
        right=0.995,
        bottom=0.005,
        top=0.95,
        wspace=0.01,
        hspace=0.01,
    )
    if transpose:
        for row, (family, family_layouts) in enumerate(zip(LAYOUT_FAMILIES, selected)):
            axes[row, 0].text(
                -0.04,
                0.5,
                family.replace("-", " ").title(),
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
            zip(LAYOUT_FAMILIES, selected)
        ):
            axes[0, column].set_title(
                family.replace("-", " ").title(), fontsize=14, pad=5
            )
            for row, layout_file in enumerate(family_layouts):
                axis = axes[row, column]
                world = World.from_file(str(layout_file))
                world.reset()
                axis.imshow(world.get_image())
                axis.set_axis_off()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=200, bbox_inches="tight", pad_inches=0.02)
    plt.close(figure)


def render_generalization_layouts(
    layouts: Path, output: Path, *, seed: int = 42
) -> None:
    """Render five sampled layouts from each 5x5 layout class."""
    rng = random.Random(seed)
    selected: dict[str, list[Path]] = {}
    for layout_class in LAYOUT_CLASSES:
        files = sorted((layouts / layout_class).glob("*.txt"))
        if len(files) < 5:
            raise ValueError(
                f"{layout_class!r} contains {len(files)} layouts; 5 are required"
            )
        selected[layout_class] = rng.sample(files, 5)
    figure, axes = plt.subplots(2, 5, figsize=(10, 4), squeeze=False)
    figure.subplots_adjust(
        left=0.06, right=0.995, bottom=0.01, top=0.99, wspace=0.02, hspace=0.02
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
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=200, bbox_inches="tight", pad_inches=0.02)
    plt.close(figure)


def load_trajectories(source: str) -> list[Trajectory]:
    path = Path(source)
    raw = path.read_text() if path.is_file() else source
    return [
        [(int(row), int(column)) for row, column in trajectory]
        for trajectory in json.loads(raw)
    ]


def tile_centre(position: Position) -> tuple[float, float]:
    row, column = position
    return (
        column * TILE_STRIDE + GRID_SIZE + (TILE_SIZE - 1) / 2,
        row * TILE_STRIDE + GRID_SIZE + (TILE_SIZE - 1) / 2,
    )


def solve_world(
    world: lle.World, t_max: int | None | Literal["auto"]
) -> list[list[Action]]:
    plan = lle.solve(world, "auto" if t_max is None else t_max, shuffle=True)
    if plan is None:
        raise ValueError(f"No solution found with t_max={t_max or 'auto'}")
    return [list(actions) for actions in plan]


def compute_trajectories(
    world: lle.World, plan: Sequence[Sequence[Action]]
) -> list[Trajectory]:
    world.reset()
    trajectories: list[Trajectory] = [[position] for position in world.agents_positions]
    for joint_action in plan:
        world.step(joint_action)
        for agent_id, position in enumerate(world.agents_positions):
            trajectories[agent_id].append(position)
    return trajectories


def visible_label_steps(trajectory: Sequence[Position]) -> set[int]:
    visible: set[int] = set()
    start = 0
    for step in range(1, len(trajectory) + 1):
        if step == len(trajectory) or trajectory[step] != trajectory[start]:
            visible.add(start)
            start = step
    return visible


def drawn_label_steps(trajectory: Sequence[Position], *, first_label: bool) -> set[int]:
    steps = visible_label_steps(trajectory)
    return steps if first_label else steps - {0}


def crowded_label_positions(
    trajectories: Sequence[Sequence[Position]], *, first_label: bool
) -> set[Position]:
    labelled_agents: dict[Position, set[int]] = {}
    for agent_id, trajectory in enumerate(trajectories):
        for step in drawn_label_steps(trajectory, first_label=first_label):
            labelled_agents.setdefault(trajectory[step], set()).add(agent_id)
    return {position for position, agents in labelled_agents.items() if len(agents) > 1}


def extend_point(
    start: tuple[float, float], end: tuple[float, float], distance: float
) -> tuple[float, float]:
    dx, dy = end[0] - start[0], end[1] - start[1]
    length = (dx * dx + dy * dy) ** 0.5
    if length == 0:
        return end
    return end[0] + dx / length * distance, end[1] + dy / length * distance


def draw_trajectory(
    axis: Axes,
    trajectory: Sequence[Position],
    colour: str,
    *,
    agent_id: int,
    labelled: bool,
    first_label: bool,
    crowded_positions: set[Position],
    hide_path: bool,
    bullet: bool,
    final_arrow: bool,
) -> None:
    points = [tile_centre(position) for position in trajectory]
    final_step = len(points) - 1
    while final_step > 0 and points[final_step - 1] == points[-1]:
        final_step -= 1
    path_points = points[:final_step] if final_arrow else points[: final_step + 1]
    if path_points:
        xs, ys = zip(*path_points, strict=True)
        if not hide_path:
            axis.plot(
                xs, ys, color=colour, linewidth=3, solid_capstyle="round", zorder=3
            )
        if bullet:
            axis.plot(
                xs, ys, color=colour, linestyle="", marker="o", markersize=5, zorder=3
            )
    if final_arrow and final_step > 0:
        departure = points[final_step - 1]
        axis.annotate(
            "",
            xy=extend_point(departure, points[-1], ARROW_TIP_OVERSHOOT),
            xytext=departure,
            arrowprops={
                "arrowstyle": "-|>",
                "color": colour,
                "lw": 3,
                "mutation_scale": 12,
                "shrinkB": 0,
            },
        )
    if not labelled:
        return
    steps_by_position: dict[Position, list[int]] = {}
    for step in sorted(drawn_label_steps(trajectory, first_label=first_label)):
        steps_by_position.setdefault(trajectory[step], []).append(step)
    for position, steps in steps_by_position.items():
        x, y = tile_centre(position)
        dx, dy = (
            LABEL_OFFSETS[agent_id % len(LABEL_OFFSETS)]
            if position in crowded_positions
            else (0, 0)
        )
        axis.text(
            x + dx,
            y + dy,
            ",".join(map(str, steps)),
            color=colour,
            fontsize=8,
            fontweight="bold",
            ha="center",
            va="center",
            bbox={
                "boxstyle": "round,pad=0.12",
                "facecolor": "white",
                "edgecolor": colour,
                "alpha": 0.85,
            },
            zorder=4,
        )


def draw_trajectories(
    world: lle.World,
    trajectories: Sequence[Sequence[Position]],
    *,
    labelled: bool,
    first_label: bool,
    hide_path: bool,
    bullet: bool,
    final_arrow: bool,
) -> Figure:
    image = world.get_image()
    height, width = image.shape[:2]
    figure, axis = plt.subplots(figsize=(width / 100, height / 100), dpi=100)
    axis.imshow(image)
    axis.set_axis_off()
    axis.set_xlim(-0.5, width - 0.5)
    axis.set_ylim(height - 0.5, -0.5)
    visible = trajectories[: len(AGENT_COLOURS)]
    crowded = (
        crowded_label_positions(visible, first_label=first_label) if labelled else set()
    )
    for agent_id, trajectory in enumerate(visible):
        draw_trajectory(
            axis,
            trajectory,
            AGENT_COLOURS[agent_id],
            agent_id=agent_id,
            labelled=labelled,
            first_label=first_label,
            crowded_positions=crowded,
            hide_path=hide_path,
            bullet=bullet,
            final_arrow=final_arrow,
        )
    figure.subplots_adjust(left=0, right=1, top=1, bottom=0)
    return figure


def render_trajectory(
    layout: Path,
    output: Path,
    *,
    trajectories: str | None = None,
    t_max: int | None = None,
    labelled: bool = False,
    first_label: bool = False,
    hide_path: bool = False,
    bullet: bool = False,
    final_arrow: bool = False,
) -> None:
    world = lle.World.from_file(layout)
    paths = (
        compute_trajectories(world, solve_world(world, t_max))
        if trajectories is None
        else load_trajectories(trajectories)
    )
    world.reset()
    figure = draw_trajectories(
        world,
        paths,
        labelled=labelled,
        first_label=first_label,
        hide_path=hide_path,
        bullet=bullet,
        final_arrow=final_arrow,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=figure.dpi)
    plt.close(figure)
