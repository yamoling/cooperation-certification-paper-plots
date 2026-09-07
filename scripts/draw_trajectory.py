"""Draw solved LLE agent trajectories on top of a world image."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import lle
import matplotlib.pyplot as plt
from lle import Action
from matplotlib.axes import Axes
from matplotlib.figure import Figure

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


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    @ai-generated
    """
    parser = argparse.ArgumentParser(
        description="Solve an LLE world layout and draw each agent's trajectory."
    )
    parser.add_argument("layout", type=Path, help="Path to an LLE world layout file.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Path of the output image. Defaults to '<layout stem>-trajectory.png' next to the input file.",
    )
    parser.add_argument(
        "--labelled",
        action="store_true",
        help="Label visible trajectory points with their time step.",
    )
    parser.add_argument(
        "--first-label",
        action="store_true",
        help="Include the start position when drawing labels.",
    )
    parser.add_argument(
        "--no-path", action="store_true", help="Do not draw trajectory path lines."
    )
    parser.add_argument(
        "--bullet",
        action="store_true",
        help="Draw a marker at each visible trajectory point.",
    )
    parser.add_argument(
        "--final-arrow",
        action="store_true",
        help="Draw an arrow head on the final trajectory segment.",
    )
    parser.add_argument(
        "--t-max",
        type=int,
        default=None,
        help="Maximum solving horizon. Defaults to LLE's automatic solver horizon.",
    )
    parser.add_argument(
        "--trajectories",
        type=str,
        default=None,
        help=(
            "Draw these trajectories instead of solving the layout. Either a path to a JSON file or inline JSON, "
            "shaped as a list of per-agent lists of [row, col] positions."
        ),
    )
    return parser.parse_args()


def load_trajectories(source: str) -> list[Trajectory]:
    """Read explicit trajectories from a JSON file path or an inline JSON string.

    @ai-generated
    """
    path = Path(source)
    raw = path.read_text() if path.is_file() else source
    return [[(int(i), int(j)) for i, j in trajectory] for trajectory in json.loads(raw)]


def tile_centre(position: Position) -> tuple[float, float]:
    """Return the pixel centre of a grid position as an `(x, y)` point."""
    i, j = position
    return (
        j * TILE_STRIDE + GRID_SIZE + (TILE_SIZE - 1) / 2,
        i * TILE_STRIDE + GRID_SIZE + (TILE_SIZE - 1) / 2,
    )


def solve_world(world: lle.World, t_max: int | None | Literal["auto"]):
    """Solve `world` and return its shortest plan."""
    if t_max is None:
        t_max = "auto"
    plan = lle.solve(world, t_max, shuffle=True)
    if plan is None:
        horizon = "automatic horizon" if t_max is None else f"t_max={t_max}"
        raise SystemExit(f"No solution found with {horizon}.")
    plan = [list(a) for a in plan]

    def replace(agent: int, actions: list[Action]):
        for a_t, replacement in zip(plan, actions):
            a_t[agent] = replacement

    return plan


def compute_trajectories(
    world: lle.World, plan: Sequence[Sequence[lle.Action]]
) -> list[Trajectory]:
    """Replay a plan and collect every agent position at each time step."""
    world.reset()
    trajectories: list[Trajectory] = [[position] for position in world.agents_positions]
    for joint_action in plan:
        world.step(joint_action)
        for agent_id, position in enumerate(world.agents_positions):
            trajectories[agent_id].append(position)
    return trajectories


def visible_label_steps(trajectory: Sequence[Position]) -> set[int]:
    """Return entry steps to label, hiding repeated positions.

    For a run of repeated positions, this keeps the step at which the agent entered the tile.

    @ai-generated
    """
    visible: set[int] = set()
    start = 0
    for step in range(1, len(trajectory) + 1):
        if step == len(trajectory) or trajectory[step] != trajectory[start]:
            visible.add(start)
            start = step
    return visible


def extend_point(
    start: tuple[float, float], end: tuple[float, float], distance: float
) -> tuple[float, float]:
    """Return `end` moved farther away from `start` by `distance` pixels.

    @ai-generated
    """
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length = (dx * dx + dy * dy) ** 0.5
    if length == 0:
        return end
    return (end[0] + dx / length * distance, end[1] + dy / length * distance)


def drawn_label_steps(trajectory: Sequence[Position], *, first_label: bool) -> set[int]:
    """Return visible label steps, optionally including the start position.

    @ai-generated
    """
    steps = visible_label_steps(trajectory)
    if first_label:
        return steps
    return steps - {0}


def crowded_label_positions(
    trajectories: Sequence[Sequence[Position]], *, first_label: bool
) -> set[Position]:
    """Return cells that contain visible labels from multiple agents.

    @ai-generated
    """
    labelled_agents: dict[Position, set[int]] = {}
    for agent_id, trajectory in enumerate(trajectories):
        for step in drawn_label_steps(trajectory, first_label=first_label):
            labelled_agents.setdefault(trajectory[step], set()).add(agent_id)
    return {
        position
        for position, agent_ids in labelled_agents.items()
        if len(agent_ids) > 1
    }


def label_offset(
    agent_id: int, position: Position, crowded_positions: set[Position]
) -> tuple[int, int]:
    """Return the label offset for an agent at a labelled position.

    @ai-generated
    """
    if position not in crowded_positions:
        return (0, 0)
    return LABEL_OFFSETS[agent_id % len(LABEL_OFFSETS)]


def draw_trajectory(
    ax: Axes,
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
    """Draw one agent trajectory and optional time-step labels.

    @ai-generated
    """
    points = [tile_centre(position) for position in trajectory]
    final_step = len(points) - 1
    while final_step > 0 and points[final_step - 1] == points[-1]:
        final_step -= 1

    path_points = points[:final_step] if final_arrow else points[: final_step + 1]
    if path_points:
        xs = [point[0] for point in path_points]
        ys = [point[1] for point in path_points]
        if not hide_path:
            ax.plot(xs, ys, color=colour, linewidth=3, solid_capstyle="round", zorder=3)
        if bullet:
            ax.plot(
                xs, ys, color=colour, linestyle="", marker="o", markersize=5, zorder=3
            )

    if final_arrow and final_step > 0:
        departure = points[final_step - 1]
        arrival = points[-1]
        arrow_tip = extend_point(departure, arrival, ARROW_TIP_OVERSHOOT)
        ax.annotate(
            "",
            xy=arrow_tip,
            xytext=departure,
            arrowprops={
                "arrowstyle": "-|>",
                "color": colour,
                "lw": 3,
                "mutation_scale": 12,
                "shrinkB": 0,
            },
            # zorder=5,
        )

    if not labelled:
        return

    steps_by_position: dict[Position, list[int]] = {}
    for step in sorted(drawn_label_steps(trajectory, first_label=first_label)):
        steps_by_position.setdefault(trajectory[step], []).append(step)

    for position, steps in steps_by_position.items():
        x, y = tile_centre(position)
        dx, dy = label_offset(agent_id, position, crowded_positions)
        ax.text(
            x + dx,
            y + dy,
            ",".join(str(step) for step in steps),
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
    """Create the trajectory figure for solved agents.

    @ai-generated
    """
    image = world.get_image()
    height, width = image.shape[:2]
    fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)
    ax.imshow(image)
    ax.set_axis_off()
    ax.set_xlim(-0.5, width - 0.5)
    ax.set_ylim(height - 0.5, -0.5)

    visible_trajectories = trajectories[: len(AGENT_COLOURS)]
    crowded_positions = (
        crowded_label_positions(visible_trajectories, first_label=first_label)
        if labelled
        else set()
    )

    for agent_id, trajectory in enumerate(visible_trajectories):
        draw_trajectory(
            ax,
            trajectory,
            AGENT_COLOURS[agent_id],
            agent_id=agent_id,
            labelled=labelled,
            first_label=first_label,
            crowded_positions=crowded_positions,
            hide_path=hide_path,
            bullet=bullet,
            final_arrow=final_arrow,
        )

    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    return fig


def main() -> None:
    """Load, solve, draw, and save an LLE world trajectory.

    @ai-generated
    """
    args = parse_args()
    output = args.output or args.layout.with_name(f"{args.layout.stem}-trajectory.png")

    world = lle.World.from_file(args.layout)
    if args.trajectories is None:
        plan = solve_world(world, args.t_max)
        trajectories = compute_trajectories(world, plan)
        n_steps = len(plan)
    else:
        trajectories = load_trajectories(args.trajectories)
        n_steps = max(len(trajectory) for trajectory in trajectories) - 1
    world.reset()
    fig = draw_trajectories(
        world,
        trajectories,
        labelled=args.labelled,
        first_label=args.first_label,
        hide_path=args.no_path,
        bullet=args.bullet,
        final_arrow=args.final_arrow,
    )
    fig.savefig(output, dpi=fig.dpi)
    plt.close(fig)
    print(f"Saved trajectory with {n_steps} steps to {output}")


if __name__ == "__main__":
    main()
