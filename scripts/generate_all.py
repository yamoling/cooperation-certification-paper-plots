"""Regenerate the paper's plots, tables and derived data from `logs/` and `layouts/`.

Run with no arguments to regenerate everything, or name one or more targets to
regenerate only those (see `--list` for the full set):

    uv run python scripts/generate_all.py
    uv run python scripts/generate_all.py coop-profiles laser-colour
    uv run python scripts/generate_all.py --list

Each target shells out to the corresponding script under `scripts/` with the
argument combination used to produce the paper's actual figures/tables, so it
stays a thin, auditable wrapper rather than a second copy of the logic.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
N_WORKERS = os.cpu_count() or 1

CANONICAL_POOLS = (
    "asymmetric",
    "convergent-2",
    "divergent-2",
    "sequential-2",
    "interdependent-2",
)


def run(*args: str) -> None:
    """Run one `scripts/*.py` invocation with the current interpreter, from ROOT.

    @ai-generated
    """
    command = [sys.executable, str(SCRIPTS / args[0]), *args[1:]]
    print(f"$ {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def copy(source: str, *destinations: str) -> None:
    """Copy a generated file to one or more paths expected by the LaTeX project.

    @ai-generated
    """
    for destination in destinations:
        target = ROOT / destination
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / source, target)
        print(f"copied {source} -> {destination}")


def target_pool_characteristics(layout_limit: int) -> None:
    """Characterize each canonical layout pool (up to `layout_limit` layouts each,
    or every layout when `layout_limit` is 0), then render the summary table.

    @ai-generated
    """
    output = "data/layout-characteristics.csv"
    for index, pool in enumerate(CANONICAL_POOLS):
        args = [
            "pool-characteristics.py",
            f"layouts/canonical/{pool}",
            "--n-workers",
            str(N_WORKERS),
            "--output",
            output,
        ]
        if layout_limit > 0:
            args += ["--limit", str(layout_limit)]
        if index == 0:
            args.append("--overwrite")
        run(*args)
    run(
        "pool-characteristics.py",
        "--table",
        "--output",
        output,
        "--table-output",
        "tables/pool_characteristics_table.tex",
    )


def target_sat_duration() -> None:
    """Plot the SAT horizon sweep and copy the paper's figure into latex/plots.

    @ai-generated
    """
    run("plot_sat_measurements.py", "--compare-shuffled")
    copy(
        "plots/solving_duration-comparison.pdf",
        "latex/plots/solving_duration-comparison.pdf",
    )
    copy(
        "plots/solving_duration-comparison.png",
        "latex/plots/solving_duration-comparison.png",
    )


# Each target's help text plus the actions that reproduce it, one per action
# taking the parsed CLI args (most ignore it). Ordered roughly by how the
# corresponding results appear in the paper.
TARGETS: dict[str, tuple[str, Sequence]] = {
    "coop-profiles": (
        "Cooperation-profile sweep: 3 figures + the algorithm/profile table + ~30 CSVs.",
        [lambda args: run("coop_profiles.py")],
    ),
    "coop-profile-certificates": (
        "Certificate/theorem validation and outcome-conditioned predicate tables.",
        [
            lambda args: run("coop_profile_certificates.py"),
            lambda args: run("coop_profile_certificates.py", "--endpoint-only"),
        ],
    ),
    "coop-profile-failures": (
        "Death/timeout/joint-success breakdown table (tab:coop-profile-failures).",
        [lambda args: run("coop_profile_failures_table.py")],
    ),
    "laser-colour": (
        "Laser-colour-matching effect on agent exits.",
        [lambda args: run("coop_profile_laser_colour.py")],
    ),
    "pool-characteristics": (
        (
            "Cooperation-predicate incidence for each canonical layout pool "
            "(--layout-limit layouts per pool by default)."
        ),
        [lambda args: target_pool_characteristics(args.layout_limit)],
    ),
    "layouts-picture": (
        "Sampled layouts from each canonical profile pool (pictures/layouts.pdf).",
        [lambda args: run("render_layouts.py")],
    ),
    "generalization-layouts-picture": (
        "Sampled 5x5 independent/cooperative layouts (pictures/generalization-layouts-5x5.pdf).",
        [lambda args: run("render_generalization_layouts.py")],
    ),
    "generalization": (
        "5x5 pool-size generalization figure (plots/generalization-joint-5x5.pdf).",
        [lambda args: run("plot_generalization.py", "--joint")],
    ),
    "sat-duration": (
        "SAT construction/solving duration vs. horizon (plots/solving_duration-comparison.pdf).",
        [lambda args: target_sat_duration()],
    ),
    "acceptance-rates": (
        "Certification acceptance-rate/timing plots and the certification-yield table.",
        [lambda args: run("plot_acceptance_rates.py")],
    ),
    "exit-outcomes": (
        "5x5 sweep exit-outcome table (tab:exit-outcomes).",
        [lambda args: run("exit_outcomes_table.py")],
    ),
    "trajectory-profiles": (
        "Per-episode cooperation-predicate rates backing the results-section prose.",
        [lambda args: run("trajectory_profiles.py")],
    ),
    "exit-rate-distribution": (
        "5x5 sweep none/single/joint exit-rate distribution report.",
        [lambda args: run("exit_rate_distribution.py")],
    ),
}


def parse_args() -> argparse.Namespace:
    """Parse the optional target selection and the layout-characterization limit.

    @ai-generated
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "targets",
        nargs="*",
        choices=list(TARGETS),
        default=[],
        metavar="TARGET",
        help="One or more targets to (re)generate. Default: every target.",
    )
    parser.add_argument(
        "--list", action="store_true", help="List available targets and exit."
    )
    parser.add_argument(
        "--layout-limit",
        type=int,
        default=1000,
        help=(
            "Characterize only the first N layouts of each canonical pool in the "
            "pool-characteristics target (default: %(default)s). Pass 0 to "
            "characterize every layout of every pool."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Regenerate the selected (or every) target, in dependency-friendly order.

    @ai-generated
    """
    args = parse_args()
    if args.list:
        width = max(len(name) for name in TARGETS)
        for name, (help_text, _) in TARGETS.items():
            print(f"{name:<{width}}  {help_text}")
        return

    selected = args.targets or list(TARGETS)
    for index, name in enumerate(selected, start=1):
        print(f"\n=== [{index}/{len(selected)}] {name} ===")
        for action in TARGETS[name][1]:
            action(args)
    print(f"\nDone: {', '.join(selected)}")


if __name__ == "__main__":
    main()
