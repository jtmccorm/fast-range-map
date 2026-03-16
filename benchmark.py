#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_SCRIPT = REPO_ROOT / "maritime_reach_map.py"
DEFAULT_OUTDIR = REPO_ROOT / "benchmarks"
DEFAULT_RESULTS = "runtime_results.csv"
DEFAULT_STEPS = (12.0, 8.0, 6.0, 4.0, 2.0)
DEFAULT_HUBS = ((14.829, 120.283),)
DEFAULT_RANGE_NM = 2000.0
LAT_KM = 60 * 111
LON_KM = 100 * 111

MEASURE_CHILD_CODE = """
import resource
import subprocess
import sys
import time

command = sys.argv[1:]
started = time.perf_counter()
completed = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
elapsed = time.perf_counter() - started

if completed.returncode != 0:
    sys.stderr.write(completed.stderr)
    sys.exit(completed.returncode)

max_rss = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
if sys.platform == "darwin":
    max_rss //= 1024

print(f"{elapsed:.3f},{max_rss}")
""".strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark maritime_reach_map.py across grid resolutions.")
    parser.add_argument(
        "--script",
        type=Path,
        default=DEFAULT_SCRIPT,
        help=f"Path to the routing script. Default: {DEFAULT_SCRIPT.name}.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=DEFAULT_OUTDIR,
        help=f"Directory for benchmark PNGs and CSV output. Default: {DEFAULT_OUTDIR.name}/.",
    )
    parser.add_argument(
        "--results",
        default=DEFAULT_RESULTS,
        help=f"Results CSV filename inside --outdir. Default: {DEFAULT_RESULTS}.",
    )
    parser.add_argument(
        "--range-nm",
        type=float,
        default=DEFAULT_RANGE_NM,
        help=f"Range used for benchmarking. Default: {DEFAULT_RANGE_NM:.0f}.",
    )
    parser.add_argument(
        "--step-km",
        dest="steps",
        type=float,
        nargs="+",
        default=list(DEFAULT_STEPS),
        help="Grid resolutions in kilometers. Default: 12 8 6 4 2.",
    )
    parser.add_argument(
        "--hub",
        action="append",
        nargs=2,
        metavar=("LAT", "LON"),
        type=float,
        help="Hub latitude and longitude. Repeat for multiple hubs.",
    )
    return parser.parse_args()


def format_step(step: float) -> str:
    return str(int(step)) if float(step).is_integer() else str(step)


def build_hub_args(hubs: list[list[float]] | None) -> list[str]:
    if not hubs:
        hubs = [list(coords) for coords in DEFAULT_HUBS]

    hub_args: list[str] = []
    for lat, lon in hubs:
        hub_args.extend(["--hub", str(lat), str(lon)])
    return hub_args


def estimate_grid_cells(step_km: float) -> int:
    grid_y = int(math.ceil(LAT_KM / step_km))
    grid_x = int(math.ceil(LON_KM / step_km))
    return grid_x * grid_y


def measure_run(
    script_path: Path,
    range_nm: float,
    step_km: float,
    output_png: Path,
    hub_args: list[str],
) -> tuple[str, str]:
    command = [
        sys.executable,
        str(script_path),
        *hub_args,
        "--range-nm",
        str(range_nm),
        "--step-km",
        str(step_km),
        "--output",
        str(output_png),
    ]
    measurement = subprocess.run(
        [sys.executable, "-c", MEASURE_CHILD_CODE, *command],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if measurement.returncode != 0:
        if measurement.stderr:
            sys.stderr.write(measurement.stderr)
        raise SystemExit(measurement.returncode)

    elapsed, max_memory_kb = measurement.stdout.strip().split(",", maxsplit=1)
    if not elapsed or not max_memory_kb:
        raise SystemExit(f"Failed to parse timing output for step_km={format_step(step_km)}")
    return elapsed, max_memory_kb


def main() -> int:
    args = parse_args()
    script_path = args.script if args.script.is_absolute() else (REPO_ROOT / args.script)
    outdir = args.outdir if args.outdir.is_absolute() else (REPO_ROOT / args.outdir)
    results_path = outdir / args.results
    hub_args = build_hub_args(args.hub)

    outdir.mkdir(parents=True, exist_ok=True)
    results_path.write_text("step_km,grid_cells,elapsed_seconds,max_memory_kb\n", encoding="ascii")

    print("Running routing benchmarks...")
    for step_km in args.steps:
        step_label = format_step(step_km)
        print("--------------------------------------")
        print(f"Benchmarking step_km={step_label}")

        output_png = outdir / f"benchmark_{step_label}km.png"
        elapsed, max_memory_kb = measure_run(script_path, args.range_nm, step_km, output_png, hub_args)
        grid_cells = estimate_grid_cells(step_km)

        with results_path.open("a", encoding="ascii") as results_file:
            results_file.write(f"{step_label},{grid_cells},{elapsed},{max_memory_kb}\n")

    print()
    print("Benchmark complete.")
    try:
        display_results_path = results_path.relative_to(REPO_ROOT)
    except ValueError:
        display_results_path = results_path
    print(f"Results saved to: {display_results_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
