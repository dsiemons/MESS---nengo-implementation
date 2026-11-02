#!/usr/bin/env python3
"""Run the RCPSP project scripts in their required order.

Usage:
    python run_all.py
    python run_all.py --project-dir path/to/project
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


SCRIPT_ORDER = (
    "SGS.py",
    "Instance_generation.py",
    "batchRCPSP_Experiments.py",
    "analysis.py",
    "ClarityofOutput_test.py",
    "test_BG_input.py",
    "violations.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run all RCPSP scripts sequentially."
    )
    parser.add_argument(
        "--project-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory containing the project scripts (default: this file's directory).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_dir = args.project_dir.expanduser().resolve()

    if not project_dir.is_dir():
        print(f"Error: project directory does not exist: {project_dir}", file=sys.stderr)
        return 2

    missing_scripts = [
        script for script in SCRIPT_ORDER if not (project_dir / script).is_file()
    ]
    if missing_scripts:
        print("Error: the following required scripts were not found:", file=sys.stderr)
        for script in missing_scripts:
            print(f"  - {project_dir / script}", file=sys.stderr)
        return 2

    print(f"Project directory: {project_dir}")
    print(f"Python interpreter: {sys.executable}")
    print("Scripts will run sequentially; execution stops on the first failure.")

    for index, script in enumerate(SCRIPT_ORDER, start=1):
        script_path = project_dir / script
        print(f"\n[{index}/{len(SCRIPT_ORDER)}] Running {script}", flush=True)

        try:
            subprocess.run(
                [sys.executable, str(script_path)],
                cwd=project_dir,
                check=True,
            )
        except subprocess.CalledProcessError as exc:
            print(
                f"\nError: {script} failed with exit code {exc.returncode}. "
                "The remaining scripts were not run.",
                file=sys.stderr,
            )
            return exc.returncode or 1
        except KeyboardInterrupt:
            print("\nExecution interrupted by user.", file=sys.stderr)
            return 130

        print(f"[{index}/{len(SCRIPT_ORDER)}] Completed {script}", flush=True)

    print("\nAll scripts completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
