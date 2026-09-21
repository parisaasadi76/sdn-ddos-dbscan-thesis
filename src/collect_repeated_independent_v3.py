#!/usr/bin/env python3
"""Collect additional independent SDN repetitions without overwriting raw data.

The clean v2 experiment is retained as repetition 01.  This script creates
repetitions 02 onward in separate directories.  Each repetition contains a
fresh Normal calibration run, a fresh Normal test run, and a fresh High-Rate
ICMP test run.  A temporary directory is renamed only after all three runs and
their quality checks complete successfully.
"""

from __future__ import annotations

import argparse
import os
import pwd
from pathlib import Path

import collect_independent_runs_v2 as validated_v2


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = BASE_DIR / "data" / "raw" / "independent_repeats"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect repeated independent Normal/High-Rate ICMP runs.",
    )
    parser.add_argument("--start-repeat", type=int, default=2)
    parser.add_argument("--end-repeat", type=int, default=5)
    parser.add_argument("--duration", type=int, default=120)
    parser.add_argument("--collector-interval", type=float, default=2.0)
    parser.add_argument("--controller-ip", default="127.0.0.1")
    parser.add_argument("--controller-port", type=int, default=6633)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def restore_tree_ownership(path: Path) -> None:
    invoking_uid = os.environ.get("SUDO_UID") or os.environ.get("PKEXEC_UID")
    if invoking_uid is None:
        return
    uid = int(invoking_uid)
    invoking_gid = os.environ.get("SUDO_GID")
    gid = int(invoking_gid) if invoking_gid is not None else pwd.getpwuid(uid).pw_gid
    os.chown(path, uid, gid)
    for child in path.rglob("*"):
        os.chown(child, uid, gid)


def run_specs(repeat_id: int):
    run_spec = validated_v2.implementation.RunSpec
    suffix = f"{repeat_id:02d}"
    return (
        run_spec(
            filename=f"calibration_normal_{suffix}.csv",
            scenario="normal",
            role="calibration",
            source_host="h1",
            destination_host="h2",
            ping_interval_seconds=1.0,
        ),
        run_spec(
            filename=f"test_normal_{suffix}.csv",
            scenario="normal",
            role="test",
            source_host="h3",
            destination_host="h2",
            ping_interval_seconds=1.0,
        ),
        run_spec(
            filename=f"test_high_rate_icmp_{suffix}.csv",
            scenario="high_rate",
            role="test",
            source_host="h1",
            destination_host="h2",
            ping_interval_seconds=0.1,
        ),
    )


def main() -> None:
    args = parse_args()
    implementation = validated_v2.implementation
    implementation.require_root()

    if args.start_repeat < 2:
        raise SystemExit("ERROR: repetition 01 is the preserved v2 experiment.")
    if args.end_repeat < args.start_repeat:
        raise SystemExit("ERROR: end-repeat must be greater than start-repeat.")

    args.output_root.mkdir(parents=True, exist_ok=True)
    restore_tree_ownership(args.output_root)

    check_args = argparse.Namespace(
        duration=args.duration,
        collector_interval=args.collector_interval,
        controller_ip=args.controller_ip,
        controller_port=args.controller_port,
    )
    implementation.check_requirements(check_args)

    repeat_ids = list(range(args.start_repeat, args.end_repeat + 1))
    print("Repeated independent SDN collection plan", flush=True)
    print(f"Repetitions: {repeat_ids}", flush=True)
    print(f"Three traffic runs per repetition, {args.duration}s each", flush=True)
    print(f"Collector interval: {args.collector_interval}s", flush=True)
    print(f"Output root: {args.output_root}", flush=True)

    for repeat_id in repeat_ids:
        final_dir = args.output_root / f"run_{repeat_id:02d}"
        temporary_dir = args.output_root / f".run_{repeat_id:02d}.inprogress"
        if final_dir.exists() or temporary_dir.exists():
            raise SystemExit(
                "ERROR: refusing to overwrite an existing repetition: "
                f"{final_dir if final_dir.exists() else temporary_dir}"
            )

        temporary_dir.mkdir(parents=True)
        restore_tree_ownership(temporary_dir)
        specs = run_specs(repeat_id)
        run_args = argparse.Namespace(
            output_dir=temporary_dir,
            duration=args.duration,
            collector_interval=args.collector_interval,
            controller_ip=args.controller_ip,
            controller_port=args.controller_port,
        )

        print(f"\n=== Repetition {repeat_id:02d} ===", flush=True)
        try:
            results = [
                validated_v2.collect_one_run(run_args, spec) for spec in specs
            ]
            implementation.write_manifest(temporary_dir, results)
            temporary_dir.rename(final_dir)
            restore_tree_ownership(final_dir)
        except BaseException:
            restore_tree_ownership(temporary_dir)
            print(
                f"Partial data retained for diagnosis: {temporary_dir}",
                flush=True,
            )
            raise
        print(f"Completed repetition {repeat_id:02d}: {final_dir}", flush=True)

    print("\nAll requested repetitions completed.", flush=True)


if __name__ == "__main__":
    main()
