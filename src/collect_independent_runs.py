#!/usr/bin/env python3

"""Collect three isolated Mininet runs for independent DBSCAN evaluation.

Run this script with sudo. It connects each fresh Mininet topology to the
already-running POX controller on 127.0.0.1:6633, polls switch s1 through the
existing flow_collector.py, and never overwrites an existing raw data file.
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import signal
import socket
import subprocess
import time
from dataclasses import dataclass
from functools import partial
from pathlib import Path

from mininet.clean import cleanup
from mininet.net import Mininet
from mininet.node import OVSSwitch, RemoteController
from mininet.topo import SingleSwitchTopo


BASE_DIR = Path(__file__).resolve().parents[1]
COLLECTOR_PATH = BASE_DIR / "flow_collector.py"
DEFAULT_OUTPUT_DIR = BASE_DIR / "data" / "raw" / "independent"


@dataclass(frozen=True)
class RunSpec:
    filename: str
    scenario: str
    role: str
    source_host: str
    destination_host: str
    ping_interval_seconds: float


RUN_SPECS = (
    RunSpec(
        filename="calibration_normal_01.csv",
        scenario="normal",
        role="calibration",
        source_host="h1",
        destination_host="h2",
        ping_interval_seconds=1.0,
    ),
    RunSpec(
        filename="test_normal_01.csv",
        scenario="normal",
        role="test",
        source_host="h3",
        destination_host="h2",
        ping_interval_seconds=1.0,
    ),
    RunSpec(
        filename="test_high_rate_icmp_01.csv",
        scenario="high_rate",
        role="test",
        source_host="h1",
        destination_host="h2",
        ping_interval_seconds=0.1,
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect isolated Normal and High-Rate ICMP Mininet runs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=120,
        help="Traffic duration per run in seconds (default: 120).",
    )
    parser.add_argument(
        "--collector-interval",
        type=float,
        default=5.0,
        help="Flow polling interval in seconds (default: 5).",
    )
    parser.add_argument("--controller-ip", default="127.0.0.1")
    parser.add_argument("--controller-port", type=int, default=6633)
    return parser.parse_args()


def require_root() -> None:
    if os.geteuid() != 0:
        raise SystemExit(
            "ERROR: root access is required. Run with: "
            "sudo python3 src/collect_independent_runs.py"
        )


def check_requirements(args: argparse.Namespace) -> None:
    if args.duration < 30:
        raise SystemExit("ERROR: duration must be at least 30 seconds per run.")
    if args.collector_interval <= 0:
        raise SystemExit("ERROR: collector interval must be positive.")
    if not COLLECTOR_PATH.exists():
        raise SystemExit(f"ERROR: collector not found: {COLLECTOR_PATH}")
    if shutil.which("ovs-ofctl") is None:
        raise SystemExit("ERROR: ovs-ofctl is not available.")

    try:
        with socket.create_connection(
            (args.controller_ip, args.controller_port),
            timeout=3,
        ):
            pass
    except OSError as error:
        raise SystemExit(
            "ERROR: POX controller is not reachable at "
            f"{args.controller_ip}:{args.controller_port}: {error}"
        )


def ensure_new_targets(output_dir: Path) -> None:
    targets = [output_dir / spec.filename for spec in RUN_SPECS]
    targets.extend(
        output_dir / f"{Path(spec.filename).stem}.collector.log"
        for spec in RUN_SPECS
    )
    targets.append(output_dir / "run_manifest.csv")
    existing = [path for path in targets if path.exists()]
    if existing:
        listing = "\n".join(f"  {path}" for path in existing)
        raise SystemExit(
            "ERROR: refusing to overwrite existing experiment files:\n" + listing
        )


def start_network(controller_ip: str, controller_port: int) -> Mininet:
    topology = SingleSwitchTopo(k=3)
    switch_class = partial(OVSSwitch, protocols="OpenFlow10")
    network = Mininet(
        topo=topology,
        controller=None,
        switch=switch_class,
        autoSetMacs=True,
        build=True,
    )
    network.addController(
        "c0",
        controller=RemoteController,
        ip=controller_ip,
        port=controller_port,
    )
    network.start()
    return network


def stop_process(process: subprocess.Popen[bytes] | None) -> None:
    if process is None or process.poll() is not None:
        return
    process.send_signal(signal.SIGINT)
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


def count_csv_rows(path: Path) -> tuple[int, int, int]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))

    icmp = 0
    arp = 0
    for row in rows:
        match_value = (row.get("match") or "").lower()
        if "icmp" in match_value:
            icmp += 1
        elif "arp" in match_value:
            arp += 1
    return len(rows), icmp, arp


def restore_user_ownership(path: Path) -> None:
    sudo_uid = os.environ.get("SUDO_UID")
    sudo_gid = os.environ.get("SUDO_GID")
    if sudo_uid is None or sudo_gid is None:
        return
    os.chown(path, int(sudo_uid), int(sudo_gid))


def collect_one_run(
    args: argparse.Namespace,
    spec: RunSpec,
) -> dict[str, object]:
    output_path = args.output_dir / spec.filename
    log_path = args.output_dir / f"{output_path.stem}.collector.log"
    network: Mininet | None = None
    collector: subprocess.Popen[bytes] | None = None
    traffic: subprocess.Popen[bytes] | None = None
    started_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")

    print()
    print(f"Starting {spec.role} run: {spec.scenario}")
    print(f"Output: {output_path}")

    cleanup()
    time.sleep(2)

    try:
        network = start_network(args.controller_ip, args.controller_port)
        time.sleep(5)

        source = network.get(spec.source_host)
        destination = network.get(spec.destination_host)
        destination_ip = destination.IP()

        collector_environment = os.environ.copy()
        collector_environment.update(
            {
                "FC_SWITCH": "s1",
                "FC_INTERVAL": str(args.collector_interval),
                "FC_CSV": str(output_path),
            }
        )

        with log_path.open("wb") as log_file:
            collector = subprocess.Popen(
                ["python3", "-u", str(COLLECTOR_PATH)],
                stdout=log_file,
                stderr=subprocess.STDOUT,
                env=collector_environment,
            )
            time.sleep(args.collector_interval + 1)

            traffic = source.popen(
                [
                    "ping",
                    "-n",
                    "-i",
                    str(spec.ping_interval_seconds),
                    destination_ip,
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

            for elapsed in range(args.duration):
                time.sleep(1)
                if (elapsed + 1) % 20 == 0 or elapsed + 1 == args.duration:
                    print(
                        f"  {spec.filename}: {elapsed + 1}/{args.duration}s",
                        flush=True,
                    )

            stop_process(traffic)
            traffic = None
            time.sleep(args.collector_interval + 1)
            stop_process(collector)
            collector = None

    finally:
        stop_process(traffic)
        stop_process(collector)
        if network is not None:
            network.stop()
        cleanup()

    if not output_path.exists() or output_path.stat().st_size == 0:
        raise SystemExit(f"ERROR: collector produced no data: {output_path}")

    total_rows, icmp_rows, arp_rows = count_csv_rows(output_path)
    if icmp_rows == 0:
        raise SystemExit(f"ERROR: no ICMP rows collected: {output_path}")

    restore_user_ownership(output_path)
    restore_user_ownership(log_path)

    print(
        f"Completed {spec.filename}: total={total_rows}, "
        f"icmp={icmp_rows}, arp={arp_rows}"
    )

    return {
        "filename": spec.filename,
        "role": spec.role,
        "scenario": spec.scenario,
        "source_host": spec.source_host,
        "destination_host": spec.destination_host,
        "ping_interval_seconds": spec.ping_interval_seconds,
        "traffic_duration_seconds": args.duration,
        "collector_interval_seconds": args.collector_interval,
        "controller": f"{args.controller_ip}:{args.controller_port}",
        "switch": "s1",
        "started_at": started_at,
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "total_rows": total_rows,
        "icmp_rows": icmp_rows,
        "arp_rows": arp_rows,
    }


def write_manifest(output_dir: Path, rows: list[dict[str, object]]) -> Path:
    path = output_dir / "run_manifest.csv"
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    restore_user_ownership(path)
    return path


def main() -> None:
    args = parse_args()
    require_root()
    check_requirements(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    restore_user_ownership(args.output_dir)
    ensure_new_targets(args.output_dir)

    print("Independent SDN collection plan")
    print(f"Runs: {len(RUN_SPECS)}")
    print(f"Duration per run: {args.duration}s")
    print(f"Collector interval: {args.collector_interval}s")
    print(f"Destination: {args.output_dir}")

    results = [collect_one_run(args, spec) for spec in RUN_SPECS]
    manifest_path = write_manifest(args.output_dir, results)

    print()
    print("All independent runs completed.")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
