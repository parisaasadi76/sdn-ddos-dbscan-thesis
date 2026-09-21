#!/usr/bin/env python3
"""Collect stable OpenFlow counter deltas without treating metadata as identity.

Version 2 keeps the raw files produced by the earlier collector untouched.  A
flow's volatile fields (for example idle_age and duration) are excluded from
its identity.  The first observation of a flow generation is used only as a
baseline, so a cumulative counter is never reported as an interval delta.
"""

from __future__ import annotations

import csv
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


SWITCH = os.environ.get("FC_SWITCH", "br0")
INTERVAL = float(os.environ.get("FC_INTERVAL", "5"))
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = Path(
    os.environ.get("FC_CSV", PROJECT_ROOT / "data" / "raw" / "flows_v2.csv")
)

FIELDS = [
    "timestamp",
    "switch",
    "duration_sec",
    "table",
    "priority",
    "match",
    "actions",
    "packets",
    "bytes",
    "packets_delta",
    "bytes_delta",
    "packets_rate",
    "bytes_rate",
]

VOLATILE_PREFIXES = {
    "cookie",
    "duration",
    "table",
    "n_packets",
    "n_bytes",
    "idle_timeout",
    "hard_timeout",
    "idle_age",
    "priority",
}


def run_dump_flows(switch: str) -> str:
    ovs = shutil.which("ovs-ofctl")
    if ovs is None:
        raise FileNotFoundError("ovs-ofctl not found in PATH")
    command = [ovs, "-O", "OpenFlow10", "dump-flows", switch]
    process = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if process.returncode != 0:
        raise RuntimeError("ovs-ofctl failed: " + process.stderr.strip())
    return process.stdout


def metadata_value(line: str, name: str, default: str = "") -> str:
    match = re.search(rf"\b{re.escape(name)}=([^,\s]+)", line)
    return match.group(1) if match else default


def stable_match_from_head(head: str) -> str:
    stable_tokens: list[str] = []
    for raw_token in head.split(","):
        token = raw_token.strip()
        if not token:
            continue
        key = token.split("=", 1)[0].strip()
        if key in VOLATILE_PREFIXES:
            continue
        stable_tokens.append(token)
    return ",".join(stable_tokens)


def parse_flow_line(line: str) -> dict[str, object] | None:
    line = line.strip()
    if not line or line.startswith(("OFPST_FLOW", "NXST_FLOW")):
        return None

    actions_position = line.find("actions=")
    if actions_position < 0:
        return None

    packets_text = metadata_value(line, "n_packets")
    bytes_text = metadata_value(line, "n_bytes")
    if not packets_text or not bytes_text:
        return None

    duration_text = metadata_value(line, "duration", "0").rstrip("s")
    try:
        duration = float(duration_text)
        packets = int(packets_text)
        byte_count = int(bytes_text)
    except ValueError:
        return None

    head = line[:actions_position].strip(" ,")
    actions = line[actions_position + len("actions=") :].strip().rstrip(",")
    return {
        "duration_sec": duration,
        "table": metadata_value(line, "table"),
        "priority": metadata_value(line, "priority"),
        "match": stable_match_from_head(head),
        "actions": actions,
        "packets": packets,
        "bytes": byte_count,
    }


def parse_dump(text: str) -> list[dict[str, object]]:
    parsed = []
    for line in text.splitlines():
        flow = parse_flow_line(line)
        if flow is not None:
            parsed.append(flow)
    return parsed


def flow_key(flow: dict[str, object]) -> tuple[object, ...]:
    return (
        flow["table"],
        flow["priority"],
        flow["match"],
        flow["actions"],
    )


def is_same_generation(
    previous: tuple[int, int, float],
    flow: dict[str, object],
) -> bool:
    old_packets, old_bytes, old_duration = previous
    return (
        int(flow["packets"]) >= old_packets
        and int(flow["bytes"]) >= old_bytes
        and float(flow["duration_sec"]) >= old_duration
    )


def main() -> None:
    if INTERVAL <= 0:
        raise SystemExit("ERROR: FC_INTERVAL must be positive.")

    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    new_file = not CSV_PATH.exists() or CSV_PATH.stat().st_size == 0
    previous: dict[tuple[object, ...], tuple[int, int, float]] = {}
    previous_poll = time.monotonic()

    print(
        f"Collecting stable flow deltas from {SWITCH} every {INTERVAL}s "
        f"-> {CSV_PATH}",
        flush=True,
    )

    while True:
        try:
            flows = parse_dump(run_dump_flows(SWITCH))
        except Exception as error:
            sys.stderr.write(f"collect error: {error}\n")
            time.sleep(INTERVAL)
            previous_poll = time.monotonic()
            continue

        poll_time = time.monotonic()
        elapsed = max(poll_time - previous_poll, 0.001)
        previous_poll = poll_time
        now = datetime.now().astimezone().isoformat(timespec="seconds")
        current: dict[tuple[object, ...], tuple[int, int, float]] = {}

        try:
            with CSV_PATH.open("a", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                if new_file:
                    writer.writerow(FIELDS)
                    new_file = False

                for flow in flows:
                    key = flow_key(flow)
                    state = (
                        int(flow["packets"]),
                        int(flow["bytes"]),
                        float(flow["duration_sec"]),
                    )
                    current[key] = state
                    old_state = previous.get(key)

                    # A first observation or a counter reset is a baseline, not
                    # a valid measurement interval.  Do not write a misleading
                    # zero or cumulative rate for it.
                    if old_state is None or not is_same_generation(old_state, flow):
                        continue

                    packets_delta = state[0] - old_state[0]
                    bytes_delta = state[1] - old_state[1]
                    writer.writerow(
                        [
                            now,
                            SWITCH,
                            f"{state[2]:.3f}",
                            flow["table"],
                            flow["priority"],
                            flow["match"],
                            flow["actions"],
                            state[0],
                            state[1],
                            packets_delta,
                            bytes_delta,
                            f"{packets_delta / elapsed:.6f}",
                            f"{bytes_delta / elapsed:.6f}",
                        ]
                    )
        except OSError as error:
            sys.stderr.write(f"csv write error: {error}\n")

        previous = current
        time.sleep(INTERVAL)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
