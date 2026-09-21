#!/usr/bin/env python3
"""Run the independent experiment with the stable-delta collector (v2)."""

from __future__ import annotations

import csv
import math
import re
from pathlib import Path
from statistics import median

import collect_independent_runs as implementation


BASE_DIR = Path(__file__).resolve().parents[1]
implementation.COLLECTOR_PATH = BASE_DIR / "flow_collector_v2.py"
implementation.DEFAULT_OUTPUT_DIR = BASE_DIR / "data" / "raw" / "independent_v2"

original_collect_one_run = implementation.collect_one_run


def validate_rates(path: Path, expected_rate: float) -> dict[str, float]:
    rates: list[float] = []
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            if re.search(r"\bicmp\b", (row.get("match") or "").lower()):
                try:
                    rate = float(row["packets_rate"])
                except (KeyError, TypeError, ValueError):
                    raise SystemExit(
                        f"ERROR: invalid ICMP packet rate in {path}"
                    )
                if not math.isfinite(rate):
                    raise SystemExit(
                        f"ERROR: non-finite ICMP packet rate in {path}"
                    )
                rates.append(rate)

    if len(rates) < 20:
        raise SystemExit(
            f"ERROR: too few valid ICMP rate rows in {path}: {len(rates)}"
        )

    measured_median = median(rates)
    # A packet landing on either side of a polling boundary can produce one
    # fewer or one extra packet in a Normal two-second window.
    lower = expected_rate * 0.45
    upper = expected_rate * 1.55
    outside = sum(rate < lower or rate > upper for rate in rates)
    outside_fraction = outside / len(rates)

    if not lower <= measured_median <= upper:
        raise SystemExit(
            f"ERROR: median ICMP rate {measured_median:.3f} is outside "
            f"the expected range [{lower:.3f}, {upper:.3f}] in {path}"
        )
    if outside_fraction > 0.10:
        raise SystemExit(
            f"ERROR: {outside_fraction:.1%} of ICMP rates are outside "
            f"the expected range in {path}"
        )

    return {
        "rate_min": min(rates),
        "rate_median": measured_median,
        "rate_max": max(rates),
        "rate_outside_expected_fraction": outside_fraction,
    }


def collect_one_run(args, spec):
    result = original_collect_one_run(args, spec)
    quality = validate_rates(
        args.output_dir / spec.filename,
        expected_rate=1.0 / spec.ping_interval_seconds,
    )
    result.update(quality)
    print(
        "Quality check passed: "
        f"rate min/median/max={quality['rate_min']:.3f}/"
        f"{quality['rate_median']:.3f}/{quality['rate_max']:.3f} pkt/s"
    )
    return result


implementation.collect_one_run = collect_one_run


if __name__ == "__main__":
    implementation.main()
