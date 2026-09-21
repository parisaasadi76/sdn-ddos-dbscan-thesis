#!/usr/bin/env python3

import csv
import math
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUT_DIR = PROJECT_ROOT / "data" / "processed"

INPUTS = [
    ("normal_01.csv", "normal", 0),
    ("high_rate_icmp_01.csv", "high_rate", 1),
]

ALL_NUMERIC_FEATURES = [
    "packets",
    "bytes",
    "packets_delta",
    "bytes_delta",
    "packets_rate",
    "bytes_rate",
    "duration_sec",
]

MODEL_FEATURES = [
    "packets_rate",
    "bytes_rate",
    "duration_sec",
]


def to_float(value):
    if value is None:
        return None

    text = str(value).strip()

    if text.endswith("s"):
        text = text[:-1]

    try:
        number = float(text)
    except ValueError:
        return None

    if not math.isfinite(number):
        return None

    return number


def extract_protocol(match_value):
    text = (match_value or "").lower()

    if re.search(r"\bicmp\b", text):
        return "icmp"

    if re.search(r"\barp\b", text):
        return "arp"

    return "other"


def write_csv(path, fieldnames, rows):
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    processed_rows = []

    for filename, scenario, label in INPUTS:
        input_path = RAW_DIR / filename

        if not input_path.exists():
            raise SystemExit(
                f"ERROR: input file not found: {input_path}"
            )

        total_rows = 0
        icmp_rows = 0
        skipped_rows = 0

        with input_path.open(
            "r",
            encoding="utf-8",
            newline="",
        ) as file:
            reader = csv.DictReader(file)
            fieldnames = reader.fieldnames or []

            required_fields = {
                "timestamp",
                "switch",
                "match",
                *ALL_NUMERIC_FEATURES,
            }

            missing_fields = sorted(
                required_fields - set(fieldnames)
            )

            if missing_fields:
                raise SystemExit(
                    f"ERROR: {filename} is missing fields: "
                    f"{missing_fields}"
                )

            for row in reader:
                total_rows += 1

                protocol = extract_protocol(row.get("match"))

                if protocol != "icmp":
                    continue

                icmp_rows += 1

                numeric_values = {
                    feature: to_float(row.get(feature))
                    for feature in ALL_NUMERIC_FEATURES
                }

                if any(
                    value is None
                    for value in numeric_values.values()
                ):
                    skipped_rows += 1
                    continue

                processed_rows.append(
                    {
                        "timestamp": row.get("timestamp", ""),
                        "source_file": filename,
                        "scenario": scenario,
                        "label": label,
                        "switch": row.get("switch", ""),
                        "protocol": protocol,
                        **numeric_values,
                    }
                )

        print(
            f"{filename}: "
            f"total={total_rows}, "
            f"icmp={icmp_rows}, "
            f"skipped_numeric={skipped_rows}"
        )

    if not processed_rows:
        raise SystemExit("ERROR: no valid ICMP rows were found.")

    for row_id, row in enumerate(processed_rows, start=1):
        row["row_id"] = row_id

    processed_fields = [
        "row_id",
        "timestamp",
        "source_file",
        "scenario",
        "label",
        "switch",
        "protocol",
        *ALL_NUMERIC_FEATURES,
    ]

    processed_path = OUT_DIR / "processed_icmp.csv"

    write_csv(
        processed_path,
        processed_fields,
        processed_rows,
    )

    means = {}
    standard_deviations = {}

    for feature in MODEL_FEATURES:
        values = [row[feature] for row in processed_rows]

        mean = sum(values) / len(values)

        variance = sum(
            (value - mean) ** 2
            for value in values
        ) / len(values)

        standard_deviation = math.sqrt(variance)

        if standard_deviation < 1e-12:
            standard_deviation = 1.0

        means[feature] = mean
        standard_deviations[feature] = standard_deviation

    scaled_path = OUT_DIR / "X_icmp_scaled.csv"

    with scaled_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.writer(file)
        writer.writerow(MODEL_FEATURES)

        for row in processed_rows:
            scaled_values = []

            for feature in MODEL_FEATURES:
                scaled_value = (
                    row[feature] - means[feature]
                ) / standard_deviations[feature]

                scaled_values.append(
                    f"{scaled_value:.10f}"
                )

            writer.writerow(scaled_values)

    metadata_path = OUT_DIR / "metadata_icmp.csv"

    metadata_fields = [
        "row_id",
        "timestamp",
        "source_file",
        "scenario",
        "label",
        "protocol",
    ]

    metadata_rows = [
        {
            field: row[field]
            for field in metadata_fields
        }
        for row in processed_rows
    ]

    write_csv(
        metadata_path,
        metadata_fields,
        metadata_rows,
    )

    stats_path = OUT_DIR / "scaler_stats.csv"

    with stats_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.writer(file)

        writer.writerow(
            [
                "feature",
                "mean",
                "standard_deviation",
            ]
        )

        for feature in MODEL_FEATURES:
            writer.writerow(
                [
                    feature,
                    f"{means[feature]:.10f}",
                    f"{standard_deviations[feature]:.10f}",
                ]
            )

    print()
    print(f"Valid ICMP rows: {len(processed_rows)}")
    print(
        "Model features: "
        + ", ".join(MODEL_FEATURES)
    )
    print(f"Created: {processed_path}")
    print(f"Created: {scaled_path}")
    print(f"Created: {metadata_path}")
    print(f"Created: {stats_path}")


if __name__ == "__main__":
    main()
