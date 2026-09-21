#!/usr/bin/env python3

import csv
from pathlib import Path

from sklearn.neighbors import NearestNeighbors


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "X_icmp_scaled.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "k_distance_min5.csv"

MIN_SAMPLES = 5


def read_matrix(path):
    if not path.exists():
        raise SystemExit(f"ERROR: input file not found: {path}")

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        reader = csv.reader(file)
        header = next(reader, None)

        if header is None:
            raise SystemExit("ERROR: input file has no header.")

        rows = []

        for line_number, row in enumerate(reader, start=2):
            if not row or all(not value.strip() for value in row):
                continue

            if len(row) != len(header):
                raise SystemExit(
                    f"ERROR: invalid column count at line {line_number}"
                )

            try:
                values = [float(value.strip()) for value in row]
            except ValueError:
                raise SystemExit(
                    f"ERROR: non-numeric value at line {line_number}"
                )

            rows.append(values)

    if not rows:
        raise SystemExit("ERROR: input matrix has no data rows.")

    return header, rows


def percentile(values, percentage):
    ordered = sorted(values)

    position = (len(ordered) - 1) * percentage / 100.0
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(ordered) - 1)

    fraction = position - lower_index

    return (
        ordered[lower_index]
        + fraction
        * (ordered[upper_index] - ordered[lower_index])
    )


def main():
    header, rows = read_matrix(DATA_PATH)

    if len(rows) < MIN_SAMPLES:
        raise SystemExit(
            "ERROR: number of rows is smaller than min_samples."
        )

    model = NearestNeighbors(n_neighbors=MIN_SAMPLES)
    model.fit(rows)

    distances, _ = model.kneighbors(rows)

    kth_distances = sorted(
        row_distances[-1]
        for row_distances in distances
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.writer(file)

        writer.writerow(
            [
                "rank",
                f"distance_to_neighbor_{MIN_SAMPLES}",
            ]
        )

        for rank, distance in enumerate(kth_distances, start=1):
            writer.writerow(
                [
                    rank,
                    f"{distance:.10f}",
                ]
            )

    print(f"Rows: {len(rows)}")
    print(f"Features: {', '.join(header)}")
    print(f"min_samples: {MIN_SAMPLES}")
    print()
    print("k-distance percentiles:")

    for percentage in (50, 75, 80, 85, 90, 95, 99):
        value = percentile(kth_distances, percentage)
        print(f"  P{percentage}: {value:.6f}")

    print()
    print(f"Created: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
