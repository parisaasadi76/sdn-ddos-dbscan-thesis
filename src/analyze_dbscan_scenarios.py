#!/usr/bin/env python3

import csv
from collections import Counter
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "data" / "processed"

METADATA_PATH = PROCESSED_DIR / "metadata_icmp.csv"

EPS_TAGS = (
    "0p28",
    "0p31",
    "0p35",
    "0p40",
    "0p43",
    "0p46",
)

CROSSTAB_PATH = (
    PROCESSED_DIR / "dbscan_scenario_crosstab.csv"
)

NOISE_METRICS_PATH = (
    PROCESSED_DIR / "dbscan_noise_metrics.csv"
)


def read_metadata(path):
    if not path.exists():
        raise SystemExit(f"ERROR: metadata file not found: {path}")

    rows = []

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        for expected_id, row in enumerate(reader, start=1):
            try:
                row_id = int(row["row_id"])
            except (KeyError, ValueError):
                raise SystemExit(
                    f"ERROR: invalid row_id at row {expected_id}"
                )

            if row_id != expected_id:
                raise SystemExit(
                    "ERROR: metadata row_id is not sequential: "
                    f"expected {expected_id}, got {row_id}"
                )

            rows.append(row)

    if not rows:
        raise SystemExit("ERROR: metadata is empty.")

    return rows


def read_labels(path, expected_count):
    if not path.exists():
        raise SystemExit(f"ERROR: labels file not found: {path}")

    labels = {}

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        for row in reader:
            try:
                sample_index = int(row["sample_index"])
                cluster_label = int(row["cluster_label"])
            except (KeyError, ValueError):
                raise SystemExit(
                    f"ERROR: invalid label row in {path}"
                )

            if sample_index in labels:
                raise SystemExit(
                    f"ERROR: duplicate sample_index: {sample_index}"
                )

            labels[sample_index] = cluster_label

    expected_indices = set(range(1, expected_count + 1))

    if set(labels) != expected_indices:
        missing = sorted(expected_indices - set(labels))
        extra = sorted(set(labels) - expected_indices)

        raise SystemExit(
            f"ERROR: index mismatch in {path}; "
            f"missing={missing}, extra={extra}"
        )

    return labels


def percent(numerator, denominator):
    if denominator == 0:
        return ""

    return f"{100.0 * numerator / denominator:.4f}"


def cluster_column(cluster_label):
    if cluster_label == -1:
        return "noise"

    return f"cluster_{cluster_label}"


def main():
    metadata = read_metadata(METADATA_PATH)

    print(f"Metadata rows: {len(metadata)}")

    scenario_counts = Counter(
        row["scenario"]
        for row in metadata
    )

    print("Scenario counts:")
    for scenario, count in scenario_counts.items():
        print(f"  {scenario}: {count}")

    analyses = []
    all_cluster_labels = set()

    for eps_tag in EPS_TAGS:
        labels_path = (
            PROCESSED_DIR
            / f"dbscan_labels_eps_{eps_tag}.csv"
        )

        labels = read_labels(
            labels_path,
            expected_count=len(metadata),
        )

        counts_by_scenario = {}

        for row_id, row in enumerate(metadata, start=1):
            scenario = row["scenario"]
            cluster_label = labels[row_id]

            if scenario not in counts_by_scenario:
                counts_by_scenario[scenario] = Counter()

            counts_by_scenario[scenario][cluster_label] += 1
            all_cluster_labels.add(cluster_label)

        true_normal = 0
        true_high_rate = 0

        for row_id, row in enumerate(metadata, start=1):
            is_high_rate = str(row["label"]) == "1"
            is_noise = labels[row_id] == -1

            if is_high_rate and is_noise:
                true_high_rate += 1

            if (not is_high_rate) and (not is_noise):
                true_normal += 1

        true_positive = 0
        false_positive = 0
        true_negative = 0
        false_negative = 0

        for row_id, row in enumerate(metadata, start=1):
            actual_anomaly = str(row["label"]) == "1"
            predicted_anomaly = labels[row_id] == -1

            if actual_anomaly and predicted_anomaly:
                true_positive += 1
            elif (not actual_anomaly) and predicted_anomaly:
                false_positive += 1
            elif (not actual_anomaly) and (not predicted_anomaly):
                true_negative += 1
            else:
                false_negative += 1

        analyses.append(
            {
                "eps": eps_tag,
                "counts": counts_by_scenario,
                "tp": true_positive,
                "fp": false_positive,
                "tn": true_negative,
                "fn": false_negative,
            }
        )

    ordered_clusters = sorted(
        all_cluster_labels,
        key=lambda value: (value != -1, value),
    )

    cluster_columns = [
        cluster_column(value)
        for value in ordered_clusters
    ]

    print()

    for analysis in analyses:
        eps_text = analysis["eps"].replace("p", ".")

        print(f"=== eps={eps_text} ===")

        columns = [
            "scenario",
            "total",
            *cluster_columns,
            "noise_percent",
        ]

        print(" | ".join(f"{column:>15}" for column in columns))
        print("-" * (19 * len(columns)))

        for scenario in ("normal", "high_rate"):
            counter = analysis["counts"].get(
                scenario,
                Counter(),
            )

            total = sum(counter.values())

            values = [
                scenario,
                str(total),
            ]

            for cluster_label in ordered_clusters:
                values.append(
                    str(counter.get(cluster_label, 0))
                )

            values.append(
                percent(counter.get(-1, 0), total)
            )

            print(" | ".join(f"{value:>15}" for value in values))

        tp = analysis["tp"]
        fp = analysis["fp"]
        tn = analysis["tn"]
        fn = analysis["fn"]

        print()
        print(
            "Noise-only metrics "
            "(only -1 is considered anomaly):"
        )
        print(f"  TP={tp}, FP={fp}, TN={tn}, FN={fn}")
        print(
            f"  precision={percent(tp, tp + fp)}%"
        )
        print(
            f"  recall={percent(tp, tp + fn)}%"
        )
        print(
            f"  false_positive_rate={percent(fp, fp + tn)}%"
        )
        print()

    crosstab_fields = [
        "eps",
        "scenario",
        "total",
        *cluster_columns,
        "noise_percent",
    ]

    with CROSSTAB_PATH.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=crosstab_fields,
        )

        writer.writeheader()

        for analysis in analyses:
            for scenario in ("normal", "high_rate"):
                counter = analysis["counts"].get(
                    scenario,
                    Counter(),
                )

                total = sum(counter.values())

                output_row = {
                    "eps": analysis["eps"].replace("p", "."),
                    "scenario": scenario,
                    "total": total,
                    "noise_percent": percent(
                        counter.get(-1, 0),
                        total,
                    ),
                }

                for cluster_label in ordered_clusters:
                    output_row[
                        cluster_column(cluster_label)
                    ] = counter.get(cluster_label, 0)

                writer.writerow(output_row)

    metrics_fields = [
        "eps",
        "true_positive",
        "false_positive",
        "true_negative",
        "false_negative",
        "noise_precision_percent",
        "noise_recall_percent",
        "noise_false_positive_rate_percent",
    ]

    with NOISE_METRICS_PATH.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=metrics_fields,
        )

        writer.writeheader()

        for analysis in analyses:
            tp = analysis["tp"]
            fp = analysis["fp"]
            tn = analysis["tn"]
            fn = analysis["fn"]

            writer.writerow(
                {
                    "eps": analysis["eps"].replace("p", "."),
                    "true_positive": tp,
                    "false_positive": fp,
                    "true_negative": tn,
                    "false_negative": fn,
                    "noise_precision_percent": percent(
                        tp,
                        tp + fp,
                    ),
                    "noise_recall_percent": percent(
                        tp,
                        tp + fn,
                    ),
                    "noise_false_positive_rate_percent": percent(
                        fp,
                        fp + tn,
                    ),
                }
            )

    print(f"Crosstab saved to: {CROSSTAB_PATH}")
    print(
        f"Noise metrics saved to: {NOISE_METRICS_PATH}"
    )


if __name__ == "__main__":
    main()
