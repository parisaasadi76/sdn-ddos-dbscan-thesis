#!/usr/bin/env python3

"""Evaluate a Normal-calibrated DBSCAN model on independent SDN runs.

The scaler and DBSCAN model are fitted only on a Normal calibration run.
Ground-truth scenario names are supplied at file level and are used only for
evaluation. New samples are accepted when they are within eps of at least one
DBSCAN core point from the calibration set.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


FEATURES = ("packets_rate", "bytes_rate", "duration_sec")
DEFAULT_EPS_VALUES = (0.28, 0.31, 0.35, 0.40, 0.43, 0.46)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Independent DBSCAN evaluation using Normal calibration.",
    )
    parser.add_argument("--calibration-normal", type=Path, required=True)
    parser.add_argument("--test-normal", type=Path, required=True)
    parser.add_argument("--test-high-rate", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--eps",
        type=float,
        nargs="+",
        default=DEFAULT_EPS_VALUES,
    )
    parser.add_argument("--min-samples", type=int, default=5)
    return parser.parse_args()


def is_icmp(match_value: str) -> bool:
    return re.search(r"\bicmp\b", (match_value or "").lower()) is not None


def parse_number(value: str) -> float | None:
    text = (value or "").strip()
    if text.endswith("s"):
        text = text[:-1]
    try:
        number = float(text)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def read_icmp_features(path: Path) -> tuple[np.ndarray, list[dict[str, str]]]:
    if not path.exists():
        raise SystemExit(f"ERROR: input file not found: {path}")

    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        header = set(reader.fieldnames or [])
        required = {"timestamp", "switch", "match", *FEATURES}
        missing = sorted(required - header)
        if missing:
            raise SystemExit(f"ERROR: {path.name} missing columns: {missing}")

        matrix_rows: list[list[float]] = []
        metadata_rows: list[dict[str, str]] = []
        skipped_numeric = 0

        for source_row, row in enumerate(reader, start=2):
            if not is_icmp(row.get("match", "")):
                continue

            values = [parse_number(row.get(feature, "")) for feature in FEATURES]
            if any(value is None for value in values):
                skipped_numeric += 1
                continue

            matrix_rows.append([float(value) for value in values])
            metadata_rows.append(
                {
                    "source_row": str(source_row),
                    "timestamp": row.get("timestamp", ""),
                    "switch": row.get("switch", ""),
                }
            )

    if skipped_numeric:
        raise SystemExit(
            f"ERROR: {path.name} contains {skipped_numeric} ICMP rows with invalid "
            "numeric features. Resolve them before evaluation."
        )
    if not matrix_rows:
        raise SystemExit(f"ERROR: no valid ICMP rows found in {path}")

    return np.asarray(matrix_rows, dtype=float), metadata_rows


def divide(numerator: float, denominator: float) -> float | None:
    return None if denominator == 0 else numerator / denominator


def number(value: float | None) -> str:
    if value is None or not math.isfinite(value):
        return ""
    return f"{value:.8f}"


def write_rows(
    path: Path,
    fieldnames: list[str],
    rows: list[dict[str, object]],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.min_samples < 2:
        raise SystemExit("ERROR: min_samples must be at least 2.")
    if any(eps <= 0 for eps in args.eps):
        raise SystemExit("ERROR: every eps value must be positive.")

    calibration_raw, calibration_meta = read_icmp_features(
        args.calibration_normal
    )
    test_normal_raw, test_normal_meta = read_icmp_features(args.test_normal)
    test_high_raw, test_high_meta = read_icmp_features(args.test_high_rate)

    if len(calibration_raw) < args.min_samples:
        raise SystemExit(
            "ERROR: calibration set has fewer samples than min_samples."
        )

    scaler = StandardScaler()
    calibration = scaler.fit_transform(calibration_raw)
    test_normal = scaler.transform(test_normal_raw)
    test_high = scaler.transform(test_high_raw)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    scaler_rows = [
        {
            "feature": feature,
            "calibration_mean": f"{scaler.mean_[index]:.10f}",
            "calibration_scale": f"{scaler.scale_[index]:.10f}",
        }
        for index, feature in enumerate(FEATURES)
    ]
    write_rows(
        args.output_dir / "calibration_scaler_stats.csv",
        ["feature", "calibration_mean", "calibration_scale"],
        scaler_rows,
    )

    neighbor_model = NearestNeighbors(n_neighbors=args.min_samples)
    neighbor_model.fit(calibration)
    distances, _ = neighbor_model.kneighbors(calibration)
    kth_distances = sorted(float(row[-1]) for row in distances)
    write_rows(
        args.output_dir / "calibration_k_distance.csv",
        ["rank", f"distance_to_neighbor_{args.min_samples}"],
        [
            {
                "rank": rank,
                f"distance_to_neighbor_{args.min_samples}": f"{distance:.10f}",
            }
            for rank, distance in enumerate(kth_distances, start=1)
        ],
    )

    result_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []

    test_matrix = np.vstack([test_normal, test_high])
    test_actual = np.concatenate(
        [
            np.zeros(len(test_normal), dtype=int),
            np.ones(len(test_high), dtype=int),
        ]
    )
    test_scenarios = ["normal"] * len(test_normal) + ["high_rate"] * len(
        test_high
    )
    test_metadata = test_normal_meta + test_high_meta
    test_sources = [args.test_normal.name] * len(test_normal) + [
        args.test_high_rate.name
    ] * len(test_high)

    for eps in args.eps:
        model = DBSCAN(
            eps=eps,
            min_samples=args.min_samples,
            metric="euclidean",
            n_jobs=-1,
        ).fit(calibration)

        core_indices = np.asarray(model.core_sample_indices_, dtype=int)
        if len(core_indices) == 0:
            raise SystemExit(
                f"ERROR: eps={eps:.2f} produced no calibration core points."
            )

        core_points = calibration[core_indices]
        nearest_core = NearestNeighbors(n_neighbors=1).fit(core_points)
        test_distances, nearest_indices = nearest_core.kneighbors(test_matrix)
        nearest_distances = test_distances[:, 0]
        predicted_anomaly = nearest_distances > eps

        actual_anomaly = test_actual == 1
        tp = int(np.sum(actual_anomaly & predicted_anomaly))
        fp = int(np.sum(~actual_anomaly & predicted_anomaly))
        tn = int(np.sum(~actual_anomaly & ~predicted_anomaly))
        fn = int(np.sum(actual_anomaly & ~predicted_anomaly))

        precision = divide(tp, tp + fp)
        recall = divide(tp, tp + fn)
        specificity = divide(tn, tn + fp)
        fpr = divide(fp, fp + tn)
        f1 = (
            divide(2.0 * precision * recall, precision + recall)
            if precision is not None and recall is not None
            else None
        )
        balanced_accuracy = (
            (recall + specificity) / 2.0
            if recall is not None and specificity is not None
            else None
        )

        result_rows.append(
            {
                "eps": f"{eps:.2f}",
                "min_samples": args.min_samples,
                "calibration_samples": len(calibration),
                "calibration_clusters": len(set(model.labels_) - {-1}),
                "calibration_core_points": len(core_indices),
                "calibration_noise_points": int(np.sum(model.labels_ == -1)),
                "test_normal_samples": len(test_normal),
                "test_high_rate_samples": len(test_high),
                "true_positive": tp,
                "false_positive": fp,
                "true_negative": tn,
                "false_negative": fn,
                "precision": number(precision),
                "recall": number(recall),
                "f1_score": number(f1),
                "false_positive_rate": number(fpr),
                "specificity": number(specificity),
                "balanced_accuracy": number(balanced_accuracy),
            }
        )

        for index, is_anomaly in enumerate(predicted_anomaly):
            prediction_rows.append(
                {
                    "eps": f"{eps:.2f}",
                    "test_index": index + 1,
                    "source_file": test_sources[index],
                    "source_row": test_metadata[index]["source_row"],
                    "timestamp": test_metadata[index]["timestamp"],
                    "switch": test_metadata[index]["switch"],
                    "scenario": test_scenarios[index],
                    "actual_label": int(test_actual[index]),
                    "predicted_anomaly": int(is_anomaly),
                    "nearest_core_distance": f"{nearest_distances[index]:.10f}",
                    "nearest_calibration_core_index": int(
                        core_indices[nearest_indices[index, 0]] + 1
                    ),
                }
            )

    write_rows(
        args.output_dir / "independent_eps_metrics.csv",
        list(result_rows[0]),
        result_rows,
    )
    write_rows(
        args.output_dir / "independent_predictions.csv",
        list(prediction_rows[0]),
        prediction_rows,
    )

    print(f"Calibration Normal ICMP samples: {len(calibration)}")
    print(f"Test Normal ICMP samples: {len(test_normal)}")
    print(f"Test High-Rate ICMP samples: {len(test_high)}")
    print(f"Scaler fitted only on: {args.calibration_normal}")
    print(f"Created: {args.output_dir / 'calibration_scaler_stats.csv'}")
    print(f"Created: {args.output_dir / 'calibration_k_distance.csv'}")
    print(f"Created: {args.output_dir / 'independent_eps_metrics.csv'}")
    print(f"Created: {args.output_dir / 'independent_predictions.csv'}")


if __name__ == "__main__":
    main()
