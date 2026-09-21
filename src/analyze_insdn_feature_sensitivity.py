#!/usr/bin/env python3

"""Feature-sensitivity evaluation of a normal-only DBSCAN detector on InSDN.

Feature sets and transforms are fixed before evaluating DDoS labels. Normal
flows are split by Flow ID into calibration, validation, and test partitions.
StandardScaler, DBSCAN core points, and eps selection use normal data only.
"""

from __future__ import annotations

import csv
import math
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "data/public/insdn/InSDN_DatasetCSV"
OUTPUT_DIR = ROOT / "data/processed/results/insdn_feature_sensitivity"
NORMAL_FILE = SOURCE_DIR / "Normal_data.csv"
ATTACK_FILES = (SOURCE_DIR / "OVS.csv", SOURCE_DIR / "metasploitable-2.csv")

RANDOM_SEED = 20260919
MIN_SAMPLES = 5
TARGET_VALIDATION_FPR = 0.05
EPS_QUANTILES = (0.90, 0.925, 0.95, 0.96, 0.97, 0.975, 0.98, 0.985, 0.99, 0.995)

BASELINE3 = ("Flow Pkts/s", "Flow Byts/s", "Flow Duration")
COUNTS7 = BASELINE3 + (
    "Tot Fwd Pkts",
    "Tot Bwd Pkts",
    "TotLen Fwd Pkts",
    "TotLen Bwd Pkts",
)
SHAPE12 = COUNTS7 + (
    "Down/Up Ratio",
    "Pkt Len Mean",
    "Pkt Len Std",
    "Flow IAT Mean",
    "Flow IAT Std",
)
ALL_NUMERIC_COLUMNS = SHAPE12

VARIANTS = (
    {
        "name": "baseline3_raw",
        "features": BASELINE3,
        "transform": "standardize_raw",
        "protocol": False,
        "role": "baseline",
    },
    {
        "name": "counts7_raw",
        "features": COUNTS7,
        "transform": "standardize_raw",
        "protocol": False,
        "role": "ablation",
    },
    {
        "name": "counts7_log1p",
        "features": COUNTS7,
        "transform": "log1p_then_standardize",
        "protocol": False,
        "role": "complementary_view",
    },
    {
        "name": "shape12_log1p",
        "features": SHAPE12,
        "transform": "log1p_then_standardize",
        "protocol": False,
        "role": "sensitivity",
    },
    {
        "name": "counts7_log1p_protocol",
        "features": COUNTS7,
        "transform": "log1p_then_standardize",
        "protocol": True,
        "role": "diagnostic_upper_bound_not_headline",
    },
)


def divide(numerator: float, denominator: float) -> float | None:
    return None if denominator == 0 else numerator / denominator


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty table: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_rows(path: Path, wanted_label: str) -> dict[str, object]:
    features: list[list[float]] = []
    flow_ids: list[str] = []
    protocols: list[int] = []
    total_rows = 0
    invalid_target_rows = 0
    label_counts: Counter[str] = Counter()

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"Flow ID", "Protocol", "Label", *ALL_NUMERIC_COLUMNS}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"ERROR: {path.name} missing columns: {sorted(missing)}")

        for row in reader:
            total_rows += 1
            label = (row.get("Label") or "").strip()
            label_counts[label] += 1
            if label != wanted_label:
                continue
            try:
                values = []
                for column in ALL_NUMERIC_COLUMNS:
                    value = float(row[column])
                    values.append(value)
                protocol = int(float(row["Protocol"]))
            except (TypeError, ValueError):
                invalid_target_rows += 1
                continue

            duration = values[ALL_NUMERIC_COLUMNS.index("Flow Duration")]
            if (
                not all(math.isfinite(value) for value in values)
                or any(value < 0 for value in values)
                or duration <= 0
            ):
                invalid_target_rows += 1
                continue

            features.append(values)
            flow_ids.append(row["Flow ID"])
            protocols.append(protocol)

    return {
        "path": path,
        "features": np.asarray(features, dtype=float),
        "flow_ids": np.asarray(flow_ids, dtype=object),
        "protocols": np.asarray(protocols, dtype=int),
        "total_rows": total_rows,
        "valid_target_rows": len(features),
        "invalid_target_rows": invalid_target_rows,
        "label_counts": label_counts,
    }


def grouped_normal_split(flow_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = np.arange(len(flow_ids))
    splitter = GroupShuffleSplit(n_splits=1, train_size=0.50, random_state=RANDOM_SEED)
    train, remainder = next(splitter.split(indices, groups=flow_ids))
    splitter = GroupShuffleSplit(n_splits=1, train_size=0.50, random_state=RANDOM_SEED + 1)
    validation_relative, test_relative = next(
        splitter.split(remainder, groups=flow_ids[remainder])
    )
    return train, remainder[validation_relative], remainder[test_relative]


def protocol_one_hot(protocols: np.ndarray) -> np.ndarray:
    categories = (0, 6, 17)
    columns = [(protocols == value).astype(float) for value in categories]
    columns.append((~np.isin(protocols, categories)).astype(float))
    return np.column_stack(columns)


def select_columns(features: np.ndarray, names: tuple[str, ...]) -> np.ndarray:
    indices = [ALL_NUMERIC_COLUMNS.index(name) for name in names]
    return features[:, indices]


def transform_numeric(values: np.ndarray, transform: str) -> np.ndarray:
    if transform == "log1p_then_standardize":
        return np.log1p(values)
    return values.copy()


def nearest_distances(reference: np.ndarray, samples: np.ndarray) -> np.ndarray:
    model = NearestNeighbors(n_neighbors=1, algorithm="auto", n_jobs=-1).fit(reference)
    return model.kneighbors(samples, return_distance=True)[0][:, 0]


def prepare_variant(
    variant: dict[str, object],
    normal_features: np.ndarray,
    attack_features: np.ndarray,
    normal_protocols: np.ndarray,
    attack_protocols: np.ndarray,
    train: np.ndarray,
    validation: np.ndarray,
    test: np.ndarray,
) -> dict[str, object]:
    feature_names = tuple(variant["features"])
    normal_numeric = transform_numeric(
        select_columns(normal_features, feature_names), str(variant["transform"])
    )
    attack_numeric = transform_numeric(
        select_columns(attack_features, feature_names), str(variant["transform"])
    )

    scaler = StandardScaler()
    calibration = scaler.fit_transform(normal_numeric[train])
    validation_data = scaler.transform(normal_numeric[validation])
    test_normal = scaler.transform(normal_numeric[test])
    test_attack = scaler.transform(attack_numeric)

    if bool(variant["protocol"]):
        calibration = np.column_stack((calibration, protocol_one_hot(normal_protocols[train])))
        validation_data = np.column_stack(
            (validation_data, protocol_one_hot(normal_protocols[validation]))
        )
        test_normal = np.column_stack((test_normal, protocol_one_hot(normal_protocols[test])))
        test_attack = np.column_stack((test_attack, protocol_one_hot(attack_protocols)))

    neighbors = NearestNeighbors(
        n_neighbors=MIN_SAMPLES, algorithm="auto", n_jobs=-1
    ).fit(calibration)
    kth_distance = neighbors.kneighbors(calibration, return_distance=True)[0][:, -1]

    validation_rows: list[dict[str, object]] = []
    for quantile in EPS_QUANTILES:
        eps = float(np.quantile(kth_distance, quantile))
        core = calibration[kth_distance <= eps]
        distances = nearest_distances(core, validation_data)
        validation_rows.append(
            {
                "variant": variant["name"],
                "role": variant["role"],
                "eps_quantile": quantile,
                "eps": eps,
                "eps_view1": "",
                "eps_view2": "",
                "eps_quantile_view1": "",
                "eps_quantile_view2": "",
                "core_points": len(core),
                "validation_samples": len(validation_data),
                "validation_false_positives": int(np.sum(distances > eps)),
                "validation_false_positive_rate": float(np.mean(distances > eps)),
            }
        )

    eligible = [
        row
        for row in validation_rows
        if float(row["validation_false_positive_rate"]) <= TARGET_VALIDATION_FPR
    ]
    if not eligible:
        selected = min(
            validation_rows, key=lambda row: float(row["validation_false_positive_rate"])
        )
        selection_status = "fallback_lowest_validation_fpr"
    else:
        selected = min(eligible, key=lambda row: float(row["eps"]))
        selection_status = "met_validation_fpr_target"

    return {
        "calibration": calibration,
        "validation": validation_data,
        "test_normal": test_normal,
        "test_attack": test_attack,
        "kth_distance": kth_distance,
        "validation_rows": validation_rows,
        "selected_eps": float(selected["eps"]),
        "selected_quantile": float(selected["eps_quantile"]),
        "selected_validation_fpr": float(selected["validation_false_positive_rate"]),
        "selection_status": selection_status,
        "effective_dimensions": calibration.shape[1],
    }


def confusion_metrics(normal_anomaly: np.ndarray, attack_anomaly: np.ndarray) -> dict[str, object]:
    tn = int(np.sum(~normal_anomaly))
    fp = int(np.sum(normal_anomaly))
    tp = int(np.sum(attack_anomaly))
    fn = int(np.sum(~attack_anomaly))
    precision = divide(tp, tp + fp)
    recall = divide(tp, tp + fn)
    specificity = divide(tn, tn + fp)
    fpr = divide(fp, fp + tn)
    f1 = divide(2 * precision * recall, precision + recall) if precision and recall else 0.0
    balanced_accuracy = (recall + specificity) / 2 if recall is not None and specificity is not None else None
    return {
        "true_negative": tn,
        "false_positive": fp,
        "true_positive": tp,
        "false_negative": fn,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "false_positive_rate": fpr,
        "specificity": specificity,
        "balanced_accuracy": balanced_accuracy,
    }


def evaluate_variant(
    variant: dict[str, object],
    fit: dict[str, object],
    balanced_attack_indices: np.ndarray,
    attack_sources: np.ndarray,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    eps = float(fit["selected_eps"])
    calibration = np.asarray(fit["calibration"])
    kth_distance = np.asarray(fit["kth_distance"])
    core = calibration[kth_distance <= eps]
    if not len(core):
        raise SystemExit(f"ERROR: {variant['name']} produced no core points")

    normal_distances = nearest_distances(core, np.asarray(fit["test_normal"]))
    attack_distances = nearest_distances(core, np.asarray(fit["test_attack"]))
    normal_anomaly = normal_distances > eps
    full_attack_anomaly = attack_distances > eps
    balanced_metrics = confusion_metrics(
        normal_anomaly, full_attack_anomaly[balanced_attack_indices]
    )

    result = {
        "variant": variant["name"],
        "role": variant["role"],
        "numeric_feature_count": len(tuple(variant["features"])),
        "effective_dimensions": fit["effective_dimensions"],
        "transform": variant["transform"],
        "uses_protocol": bool(variant["protocol"]),
        "eps": eps,
        "eps_view1": "",
        "eps_view2": "",
        "eps_quantile_view1": "",
        "eps_quantile_view2": "",
        "eps_quantile": fit["selected_quantile"],
        "min_samples": MIN_SAMPLES,
        "selection_status": fit["selection_status"],
        "validation_false_positive_rate": fit["selected_validation_fpr"],
        "calibration_samples": len(calibration),
        "core_points": len(core),
        "test_normal_samples": len(normal_anomaly),
        "balanced_test_attack_samples": len(balanced_attack_indices),
        "full_ddos_samples": len(full_attack_anomaly),
        **balanced_metrics,
        "full_ddos_detected": int(np.sum(full_attack_anomaly)),
        "full_ddos_recall": float(np.mean(full_attack_anomaly)),
    }

    source_rows: list[dict[str, object]] = []
    for source in sorted(set(attack_sources)):
        mask = attack_sources == source
        source_rows.append(
            {
                "variant": variant["name"],
                "role": variant["role"],
                "source_file": source,
                "ddos_samples": int(np.sum(mask)),
                "detected_ddos": int(np.sum(full_attack_anomaly[mask])),
                "recall": float(np.mean(full_attack_anomaly[mask])),
            }
        )
    return result, source_rows


def prepare_and_evaluate_dual_view(
    raw_fit: dict[str, object],
    log_fit: dict[str, object],
    balanced_attack_indices: np.ndarray,
    attack_sources: np.ndarray,
) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    """Select and evaluate an OR ensemble using normal data only."""
    raw_calibration = np.asarray(raw_fit["calibration"])
    log_calibration = np.asarray(log_fit["calibration"])
    raw_kth = np.asarray(raw_fit["kth_distance"])
    log_kth = np.asarray(log_fit["kth_distance"])

    raw_cache: dict[float, tuple[float, np.ndarray, np.ndarray, np.ndarray]] = {}
    log_cache: dict[float, tuple[float, np.ndarray, np.ndarray, np.ndarray]] = {}
    for quantile in EPS_QUANTILES:
        raw_eps = float(np.quantile(raw_kth, quantile))
        raw_core = raw_calibration[raw_kth <= raw_eps]
        raw_cache[quantile] = (
            raw_eps,
            nearest_distances(raw_core, np.asarray(raw_fit["validation"])),
            nearest_distances(raw_core, np.asarray(raw_fit["test_normal"])),
            nearest_distances(raw_core, np.asarray(raw_fit["test_attack"])),
        )
        log_eps = float(np.quantile(log_kth, quantile))
        log_core = log_calibration[log_kth <= log_eps]
        log_cache[quantile] = (
            log_eps,
            nearest_distances(log_core, np.asarray(log_fit["validation"])),
            nearest_distances(log_core, np.asarray(log_fit["test_normal"])),
            nearest_distances(log_core, np.asarray(log_fit["test_attack"])),
        )

    grid_rows: list[dict[str, object]] = []
    for raw_quantile in EPS_QUANTILES:
        raw_eps, raw_validation, _, _ = raw_cache[raw_quantile]
        for log_quantile in EPS_QUANTILES:
            log_eps, log_validation, _, _ = log_cache[log_quantile]
            anomaly = (raw_validation > raw_eps) | (log_validation > log_eps)
            grid_rows.append(
                {
                    "variant": "dual_view_union",
                    "role": "recommended_enhanced_candidate",
                    "eps_quantile": "",
                    "eps": "",
                    "eps_view1": raw_eps,
                    "eps_view2": log_eps,
                    "eps_quantile_view1": raw_quantile,
                    "eps_quantile_view2": log_quantile,
                    "core_points": "",
                    "validation_samples": len(anomaly),
                    "validation_false_positives": int(np.sum(anomaly)),
                    "validation_false_positive_rate": float(np.mean(anomaly)),
                }
            )

    eligible = [
        row
        for row in grid_rows
        if float(row["validation_false_positive_rate"]) <= TARGET_VALIDATION_FPR
    ]
    if eligible:
        selected = min(
            eligible,
            key=lambda row: (
                float(row["eps_quantile_view1"]) + float(row["eps_quantile_view2"]),
                -float(row["validation_false_positive_rate"]),
            ),
        )
        selection_status = "met_validation_fpr_target"
    else:
        selected = min(grid_rows, key=lambda row: float(row["validation_false_positive_rate"]))
        selection_status = "fallback_lowest_validation_fpr"

    raw_quantile = float(selected["eps_quantile_view1"])
    log_quantile = float(selected["eps_quantile_view2"])
    raw_eps, _, raw_normal, raw_attack = raw_cache[raw_quantile]
    log_eps, _, log_normal, log_attack = log_cache[log_quantile]
    normal_anomaly = (raw_normal > raw_eps) | (log_normal > log_eps)
    attack_anomaly = (raw_attack > raw_eps) | (log_attack > log_eps)
    balanced = confusion_metrics(normal_anomaly, attack_anomaly[balanced_attack_indices])

    result = {
        "variant": "dual_view_union",
        "role": "recommended_enhanced_candidate",
        "numeric_feature_count": 7,
        "effective_dimensions": "3 + 7 separate views",
        "transform": "raw3_OR_log1p7",
        "uses_protocol": False,
        "eps": "",
        "eps_view1": raw_eps,
        "eps_view2": log_eps,
        "eps_quantile": "",
        "eps_quantile_view1": raw_quantile,
        "eps_quantile_view2": log_quantile,
        "min_samples": MIN_SAMPLES,
        "selection_status": selection_status,
        "validation_false_positive_rate": selected["validation_false_positive_rate"],
        "calibration_samples": len(raw_calibration),
        "core_points": "two separate core sets",
        "test_normal_samples": len(normal_anomaly),
        "balanced_test_attack_samples": len(balanced_attack_indices),
        "full_ddos_samples": len(attack_anomaly),
        **balanced,
        "full_ddos_detected": int(np.sum(attack_anomaly)),
        "full_ddos_recall": float(np.mean(attack_anomaly)),
    }
    source_rows: list[dict[str, object]] = []
    for source in sorted(set(attack_sources)):
        mask = attack_sources == source
        source_rows.append(
            {
                "variant": "dual_view_union",
                "role": "recommended_enhanced_candidate",
                "source_file": source,
                "ddos_samples": int(np.sum(mask)),
                "detected_ddos": int(np.sum(attack_anomaly[mask])),
                "recall": float(np.mean(attack_anomaly[mask])),
            }
        )
    return result, grid_rows, source_rows


def percent(value: object) -> str:
    return f"{100 * float(value):.2f}%"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    normal = load_rows(NORMAL_FILE, "Normal")
    attacks = [load_rows(path, "DDoS") for path in ATTACK_FILES]

    normal_features = np.asarray(normal["features"])
    normal_flow_ids = np.asarray(normal["flow_ids"])
    normal_protocols = np.asarray(normal["protocols"])
    attack_features = np.vstack([np.asarray(item["features"]) for item in attacks])
    attack_protocols = np.concatenate([np.asarray(item["protocols"]) for item in attacks])
    attack_sources = np.concatenate(
        [np.full(len(item["features"]), Path(item["path"]).name, dtype=object) for item in attacks]
    )

    train, validation, test = grouped_normal_split(normal_flow_ids)
    partitions = [set(normal_flow_ids[index]) for index in (train, validation, test)]
    if partitions[0] & partitions[1] or partitions[0] & partitions[2] or partitions[1] & partitions[2]:
        raise SystemExit("ERROR: Flow ID leakage detected between normal partitions")

    rng = np.random.default_rng(RANDOM_SEED)
    balanced_attack_indices = np.sort(
        rng.choice(len(attack_features), size=len(test), replace=False)
    )

    definitions: list[dict[str, object]] = []
    metrics: list[dict[str, object]] = []
    validation_rows: list[dict[str, object]] = []
    source_rows: list[dict[str, object]] = []
    fits: dict[str, dict[str, object]] = {}

    for variant in VARIANTS:
        definitions.append(
            {
                "variant": variant["name"],
                "role": variant["role"],
                "numeric_features": "; ".join(variant["features"]),
                "numeric_feature_count": len(tuple(variant["features"])),
                "transform": variant["transform"],
                "protocol_encoding": "one-hot: 0, 6, 17, other" if variant["protocol"] else "none",
                "selection_rule": "smallest eps with normal-validation FPR <= 5%",
            }
        )
        fit = prepare_variant(
            variant,
            normal_features,
            attack_features,
            normal_protocols,
            attack_protocols,
            train,
            validation,
            test,
        )
        fits[str(variant["name"])] = fit
        validation_rows.extend(fit["validation_rows"])
        result, by_source = evaluate_variant(
            variant, fit, balanced_attack_indices, attack_sources
        )
        metrics.append(result)
        source_rows.extend(by_source)
        print(
            f"{variant['name']}: eps={result['eps']:.8g}, "
            f"FPR={percent(result['false_positive_rate'])}, "
            f"full recall={percent(result['full_ddos_recall'])}"
        )

    dual_result, dual_validation, dual_sources = prepare_and_evaluate_dual_view(
        fits["baseline3_raw"],
        fits["counts7_log1p"],
        balanced_attack_indices,
        attack_sources,
    )
    definitions.append(
        {
            "variant": "dual_view_union",
            "role": "recommended_enhanced_candidate",
            "numeric_features": "baseline3_raw OR counts7_log1p anomaly decision",
            "numeric_feature_count": 7,
            "transform": "raw3_OR_log1p7",
            "protocol_encoding": "none",
            "selection_rule": "most sensitive quantile pair with union normal-validation FPR <= 5%",
        }
    )
    metrics.append(dual_result)
    validation_rows.extend(dual_validation)
    source_rows.extend(dual_sources)
    print(
        f"dual_view_union: eps=({dual_result['eps_view1']:.8g}, "
        f"{dual_result['eps_view2']:.8g}), "
        f"FPR={percent(dual_result['false_positive_rate'])}, "
        f"full recall={percent(dual_result['full_ddos_recall'])}"
    )

    # Verify the core-point shortcut against an exact DBSCAN fit for the main candidate.
    main_fit = fits["counts7_log1p"]
    model = DBSCAN(
        eps=float(main_fit["selected_eps"]),
        min_samples=MIN_SAMPLES,
        metric="euclidean",
        algorithm="auto",
        n_jobs=-1,
    ).fit(np.asarray(main_fit["calibration"]))
    exact_core = np.zeros(len(main_fit["calibration"]), dtype=bool)
    exact_core[model.core_sample_indices_] = True
    shortcut_core = np.asarray(main_fit["kth_distance"]) <= float(main_fit["selected_eps"])
    if not np.array_equal(exact_core, shortcut_core):
        raise SystemExit("ERROR: nearest-neighbor core rule did not match DBSCAN")

    write_rows(OUTPUT_DIR / "feature_definitions.csv", definitions)
    write_rows(OUTPUT_DIR / "feature_set_validation.csv", validation_rows)
    write_rows(OUTPUT_DIR / "feature_set_metrics.csv", metrics)
    write_rows(OUTPUT_DIR / "feature_set_source_recall.csv", source_rows)

    labels = [str(row["variant"]) for row in metrics]
    recalls = [float(row["full_ddos_recall"]) for row in metrics]
    fprs = [float(row["false_positive_rate"]) for row in metrics]
    x = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    recall_bars = ax.bar(x - width / 2, recalls, width, label="DDoS recall", color="#2F6B8A")
    fpr_bars = ax.bar(x + width / 2, fprs, width, label="Normal FPR", color="#B65F4A")
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Rate")
    ax.set_title("InSDN feature sensitivity: detection versus false alarms")
    ax.set_xticks(x, labels, rotation=18, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    for bars in (recall_bars, fpr_bars):
        for bar in bars:
            value = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.02, percent(value), ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "feature_set_metrics.png", dpi=180)
    plt.close(fig)

    source_names = ["OVS.csv", "metasploitable-2.csv"]
    source_map = {
        (str(row["variant"]), str(row["source_file"])): float(row["recall"])
        for row in source_rows
    }
    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    width = 0.36
    ovs_values = [source_map[(label, source_names[0])] for label in labels]
    meta_values = [source_map[(label, source_names[1])] for label in labels]
    ax.bar(x - width / 2, ovs_values, width, label="OVS", color="#8E4B3C")
    ax.bar(x + width / 2, meta_values, width, label="Metasploitable-2", color="#4F7C59")
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Recall")
    ax.set_title("InSDN DDoS recall by source group")
    ax.set_xticks(x, labels, rotation=18, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "source_recall_comparison.png", dpi=180)
    plt.close(fig)

    by_name = {str(row["variant"]): row for row in metrics}
    baseline = by_name["baseline3_raw"]
    log_view = by_name["counts7_log1p"]
    candidate = by_name["dual_view_union"]
    diagnostic = by_name["counts7_log1p_protocol"]
    candidate_sources = {
        str(row["source_file"]): row
        for row in source_rows
        if row["variant"] == "dual_view_union"
    }
    protocol_zero_normal = int(np.sum(normal_protocols == 0))
    protocol_zero_attack = int(np.sum(attack_protocols == 0))
    cluster_count = len(set(model.labels_) - {-1})
    noise_count = int(np.sum(model.labels_ == -1))

    report = f"""# گزارش آزمون حساسیت ویژگی‌های DBSCAN روی InSDN

## هدف

این آزمایش تکمیلی بررسی می‌کند که آیا افزودن آمار پایه جریان و تبدیل لگاریتمی می‌تواند ضعف تعمیم مدل سه‌ویژگی پایان‌نامه را روی دیتاست عمومی InSDN کاهش دهد. این تحلیل جایگزین آزمایش اصلی Mininet نیست و به‌عنوان ارزیابی خارجی و تحلیل حساسیت گزارش می‌شود.

## پروتکل بدون نشت اطلاعات

- پس از کنترل مشترک کیفیت، {len(normal_features):,} جریان عادی و {len(attack_features):,} جریان DDoS معتبر باقی ماند.
- داده عادی بر اساس `Flow ID` به {len(train):,} نمونه کالیبراسیون، {len(validation):,} نمونه اعتبارسنجی و {len(test):,} نمونه آزمون تقسیم شد و شناسه جریان میان بخش‌ها هم‌پوشانی نداشت.
- `StandardScaler`، نقاط هسته‌ای و انتخاب `eps` فقط با داده عادی انجام شدند.
- برای هر نسخه، کوچک‌ترین `eps` با FPR حداکثر پنج درصد روی اعتبارسنجی عادی انتخاب شد؛ برچسب DDoS در انتخاب پارامتر دخالت نداشت.
- ویژگی‌های زمانی مطابق فایل اصلی InSDN بر حسب میکروثانیه نگه داشته شدند. این انتخاب برای نسخه‌های لگاریتمی صریح و ضروری است، زیرا برخلاف استانداردسازی خام، `log1p` به واحد اندازه‌گیری وابسته است.
- برای Precision و F1 یک آزمون متوازن شامل {len(test):,} جریان عادی و همین تعداد DDoS استفاده شد؛ Recall نهایی علاوه بر آن روی همه {len(attack_features):,} جریان DDoS محاسبه شد.

## نتیجه مقایسه

| نسخه | ویژگی/تبدیل | FPR آزمون عادی | Recall همه DDoS | Precision متوازن | F1 متوازن | Balanced Accuracy |
|---|---|---:|---:|---:|---:|---:|
"""
    for row in metrics:
        report += (
            f"| `{row['variant']}` | {int(row['numeric_feature_count'])} ویژگی، {row['transform']}"
            f"{' + Protocol' if row['uses_protocol'] else ''} | {percent(row['false_positive_rate'])} | "
            f"{percent(row['full_ddos_recall'])} | {percent(row['precision'])} | "
            f"{percent(row['f1_score'])} | {percent(row['balanced_accuracy'])} |\n"
        )

    report += f"""

نسخه سه‌ویژگی پایه روی همه DDoSها Recall برابر {percent(baseline['full_ddos_recall'])} داشت. نمای هفت‌ویژگی لگاریتمی به‌تنهایی Recall کل {percent(log_view['full_ddos_recall'])} داشت، اما Recall آن برای OVS به‌طور محسوس افزایش یافت. چون دو نما الگوهای مکمل را تشخیص دادند، نسخه پیشنهادی `dual_view_union` یک جریان را زمانی ناهنجار می‌داند که حداقل یکی از دو DBSCAN آن را ناهنجار تشخیص دهد. Recall کل این نسخه {percent(candidate['full_ddos_recall'])} و FPR آزمون عادی آن {percent(candidate['false_positive_rate'])} شد. Recall آن برای OVS برابر {percent(candidate_sources['OVS.csv']['recall'])} و برای Metasploitable-2 برابر {percent(candidate_sources['metasploitable-2.csv']['recall'])} بود.

برای نمای هفت‌ویژگی، اجرای واقعی DBSCAN روی کالیبراسیون {cluster_count:,} خوشه، {int(log_view['core_points']):,} نقطه هسته‌ای و {noise_count:,} نقطه نویزی ایجاد کرد. برابری نقاط هسته‌ای DBSCAN با قاعده فاصله همسایه پنجم نیز به‌صورت برنامه‌ای کنترل شد. در مدل دو‌نما، آستانه هر نما جداگانه و فقط با اعتبارسنجی عادی انتخاب شد.

## چرا نتیجه Protocol نباید نتیجه اصلی باشد؟

نسخه تشخیصی شامل Protocol به Recall {percent(diagnostic['full_ddos_recall'])} رسید، اما در داده معتبر، Protocol صفر در {protocol_zero_attack:,} از {len(attack_protocols):,} نمونه DDoS و فقط {protocol_zero_normal:,} از {len(normal_protocols):,} نمونه عادی دیده شد. بنابراین Protocol می‌تواند نقش میان‌بُر وابسته به دیتاست داشته باشد. این نسخه فقط یک کران تشخیصی است و نباید به‌عنوان مدل اصلی یا شاهد تعمیم‌پذیری معرفی شود.

## تفسیر قابل دفاع

نتیجه نشان می‌دهد مشکل مدل پایه فقط از DBSCAN نیست؛ نمایش سه‌ویژگی برای پوشش همه الگوهای DDoS کافی نبوده است. نمای خام سه‌ویژگی و نمای هفت‌ویژگی لگاریتمی حساسیت‌های مکمل داشتند و اجتماع تصمیم آن‌ها پوشش حمله را بهتر کرد. با این حال، این نتیجه یک تحلیل تکمیلی روی یک تقسیم ثابت است و برای ادعای نهایی درباره پایداری بهتر است در ادامه روی چند تقسیم مستقل نیز تکرار شود.

## تغییر پیشنهادی در پیاده‌سازی

1. چهار ویژگی `Tot Fwd Pkts`، `Tot Bwd Pkts`، `TotLen Fwd Pkts` و `TotLen Bwd Pkts` به سه ویژگی فعلی افزوده شوند.
2. پیش از استانداردسازی، روی هر هفت ویژگی غیرمنفی تبدیل `log1p` اعمال شود؛ زمان‌ها باید به‌طور ثابت بر حسب میکروثانیه باشند (یا ثانیه‌ها ابتدا در یک میلیون ضرب شوند).
3. در OpenFlow، جریان‌های رفت و برگشت با پنج‌تایی مبدأ/مقصد/پورت/پروتکل جفت شوند تا شمارنده‌های دو جهت قابل محاسبه باشند.
4. `eps` برای هر محیط فقط با ترافیک عادی کالیبراسیون و اعتبارسنجی تعیین شود؛ مقدار 0.28 مستقیماً به دیتاست یا شبکه دیگر منتقل نشود.

## جمع‌بندی

نسخه مناسب برای ادامه پژوهش `dual_view_union` است که از دو DBSCAN عادی‌محور مکمل استفاده می‌کند. نسخه Protocol به علت خطر میان‌بُر داده‌ای کنار گذاشته می‌شود. پیشنهاد می‌شود این آزمایش به‌صورت «ارزیابی خارجی تکمیلی و تحلیل حساسیت ویژگی‌ها» به فصل چهارم افزوده شود، نه اینکه نتایج اصلی آزمایشگاه را حذف یا جایگزین کند.
"""
    (OUTPUT_DIR / "feature_sensitivity_report_fa.md").write_text(report, encoding="utf-8")
    print(f"Outputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
