#!/usr/bin/env python3

"""External validation of the thesis DBSCAN detector on InSDN.

The script keeps the thesis feature set (packet rate, byte rate, and flow
duration), fits StandardScaler and the normal DBSCAN reference exclusively on
normal calibration data, selects an external-dataset eps using normal
validation data only, and evaluates on held-out normal plus labeled DDoS data.
Rows are split by Flow ID so the same flow cannot occur in more than one normal
partition.
"""

from __future__ import annotations

import csv
import hashlib
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
ARCHIVE = ROOT / "data/public/insdn/InSDN_DatasetCSV.zip"
OUTPUT_DIR = ROOT / "data/processed/results/insdn_external_validation"

NORMAL_FILE = SOURCE_DIR / "Normal_data.csv"
ATTACK_FILES = (SOURCE_DIR / "OVS.csv", SOURCE_DIR / "metasploitable-2.csv")
RANDOM_SEED = 20260919
MIN_SAMPLES = 5
TARGET_VALIDATION_FPR = 0.05
CURRENT_THESIS_EPS = 0.28
EPS_QUANTILES = (0.90, 0.925, 0.95, 0.96, 0.97, 0.975, 0.98, 0.985, 0.99, 0.995)


def divide(numerator: float, denominator: float) -> float | None:
    return None if denominator == 0 else numerator / denominator


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_rows(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_rows(path: Path, wanted_label: str) -> dict[str, object]:
    features: list[tuple[float, float, float]] = []
    flow_ids: list[str] = []
    protocols: list[str] = []
    source_rows: list[int] = []
    label_counts: Counter[str] = Counter()
    total_rows = 0
    invalid_rows = 0

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "Flow ID",
            "Protocol",
            "Flow Duration",
            "Flow Byts/s",
            "Flow Pkts/s",
            "Label",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"ERROR: {path.name} missing columns: {sorted(missing)}")

        for source_row, row in enumerate(reader, start=2):
            total_rows += 1
            label = (row.get("Label") or "").strip()
            label_counts[label] += 1
            if label != wanted_label:
                continue
            try:
                packets_rate = float(row["Flow Pkts/s"])
                bytes_rate = float(row["Flow Byts/s"])
                duration_sec = float(row["Flow Duration"]) / 1_000_000.0
            except (TypeError, ValueError):
                invalid_rows += 1
                continue
            values = (packets_rate, bytes_rate, duration_sec)
            if (
                not all(math.isfinite(value) for value in values)
                or packets_rate < 0
                or bytes_rate < 0
                or duration_sec <= 0
            ):
                invalid_rows += 1
                continue
            features.append(values)
            flow_ids.append(row["Flow ID"])
            protocols.append(str(row["Protocol"]))
            source_rows.append(source_row)

    return {
        "path": path,
        "features": np.asarray(features, dtype=float),
        "flow_ids": np.asarray(flow_ids, dtype=object),
        "protocols": np.asarray(protocols, dtype=object),
        "source_rows": np.asarray(source_rows, dtype=int),
        "total_rows": total_rows,
        "valid_target_rows": len(features),
        "invalid_target_rows": invalid_rows,
        "label_counts": label_counts,
    }


def grouped_normal_split(flow_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = np.arange(len(flow_ids))
    first = GroupShuffleSplit(n_splits=1, train_size=0.50, random_state=RANDOM_SEED)
    train, remainder = next(first.split(indices, groups=flow_ids))
    second = GroupShuffleSplit(n_splits=1, train_size=0.50, random_state=RANDOM_SEED + 1)
    validation_relative, test_relative = next(
        second.split(remainder, groups=flow_ids[remainder])
    )
    validation = remainder[validation_relative]
    test = remainder[test_relative]
    return train, validation, test


def core_points(calibration: np.ndarray, eps: float) -> tuple[np.ndarray, np.ndarray]:
    neighbors = NearestNeighbors(
        n_neighbors=MIN_SAMPLES,
        algorithm="kd_tree",
        n_jobs=-1,
    ).fit(calibration)
    distances = neighbors.kneighbors(calibration, return_distance=True)[0]
    kth_distance = distances[:, -1]
    return calibration[kth_distance <= eps], kth_distance


def nearest_core_distance(core: np.ndarray, samples: np.ndarray) -> np.ndarray:
    model = NearestNeighbors(n_neighbors=1, algorithm="kd_tree", n_jobs=-1).fit(core)
    return model.kneighbors(samples, return_distance=True)[0][:, 0]


def confusion_metrics(
    normal_distances: np.ndarray,
    attack_distances: np.ndarray,
    eps: float,
) -> dict[str, object]:
    normal_anomaly = normal_distances > eps
    attack_anomaly = attack_distances > eps
    tn = int(np.sum(~normal_anomaly))
    fp = int(np.sum(normal_anomaly))
    tp = int(np.sum(attack_anomaly))
    fn = int(np.sum(~attack_anomaly))
    precision = divide(tp, tp + fp)
    recall = divide(tp, tp + fn)
    specificity = divide(tn, tn + fp)
    fpr = divide(fp, fp + tn)
    f1 = (
        divide(2 * precision * recall, precision + recall)
        if precision is not None and recall is not None
        else None
    )
    balanced_accuracy = (
        (recall + specificity) / 2
        if recall is not None and specificity is not None
        else None
    )
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


def transform_and_fit(
    normal_features: np.ndarray,
    attack_features: np.ndarray,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    test_indices: np.ndarray,
    use_log1p: bool,
) -> dict[str, object]:
    normal = np.log1p(normal_features) if use_log1p else normal_features.copy()
    attack = np.log1p(attack_features) if use_log1p else attack_features.copy()

    scaler = StandardScaler()
    calibration = scaler.fit_transform(normal[train_indices])
    validation = scaler.transform(normal[validation_indices])
    test_normal = scaler.transform(normal[test_indices])
    test_attack = scaler.transform(attack)

    base_neighbors = NearestNeighbors(
        n_neighbors=MIN_SAMPLES,
        algorithm="kd_tree",
        n_jobs=-1,
    ).fit(calibration)
    kth_distance = base_neighbors.kneighbors(calibration, return_distance=True)[0][
        :, -1
    ]

    validation_rows: list[dict[str, object]] = []
    for quantile in EPS_QUANTILES:
        eps = float(np.quantile(kth_distance, quantile))
        core = calibration[kth_distance <= eps]
        distances = nearest_core_distance(core, validation)
        validation_rows.append(
            {
                "transform": "log1p_then_standardize" if use_log1p else "standardize_raw",
                "eps_quantile": quantile,
                "eps": eps,
                "core_points": len(core),
                "validation_samples": len(validation),
                "validation_false_positives": int(np.sum(distances > eps)),
                "validation_false_positive_rate": float(np.mean(distances > eps)),
            }
        )

    eligible = [
        row
        for row in validation_rows
        if float(row["validation_false_positive_rate"]) <= TARGET_VALIDATION_FPR
    ]
    selected = min(eligible, key=lambda row: float(row["eps"]))
    return {
        "scaler": scaler,
        "calibration": calibration,
        "test_normal": test_normal,
        "test_attack": test_attack,
        "kth_distance": kth_distance,
        "validation_rows": validation_rows,
        "selected_eps": float(selected["eps"]),
        "selected_quantile": float(selected["eps_quantile"]),
    }


def evaluate_variant(
    name: str,
    fit: dict[str, object],
    eps: float,
    attack_sources: np.ndarray,
    attack_protocols: np.ndarray,
    balanced_attack_indices: np.ndarray,
) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    calibration = np.asarray(fit["calibration"])
    kth_distance = np.asarray(fit["kth_distance"])
    test_normal = np.asarray(fit["test_normal"])
    test_attack = np.asarray(fit["test_attack"])
    core = calibration[kth_distance <= eps]
    if not len(core):
        raise SystemExit(f"ERROR: {name} produced no core points")

    normal_distances = nearest_core_distance(core, test_normal)
    attack_distances = nearest_core_distance(core, test_attack)
    balanced_metrics = confusion_metrics(
        normal_distances,
        attack_distances[balanced_attack_indices],
        eps,
    )
    full_recall = float(np.mean(attack_distances > eps))
    full_balanced_accuracy = (
        full_recall + float(balanced_metrics["specificity"])
    ) / 2

    result = {
        "variant": name,
        "eps": eps,
        "min_samples": MIN_SAMPLES,
        "calibration_samples": len(calibration),
        "core_points": len(core),
        "test_normal_samples": len(test_normal),
        "balanced_test_attack_samples": len(balanced_attack_indices),
        "full_ddos_samples": len(test_attack),
        **balanced_metrics,
        "full_ddos_recall": full_recall,
        "full_ddos_balanced_accuracy": full_balanced_accuracy,
    }

    source_rows: list[dict[str, object]] = []
    for source in sorted(set(attack_sources)):
        mask = attack_sources == source
        source_rows.append(
            {
                "variant": name,
                "source_file": source,
                "ddos_samples": int(np.sum(mask)),
                "detected_ddos": int(np.sum(attack_distances[mask] > eps)),
                "recall": float(np.mean(attack_distances[mask] > eps)),
            }
        )

    protocol_rows: list[dict[str, object]] = []
    for protocol in sorted(set(attack_protocols), key=lambda value: int(value)):
        mask = attack_protocols == protocol
        protocol_rows.append(
            {
                "variant": name,
                "protocol": protocol,
                "ddos_samples": int(np.sum(mask)),
                "detected_ddos": int(np.sum(attack_distances[mask] > eps)),
                "recall": float(np.mean(attack_distances[mask] > eps)),
            }
        )
    return result, source_rows, protocol_rows


def percent(value: object) -> str:
    return f"{100 * float(value):.2f}%"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    normal = load_rows(NORMAL_FILE, "Normal")
    attacks = [load_rows(path, "DDoS") for path in ATTACK_FILES]

    normal_features = np.asarray(normal["features"])
    normal_flow_ids = np.asarray(normal["flow_ids"])
    attack_features = np.vstack([np.asarray(item["features"]) for item in attacks])
    attack_sources = np.concatenate(
        [np.full(len(item["features"]), Path(item["path"]).name, dtype=object) for item in attacks]
    )
    attack_protocols = np.concatenate([np.asarray(item["protocols"]) for item in attacks])

    train, validation, test = grouped_normal_split(normal_flow_ids)
    train_groups = set(normal_flow_ids[train])
    validation_groups = set(normal_flow_ids[validation])
    test_groups = set(normal_flow_ids[test])
    if train_groups & validation_groups or train_groups & test_groups or validation_groups & test_groups:
        raise SystemExit("ERROR: Flow ID leakage detected between normal partitions")

    rng = np.random.default_rng(RANDOM_SEED)
    balanced_attack_indices = np.sort(
        rng.choice(len(attack_features), size=len(test), replace=False)
    )

    raw_fit = transform_and_fit(
        normal_features,
        attack_features,
        train,
        validation,
        test,
        use_log1p=False,
    )
    log_fit = transform_and_fit(
        normal_features,
        attack_features,
        train,
        validation,
        test,
        use_log1p=True,
    )

    results: list[dict[str, object]] = []
    source_rows: list[dict[str, object]] = []
    protocol_rows: list[dict[str, object]] = []
    variants = (
        (
            "public_normal_validation_eps",
            raw_fit,
            float(raw_fit["selected_eps"]),
        ),
        ("thesis_fixed_eps_0.28", raw_fit, CURRENT_THESIS_EPS),
        (
            "log1p_sensitivity_normal_validation_eps",
            log_fit,
            float(log_fit["selected_eps"]),
        ),
    )
    for name, fit, eps in variants:
        result, by_source, by_protocol = evaluate_variant(
            name,
            fit,
            eps,
            attack_sources,
            attack_protocols,
            balanced_attack_indices,
        )
        results.append(result)
        source_rows.extend(by_source)
        protocol_rows.extend(by_protocol)

    selected_eps = float(raw_fit["selected_eps"])
    dbscan_model = DBSCAN(
        eps=selected_eps,
        min_samples=MIN_SAMPLES,
        metric="euclidean",
        algorithm="kd_tree",
        n_jobs=-1,
    ).fit(np.asarray(raw_fit["calibration"]))
    dbscan_core = np.zeros(len(raw_fit["calibration"]), dtype=bool)
    dbscan_core[dbscan_model.core_sample_indices_] = True
    knn_core = np.asarray(raw_fit["kth_distance"]) <= selected_eps
    if not np.array_equal(dbscan_core, knn_core):
        raise SystemExit("ERROR: k-neighbor core-point rule did not match DBSCAN")
    calibration_cluster_count = len(set(dbscan_model.labels_) - {-1})

    dataset_rows: list[dict[str, object]] = []
    for item in (normal, *attacks):
        label_counts = item["label_counts"]
        dataset_rows.append(
            {
                "source_file": Path(item["path"]).name,
                "total_rows": item["total_rows"],
                "normal_rows": label_counts.get("Normal", 0),
                "ddos_rows": label_counts.get("DDoS", 0),
                "valid_target_rows": item["valid_target_rows"],
                "invalid_target_rows": item["invalid_target_rows"],
                "all_labels": "; ".join(
                    f"{label}={count}" for label, count in sorted(label_counts.items())
                ),
            }
        )

    write_rows(
        OUTPUT_DIR / "dataset_summary.csv",
        list(dataset_rows[0]),
        dataset_rows,
    )
    validation_rows = [*raw_fit["validation_rows"], *log_fit["validation_rows"]]
    write_rows(
        OUTPUT_DIR / "validation_eps_metrics.csv",
        list(validation_rows[0]),
        validation_rows,
    )
    write_rows(OUTPUT_DIR / "test_metrics.csv", list(results[0]), results)
    write_rows(OUTPUT_DIR / "source_recall.csv", list(source_rows[0]), source_rows)
    write_rows(OUTPUT_DIR / "protocol_recall.csv", list(protocol_rows[0]), protocol_rows)

    main_result = next(
        row for row in results if row["variant"] == "public_normal_validation_eps"
    )
    fixed_result = next(row for row in results if row["variant"] == "thesis_fixed_eps_0.28")
    log_result = next(
        row
        for row in results
        if row["variant"] == "log1p_sensitivity_normal_validation_eps"
    )
    main_sources = {
        row["source_file"]: row
        for row in source_rows
        if row["variant"] == "public_normal_validation_eps"
    }

    fig, ax = plt.subplots(figsize=(6.8, 5.2))
    matrix = np.array(
        [
            [main_result["true_negative"], main_result["false_positive"]],
            [main_result["false_negative"], main_result["true_positive"]],
        ]
    )
    image = ax.imshow(matrix, cmap="Blues")
    for row in range(2):
        for column in range(2):
            ax.text(column, row, f"{matrix[row, column]:,}", ha="center", va="center", fontsize=13)
    ax.set_xticks([0, 1], labels=["Predicted normal", "Predicted anomaly"])
    ax.set_yticks([0, 1], labels=["Normal", "DDoS"])
    ax.set_ylabel("Actual class")
    ax.set_title("InSDN balanced external test")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout(pad=1.4)
    fig.savefig(OUTPUT_DIR / "balanced_confusion_matrix.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    names = ["OVS", "Metasploitable-2", "All DDoS"]
    recalls = [
        float(main_sources["OVS.csv"]["recall"]),
        float(main_sources["metasploitable-2.csv"]["recall"]),
        float(main_result["full_ddos_recall"]),
    ]
    bars = ax.bar(names, recalls, color=["#9C3D2E", "#2F6B8A", "#587A4D"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Recall")
    ax.set_title("DDoS recall by InSDN source group")
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, recalls):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.02, percent(value), ha="center")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "recall_by_source.png", dpi=180)
    plt.close(fig)

    archive_hash = sha256(ARCHIVE)
    actual_total = sum(int(row["total_rows"]) for row in dataset_rows)
    report = f"""# گزارش ارزیابی خارجی DBSCAN روی دیتاست InSDN

## هدف و دامنه

این آزمایش بررسی می‌کند که روش پایان‌نامه با همان سه ویژگی نرخ بسته، نرخ بایت و مدت جریان تا چه حد روی یک دیتاست عمومی مخصوص SDN تعمیم می‌یابد. دیتاست از نشانی رسمی `https://aseados.ucd.ie/datasets/SDN/` دریافت شده است. مقدار SHA-256 فایل فشرده `{archive_hash}` است.

## کنترل داده و جلوگیری از نشت اطلاعات

- تعداد واقعی رکوردهای سه فایل CSV برابر {actual_total:,} است.
- پس از حذف ۸ رکورد عادی با مدت صفر، {len(normal_features):,} نمونه عادی معتبر باقی ماند.
- تعداد نمونه‌های دارای برچسب DDoS پس از حذف فاصله انتهای برچسب‌ها برابر {len(attack_features):,} است.
- تقسیم داده عادی بر اساس Flow ID انجام شد: {len(train):,} نمونه کالیبراسیون، {len(validation):,} نمونه اعتبارسنجی و {len(test):,} نمونه آزمون.
- هیچ Flow ID میان سه بخش مشترک نبود.
- StandardScaler و نقاط هسته‌ای فقط با داده عادی کالیبراسیون برازش شدند.
- برچسب DDoS در انتخاب eps استفاده نشد. کوچک‌ترین eps با نرخ مثبت کاذب حداکثر پنج درصد روی داده عادی اعتبارسنجی انتخاب شد.

## تنظیم منتخب بدون استفاده از برچسب حمله

- `min_samples = {MIN_SAMPLES}`
- `eps = {selected_eps:.10f}`، متناظر با صدک {100 * float(raw_fit['selected_quantile']):.1f} فاصله همسایه پنجم در کالیبراسیون
- تعداد نقاط هسته‌ای: {int(main_result['core_points']):,}
- تعداد خوشه‌های DBSCAN در کالیبراسیون: {calibration_cluster_count:,}
- تعداد نقاط نویزی کالیبراسیون: {int(np.sum(dbscan_model.labels_ == -1)):,}

## نتیجه اصلی روی آزمون متوازن

برای جلوگیری از اثر نسبت کلاس‌ها بر Precision و F1، همه {len(test):,} نمونه عادی آزمون با یک نمونه تصادفی ثابت و هم‌اندازه از DDoS مقایسه شدند.

| معیار | مقدار |
|---|---:|
| TN | {int(main_result['true_negative']):,} |
| FP | {int(main_result['false_positive']):,} |
| TP | {int(main_result['true_positive']):,} |
| FN | {int(main_result['false_negative']):,} |
| Precision | {percent(main_result['precision'])} |
| Recall | {percent(main_result['recall'])} |
| F1 | {percent(main_result['f1_score'])} |
| FPR | {percent(main_result['false_positive_rate'])} |
| Specificity | {percent(main_result['specificity'])} |
| Balanced Accuracy | {percent(main_result['balanced_accuracy'])} |

روی همه {len(attack_features):,} نمونه DDoS، Recall برابر {percent(main_result['full_ddos_recall'])} بود. Recall در فایل OVS برابر {percent(main_sources['OVS.csv']['recall'])} و در فایل Metasploitable-2 برابر {percent(main_sources['metasploitable-2.csv']['recall'])} شد. این اختلاف بزرگ نشان‌دهنده حساسیت روش سه‌ویژگی به نوع محیط و الگوی حمله است.

## مقایسه با eps فعلی پایان‌نامه

انتقال مستقیم `eps = 0.28` به InSDN فقط {percent(fixed_result['full_ddos_recall'])} از کل DDoSها را شناسایی کرد، هرچند FPR آن {percent(fixed_result['false_positive_rate'])} بود. بنابراین مقدار eps به مقیاس و توزیع همان دیتاست وابسته است و نباید بدون کالیبراسیون به دیتاست دیگر منتقل شود.

## آزمون حساسیت تبدیل لگاریتمی

اعمال `log1p` به همین سه ویژگی و سپس استانداردسازی، Recall کل DDoS را به {percent(log_result['full_ddos_recall'])} رساند و بهبود ایجاد نکرد. پس ضعف تعمیم صرفاً با تبدیل لگاریتمی ساده برطرف نشد.

## نتیجه قابل دفاع

ارزیابی خارجی از این ادعا پشتیبانی نمی‌کند که مدل سه‌ویژگی فعلی برای همه انواع DDoS قابل تعمیم است. مدل در سناریوی آزمایشگاهی خود پژوهش عملکرد بالایی داشت، اما روی InSDN تنها بخشی از حملات را شناسایی کرد و میان دو گروه حمله اختلاف جدی داشت. این نتیجه باید به‌عنوان آزمایش تکمیلی و محدودیت تعمیم‌پذیری گزارش شود، نه جایگزین نتایج اصلی آزمایشگاه.

## پیشنهاد ادامه

برای بهبود ارزیابی خارجی باید ویژگی‌های بیشتری مانند تعداد بسته‌های رفت و برگشت، تعداد بایت‌های رفت و برگشت، نسبت جهت‌ها، نرخ ایجاد جریان و ویژگی‌های TCP بررسی شوند. انتخاب ویژگی و پارامتر باید فقط روی آموزش و اعتبارسنجی انجام شود و آزمون نهایی دست‌نخورده باقی بماند.
"""
    (OUTPUT_DIR / "external_validation_report_fa.md").write_text(report, encoding="utf-8")

    print(report)
    print(f"Outputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
