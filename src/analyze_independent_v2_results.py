#!/usr/bin/env python3
"""Validate and report the clean independent DBSCAN experiment (v2)."""

from __future__ import annotations

import csv
import hashlib
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "independent_v2"
RESULTS_DIR = PROJECT_ROOT / "data" / "processed" / "results" / "independent_v2"
SELECTED_EPS = 0.28
MIN_SAMPLES = 5

RAW_FILES = (
    "calibration_normal_01.csv",
    "test_normal_01.csv",
    "test_high_rate_icmp_01.csv",
)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def write_rows(
    path: Path,
    fieldnames: list[str],
    rows: list[dict[str, object]],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def is_icmp(row: dict[str, str]) -> bool:
    return re.search(r"\bicmp\b", (row.get("match") or "").lower()) is not None


def as_float(row: dict[str, str], field: str) -> float:
    value = float(row[field])
    if not math.isfinite(value):
        raise ValueError(f"non-finite {field}")
    return value


def save_k_distance_plot(distances: np.ndarray) -> None:
    fig, axis = plt.subplots(figsize=(8.5, 5.0))
    ranks = np.arange(1, len(distances) + 1)
    axis.plot(ranks, distances, color="#174A7E", linewidth=2)
    axis.axhline(
        SELECTED_EPS,
        color="#C44E52",
        linestyle="--",
        linewidth=1.8,
        label=f"Selected eps = {SELECTED_EPS:.2f}",
    )
    axis.set_title("Calibration Normal: 5-distance curve")
    axis.set_xlabel("Sorted sample rank")
    axis.set_ylabel("Distance to 5th neighbor")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "independent_k_distance.png", dpi=180)
    plt.close(fig)


def save_metric_plot(metrics: list[dict[str, str]]) -> None:
    eps = np.asarray([float(row["eps"]) for row in metrics])
    series = (
        ("Precision", "precision", "#4C78A8"),
        ("Recall", "recall", "#F58518"),
        ("F1", "f1_score", "#54A24B"),
        ("Specificity", "specificity", "#B279A2"),
        ("Balanced accuracy", "balanced_accuracy", "#E45756"),
    )
    fig, axis = plt.subplots(figsize=(8.5, 5.0))
    for label, field, color in series:
        axis.plot(
            eps,
            [float(row[field]) for row in metrics],
            marker="o",
            linewidth=2,
            label=label,
            color=color,
        )
    axis.set_ylim(0.95, 1.005)
    axis.set_title("Independent detection metrics")
    axis.set_xlabel("eps")
    axis.set_ylabel("Score")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "independent_metrics_by_eps.png", dpi=180)
    plt.close(fig)


def save_distance_plot(predictions: list[dict[str, str]]) -> None:
    selected = [
        row
        for row in predictions
        if math.isclose(float(row["eps"]), SELECTED_EPS)
    ]
    normal = [
        float(row["nearest_core_distance"])
        for row in selected
        if row["scenario"] == "normal"
    ]
    high_rate = [
        float(row["nearest_core_distance"])
        for row in selected
        if row["scenario"] == "high_rate"
    ]
    fig, axis = plt.subplots(figsize=(8.5, 5.0))
    bins = np.logspace(-3, 2, 38)
    axis.hist(normal, bins=bins, alpha=0.72, label="Normal", color="#4C78A8")
    axis.hist(
        high_rate,
        bins=bins,
        alpha=0.72,
        label="High-Rate ICMP",
        color="#E45756",
    )
    axis.axvline(
        SELECTED_EPS,
        color="#222222",
        linestyle="--",
        linewidth=1.8,
        label=f"Threshold = {SELECTED_EPS:.2f}",
    )
    axis.set_xscale("log")
    axis.set_title("Distance to nearest calibration core point")
    axis.set_xlabel("Nearest-core distance (log scale)")
    axis.set_ylabel("Samples")
    axis.grid(alpha=0.2, which="both")
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "independent_distance_distribution.png", dpi=180)
    plt.close(fig)


def save_confusion_matrix(metric: dict[str, str]) -> None:
    matrix = np.asarray(
        [
            [int(metric["true_negative"]), int(metric["false_positive"])],
            [int(metric["false_negative"]), int(metric["true_positive"])],
        ]
    )
    fig, axis = plt.subplots(figsize=(5.5, 5.0))
    image = axis.imshow(matrix, cmap="Blues")
    for row in range(2):
        for column in range(2):
            axis.text(
                column,
                row,
                str(matrix[row, column]),
                ha="center",
                va="center",
                fontsize=15,
                color="white" if matrix[row, column] > matrix.max() / 2 else "black",
            )
    axis.set_xticks([0, 1], labels=["Predicted Normal", "Predicted anomaly"])
    axis.set_yticks([0, 1], labels=["Actual Normal", "Actual High-Rate"])
    axis.set_title(f"Independent confusion matrix (eps={SELECTED_EPS:.2f})")
    fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "independent_confusion_matrix.png", dpi=180)
    plt.close(fig)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    validations: list[dict[str, object]] = []
    raw_summary: dict[str, dict[str, object]] = {}

    for filename in RAW_FILES:
        path = RAW_DIR / filename
        rows = read_rows(path)
        icmp_rows = [row for row in rows if is_icmp(row)]
        rates = [as_float(row, "packets_rate") for row in icmp_rows]
        invalid_delta = sum(
            as_float(row, "packets_delta") < 0
            or as_float(row, "bytes_delta") < 0
            for row in rows
        )
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        raw_summary[filename] = {
            "total": len(rows),
            "icmp": len(icmp_rows),
            "flows": len({row["match"] for row in icmp_rows}),
            "rate_min": min(rates),
            "rate_median": float(np.median(rates)),
            "rate_max": max(rates),
            "sha256": digest,
        }
        validations.append(
            {
                "check": f"{filename}: valid non-negative deltas",
                "status": "PASS" if invalid_delta == 0 else "FAIL",
                "detail": f"invalid_or_negative={invalid_delta}",
            }
        )
        validations.append(
            {
                "check": f"{filename}: ICMP sample count",
                "status": "PASS" if len(icmp_rows) == 112 else "FAIL",
                "detail": f"icmp_rows={len(icmp_rows)}",
            }
        )

    for log_path in sorted(RAW_DIR.glob("*.collector.log")):
        has_error = "error" in log_path.read_text(
            encoding="utf-8", errors="replace"
        ).lower()
        validations.append(
            {
                "check": f"{log_path.name}: collector log",
                "status": "FAIL" if has_error else "PASS",
                "detail": "contains error" if has_error else "no errors",
            }
        )

    metrics = read_rows(RESULTS_DIR / "independent_eps_metrics.csv")
    predictions = read_rows(RESULTS_DIR / "independent_predictions.csv")
    selected_metric = next(
        row
        for row in metrics
        if math.isclose(float(row["eps"]), SELECTED_EPS)
    )
    selected_predictions = [
        row
        for row in predictions
        if math.isclose(float(row["eps"]), SELECTED_EPS)
    ]

    tp = sum(
        row["actual_label"] == "1" and row["predicted_anomaly"] == "1"
        for row in selected_predictions
    )
    fp = sum(
        row["actual_label"] == "0" and row["predicted_anomaly"] == "1"
        for row in selected_predictions
    )
    tn = sum(
        row["actual_label"] == "0" and row["predicted_anomaly"] == "0"
        for row in selected_predictions
    )
    fn = sum(
        row["actual_label"] == "1" and row["predicted_anomaly"] == "0"
        for row in selected_predictions
    )
    confusion_matches = all(
        (
            int(selected_metric[field]) == value
            for field, value in (
                ("true_positive", tp),
                ("false_positive", fp),
                ("true_negative", tn),
                ("false_negative", fn),
            )
        )
    )
    validations.append(
        {
            "check": "Selected-eps confusion matrix recomputation",
            "status": "PASS" if confusion_matches else "FAIL",
            "detail": f"TP={tp}, FP={fp}, TN={tn}, FN={fn}",
        }
    )

    metric_signatures = {
        (
            row["true_positive"],
            row["false_positive"],
            row["true_negative"],
            row["false_negative"],
        )
        for row in metrics
    }
    validations.append(
        {
            "check": "Stable result across predefined eps values",
            "status": "PASS" if len(metric_signatures) == 1 else "FAIL",
            "detail": f"unique_confusion_signatures={len(metric_signatures)}",
        }
    )

    normal_distances = sorted(
        float(row["nearest_core_distance"])
        for row in selected_predictions
        if row["scenario"] == "normal"
    )
    high_distances = sorted(
        float(row["nearest_core_distance"])
        for row in selected_predictions
        if row["scenario"] == "high_rate"
    )
    validations.append(
        {
            "check": "High-Rate distances exceed selected threshold",
            "status": "PASS" if min(high_distances) > SELECTED_EPS else "FAIL",
            "detail": f"minimum_high_rate_distance={min(high_distances):.6f}",
        }
    )

    validation_path = RESULTS_DIR / "independent_validation_summary.csv"
    write_rows(validation_path, ["check", "status", "detail"], validations)
    if any(row["status"] != "PASS" for row in validations):
        raise SystemExit("ERROR: one or more independent validation checks failed")

    write_rows(
        RESULTS_DIR / "selected_eps.csv",
        ["selected_eps", "min_samples", "selection_rule", "test_use"],
        [
            {
                "selected_eps": f"{SELECTED_EPS:.2f}",
                "min_samples": MIN_SAMPLES,
                "selection_rule": (
                    "smallest predefined eps in the stable calibration-supported plateau"
                ),
                "test_use": "labels used only after model fitting for evaluation",
            }
        ],
    )

    k_distance_rows = read_rows(RESULTS_DIR / "calibration_k_distance.csv")
    distance_field = next(
        field for field in k_distance_rows[0] if field.startswith("distance_to_")
    )
    k_distances = np.asarray(
        [float(row[distance_field]) for row in k_distance_rows]
    )
    save_k_distance_plot(k_distances)
    save_metric_plot(metrics)
    save_distance_plot(predictions)
    save_confusion_matrix(selected_metric)

    false_positives = [
        row
        for row in selected_predictions
        if row["scenario"] == "normal" and row["predicted_anomaly"] == "1"
    ]
    false_positive_time = ", ".join(
        sorted({row["timestamp"] for row in false_positives})
    )
    report = f"""# گزارش ارزیابی مستقل DBSCAN ـ نسخهٔ ۲

## طراحی ارزیابی

- آموزش StandardScaler و DBSCAN فقط با فایل Normal calibration انجام شد.
- فایل‌های Normal test و High-Rate ICMP test هنگام برازش مدل استفاده نشدند.
- نام سناریو و برچسب فقط پس از پیش‌بینی برای محاسبهٔ معیارها استفاده شد.
- پارامتر `min_samples={MIN_SAMPLES}` و مقادیر `eps` پیش از این اجرای تمیز تعیین شده بودند.
- مقدار منتخب `eps={SELECTED_EPS:.2f}` کوچک‌ترین مقدار در بازهٔ پایدار نتایج است؛ این انتخاب حساسیت را حفظ می‌کند و برچسب آزمون را وارد آموزش نمی‌کند.

## کیفیت دادهٔ خام

| فایل | کل ردیف‌ها | ICMP معتبر | Flowهای ICMP | نرخ بسته کمینه | میانه | بیشینه |
|---|---:|---:|---:|---:|---:|---:|
| Calibration Normal | {raw_summary['calibration_normal_01.csv']['total']} | {raw_summary['calibration_normal_01.csv']['icmp']} | {raw_summary['calibration_normal_01.csv']['flows']} | {raw_summary['calibration_normal_01.csv']['rate_min']:.3f} | {raw_summary['calibration_normal_01.csv']['rate_median']:.3f} | {raw_summary['calibration_normal_01.csv']['rate_max']:.3f} |
| Test Normal | {raw_summary['test_normal_01.csv']['total']} | {raw_summary['test_normal_01.csv']['icmp']} | {raw_summary['test_normal_01.csv']['flows']} | {raw_summary['test_normal_01.csv']['rate_min']:.3f} | {raw_summary['test_normal_01.csv']['rate_median']:.3f} | {raw_summary['test_normal_01.csv']['rate_max']:.3f} |
| Test High-Rate ICMP | {raw_summary['test_high_rate_icmp_01.csv']['total']} | {raw_summary['test_high_rate_icmp_01.csv']['icmp']} | {raw_summary['test_high_rate_icmp_01.csv']['flows']} | {raw_summary['test_high_rate_icmp_01.csv']['rate_min']:.3f} | {raw_summary['test_high_rate_icmp_01.csv']['rate_median']:.3f} | {raw_summary['test_high_rate_icmp_01.csv']['rate_max']:.3f} |

هر سه فایل ۱۱۲ نمونهٔ ICMP معتبر دارند. هیچ delta منفی یا خطای collector ثبت نشده است. شناسهٔ Flow در نسخهٔ ۲ از فیلدهای متغیر مانند `idle_age` مستقل است و نمونه‌های baseline/reset در نرخ وارد نشده‌اند.

## نتیجه برای eps منتخب

| معیار | مقدار |
|---|---:|
| TP | {tp} |
| FP | {fp} |
| TN | {tn} |
| FN | {fn} |
| Precision | {float(selected_metric['precision']):.4f} |
| Recall | {float(selected_metric['recall']):.4f} |
| F1 | {float(selected_metric['f1_score']):.4f} |
| FPR | {float(selected_metric['false_positive_rate']):.4f} |
| Specificity | {float(selected_metric['specificity']):.4f} |
| Balanced Accuracy | {float(selected_metric['balanced_accuracy']):.4f} |

تمام ۱۱۲ نمونهٔ High-Rate ICMP شناسایی شدند. از ۱۱۲ نمونهٔ Normal، دو نمونه مثبت کاذب بودند. هر دو در آخرین timestamp اجرای Normal (`{false_positive_time}`) و در مرحلهٔ توقف ترافیک با یک پنجرهٔ نیمه‌کامل قرار دارند. این دو نمونه برای ارائهٔ ارزیابی محافظه‌کارانه حذف نشده‌اند.

فاصلهٔ نمونه‌های High-Rate ICMP از نزدیک‌ترین core calibration حداقل {min(high_distances):.3f} است، درحالی‌که آستانه {SELECTED_EPS:.2f} است. نتیجه در تمام مقادیر از پیش تعیین‌شدهٔ `eps=0.28` تا `0.46` بدون تغییر باقی ماند.

## تفسیر

در این محیط آزمایشگاهی کنترل‌شده، مدل ساختار ترافیک Normal را از سناریوی High-Rate ICMP با جدایی بسیار زیاد تشخیص داده است. این نتیجه به معنای اثبات تشخیص عمومی DDoS نیست؛ دامنهٔ ادعا فقط سناریوی High-Rate ICMP، توپولوژی و تنظیمات ثبت‌شده در این آزمایش است.

## فایل‌های بازتولیدپذیری

- `run_manifest.csv`: زمان، میزبان‌ها، نرخ ping و تعداد نمونه‌ها
- `calibration_scaler_stats.csv`: پارامترهای scaler برازش‌شده فقط روی calibration
- `calibration_k_distance.csv`: منحنی 5-distance مجموعهٔ calibration
- `independent_predictions.csv`: پیش‌بینی ردیف‌به‌ردیف و فاصله تا نزدیک‌ترین core
- `independent_eps_metrics.csv`: معیارهای تمام epsهای از پیش تعیین‌شده
- `independent_validation_summary.csv`: کنترل‌های مستقل صحت داده و معیارها
"""
    (RESULTS_DIR / "independent_evaluation_report_fa.md").write_text(
        report,
        encoding="utf-8",
    )

    print(f"Validation checks passed: {len(validations)}")
    print(f"Selected eps: {SELECTED_EPS:.2f}")
    print(f"Confusion matrix: TP={tp}, FP={fp}, TN={tn}, FN={fn}")
    for name in (
        "independent_validation_summary.csv",
        "selected_eps.csv",
        "independent_k_distance.png",
        "independent_metrics_by_eps.png",
        "independent_distance_distribution.png",
        "independent_confusion_matrix.png",
        "independent_evaluation_report_fa.md",
    ):
        print(f"Created: {RESULTS_DIR / name}")


if __name__ == "__main__":
    main()
