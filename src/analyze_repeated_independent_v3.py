#!/usr/bin/env python3
"""Evaluate and aggregate confirmatory independent DBSCAN repetitions.

Repetition 01 is treated as the pilot used to freeze eps=0.28 and
min_samples=5.  Confirmatory estimates are calculated only from repetitions
02 onward.  Each repetition fits its scaler and DBSCAN model exclusively on
that repetition's Normal calibration data, then evaluates its Normal and
High-Rate ICMP test data.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, median, stdev

import matplotlib.pyplot as plt
import numpy as np


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_RAW_ROOT = BASE_DIR / "data" / "raw" / "independent_repeats"
DEFAULT_RESULTS_DIR = (
    BASE_DIR / "data" / "processed" / "results" / "independent_repeated_v3"
)
EVALUATOR = BASE_DIR / "evaluate_independent_dbscan_v1.py"
EPS_VALUES = (0.28, 0.31, 0.35, 0.40, 0.43, 0.46)
SELECTED_EPS = 0.28
MIN_SAMPLES = 5
METRIC_FIELDS = (
    "precision",
    "recall",
    "f1_score",
    "false_positive_rate",
    "specificity",
    "balanced_accuracy",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate confirmatory DBSCAN repetitions 02 onward.",
    )
    parser.add_argument("--start-repeat", type=int, default=2)
    parser.add_argument("--end-repeat", type=int, default=5)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def write_rows(
    path: Path,
    fieldnames: list[str] | tuple[str, ...],
    rows: list[dict[str, object]],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def is_icmp(row: dict[str, str]) -> bool:
    return "icmp" in (row.get("match") or "").lower()


def safe_divide(numerator: int | float, denominator: int | float) -> float:
    return float("nan") if denominator == 0 else numerator / denominator


def derived_metrics(tp: int, fp: int, tn: int, fn: int) -> dict[str, float]:
    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    specificity = safe_divide(tn, tn + fp)
    fpr = safe_divide(fp, fp + tn)
    f1 = safe_divide(2 * precision * recall, precision + recall)
    balanced = (recall + specificity) / 2
    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "false_positive_rate": fpr,
        "specificity": specificity,
        "balanced_accuracy": balanced,
    }


def t_critical_95(sample_count: int) -> float:
    values = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571}
    return values.get(sample_count, 1.96)


def summarize(values: list[float]) -> dict[str, float]:
    avg = mean(values)
    sd = stdev(values) if len(values) > 1 else 0.0
    half_width = t_critical_95(len(values)) * sd / math.sqrt(len(values))
    return {
        "mean": avg,
        "sd": sd,
        "ci95_low": max(0.0, avg - half_width),
        "ci95_high": min(1.0, avg + half_width),
        "min": min(values),
        "max": max(values),
    }


def repeat_paths(raw_root: Path, repeat_id: int) -> dict[str, Path]:
    run_dir = raw_root / f"run_{repeat_id:02d}"
    suffix = f"{repeat_id:02d}"
    return {
        "calibration": run_dir / f"calibration_normal_{suffix}.csv",
        "normal": run_dir / f"test_normal_{suffix}.csv",
        "high_rate": run_dir / f"test_high_rate_icmp_{suffix}.csv",
        "manifest": run_dir / "run_manifest.csv",
    }


def validate_and_summarize_raw(
    repeat_id: int,
    scenario: str,
    path: Path,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    rows = read_rows(path)
    icmp = [row for row in rows if is_icmp(row)]
    packets_rate = [float(row["packets_rate"]) for row in icmp]
    bytes_rate = [float(row["bytes_rate"]) for row in icmp]
    durations = [float(row["duration_sec"]) for row in icmp]
    invalid_delta = sum(
        float(row["packets_delta"]) < 0 or float(row["bytes_delta"]) < 0
        for row in rows
    )
    finite = all(
        math.isfinite(value)
        for values in (packets_rate, bytes_rate, durations)
        for value in values
    )
    checks = [
        {
            "repeat_id": repeat_id,
            "check": f"{path.name}: at least 20 ICMP rows",
            "status": "PASS" if len(icmp) >= 20 else "FAIL",
            "detail": f"icmp_rows={len(icmp)}",
        },
        {
            "repeat_id": repeat_id,
            "check": f"{path.name}: finite features",
            "status": "PASS" if finite else "FAIL",
            "detail": "all finite" if finite else "non-finite value found",
        },
        {
            "repeat_id": repeat_id,
            "check": f"{path.name}: non-negative deltas",
            "status": "PASS" if invalid_delta == 0 else "FAIL",
            "detail": f"invalid_or_negative={invalid_delta}",
        },
    ]
    summary = {
        "repeat_id": repeat_id,
        "scenario": scenario,
        "filename": path.name,
        "total_rows": len(rows),
        "icmp_rows": len(icmp),
        "packets_rate_min": f"{min(packets_rate):.8f}",
        "packets_rate_median": f"{median(packets_rate):.8f}",
        "packets_rate_max": f"{max(packets_rate):.8f}",
        "bytes_rate_median": f"{median(bytes_rate):.8f}",
        "duration_median": f"{median(durations):.8f}",
        "sha256": digest(path),
    }
    return summary, checks


def run_evaluator(paths: dict[str, Path], output_dir: Path) -> None:
    command = [
        sys.executable,
        str(EVALUATOR),
        "--calibration-normal",
        str(paths["calibration"]),
        "--test-normal",
        str(paths["normal"]),
        "--test-high-rate",
        str(paths["high_rate"]),
        "--output-dir",
        str(output_dir),
        "--min-samples",
        str(MIN_SAMPLES),
        "--eps",
        *[str(value) for value in EPS_VALUES],
    ]
    subprocess.run(command, check=True)


def save_aggregate_metrics_plot(rows: list[dict[str, object]], path: Path) -> None:
    eps = np.asarray([float(row["eps"]) for row in rows])
    fig, axis = plt.subplots(figsize=(8.2, 4.8))
    for field, label, color in (
        ("f1_score_mean", "F1", "#1f77b4"),
        ("recall_mean", "Recall", "#2ca02c"),
        ("specificity_mean", "Specificity", "#ff7f0e"),
        ("balanced_accuracy_mean", "Balanced accuracy", "#9467bd"),
    ):
        axis.plot(eps, [float(row[field]) for row in rows], marker="o", label=label, color=color)
    axis.set_xlabel("eps")
    axis.set_ylabel("Mean across confirmatory repetitions")
    axis.set_ylim(0.0, 1.03)
    axis.grid(alpha=0.25)
    axis.legend(loc="lower right", ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_per_repeat_plot(rows: list[dict[str, object]], path: Path) -> None:
    identifiers = [str(row["repeat_id"]) for row in rows]
    x = np.arange(len(rows))
    width = 0.25
    fig, axis = plt.subplots(figsize=(8.2, 4.8))
    axis.bar(x - width, [float(row["f1_score"]) for row in rows], width, label="F1")
    axis.bar(x, [float(row["balanced_accuracy"]) for row in rows], width, label="Balanced accuracy")
    axis.bar(x + width, [float(row["false_positive_rate"]) for row in rows], width, label="FPR")
    axis.set_xticks(x, identifiers)
    axis.set_xlabel("Confirmatory repetition")
    axis.set_ylabel("Metric value")
    axis.set_ylim(0.0, 1.05)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_confusion_plot(tp: int, fp: int, tn: int, fn: int, path: Path) -> None:
    matrix = np.asarray([[tn, fp], [fn, tp]])
    fig, axis = plt.subplots(figsize=(5.4, 4.8))
    image = axis.imshow(matrix, cmap="Blues")
    for row in range(2):
        for col in range(2):
            axis.text(col, row, str(matrix[row, col]), ha="center", va="center", fontsize=14)
    axis.set_xticks([0, 1], ["Predicted normal", "Predicted anomaly"])
    axis.set_yticks([0, 1], ["Actual normal", "Actual High-Rate"])
    axis.set_title(f"Pooled confirmatory confusion matrix (eps={SELECTED_EPS:.2f})")
    fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_distance_plot(rows: list[dict[str, str]], path: Path) -> None:
    normal_groups: list[list[float]] = []
    high_groups: list[list[float]] = []
    repeat_ids = sorted({int(row["repeat_id"]) for row in rows})
    for repeat_id in repeat_ids:
        normal_groups.append([
            float(row["nearest_core_distance"])
            for row in rows
            if int(row["repeat_id"]) == repeat_id and row["scenario"] == "normal"
        ])
        high_groups.append([
            float(row["nearest_core_distance"])
            for row in rows
            if int(row["repeat_id"]) == repeat_id and row["scenario"] == "high_rate"
        ])
    positions = np.arange(len(repeat_ids)) * 3.0
    fig, axis = plt.subplots(figsize=(9.0, 5.0))
    axis.boxplot(normal_groups, positions=positions - 0.45, widths=0.7, patch_artist=True,
                 boxprops={"facecolor": "#9ecae1"}, medianprops={"color": "#08519c"})
    axis.boxplot(high_groups, positions=positions + 0.45, widths=0.7, patch_artist=True,
                 boxprops={"facecolor": "#fdae6b"}, medianprops={"color": "#a63603"})
    axis.axhline(SELECTED_EPS, color="#d62728", linestyle="--", label=f"eps={SELECTED_EPS:.2f}")
    axis.set_xticks(positions, [str(value) for value in repeat_ids])
    axis.set_xlabel("Confirmatory repetition")
    axis.set_ylabel("Distance to nearest calibration core point")
    axis.set_yscale("log")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.start_repeat < 2 or args.end_repeat < args.start_repeat:
        raise SystemExit("ERROR: confirmatory repetitions must start at 02 or later.")
    if args.output_dir.exists():
        raise SystemExit(f"ERROR: refusing to overwrite results: {args.output_dir}")
    if not EVALUATOR.exists():
        raise SystemExit(f"ERROR: evaluator not found: {EVALUATOR}")

    repeat_ids = list(range(args.start_repeat, args.end_repeat + 1))
    all_paths = {repeat_id: repeat_paths(args.raw_root, repeat_id) for repeat_id in repeat_ids}
    missing = [
        path
        for paths in all_paths.values()
        for path in paths.values()
        if not path.exists()
    ]
    if missing:
        raise SystemExit("ERROR: missing confirmatory inputs:\n" + "\n".join(map(str, missing)))

    args.output_dir.mkdir(parents=True)
    raw_summaries: list[dict[str, object]] = []
    validation_rows: list[dict[str, object]] = []
    per_run_metrics: list[dict[str, object]] = []
    selected_predictions: list[dict[str, str]] = []
    false_positive_rows: list[dict[str, object]] = []

    try:
        for repeat_id in repeat_ids:
            paths = all_paths[repeat_id]
            run_output = args.output_dir / f"run_{repeat_id:02d}"
            for scenario, key in (
                ("calibration_normal", "calibration"),
                ("test_normal", "normal"),
                ("test_high_rate_icmp", "high_rate"),
            ):
                summary, checks = validate_and_summarize_raw(
                    repeat_id, scenario, paths[key]
                )
                raw_summaries.append(summary)
                validation_rows.extend(checks)

            manifest_rows = read_rows(paths["manifest"])
            validation_rows.append({
                "repeat_id": repeat_id,
                "check": "manifest has three completed traffic runs",
                "status": "PASS" if len(manifest_rows) == 3 else "FAIL",
                "detail": f"rows={len(manifest_rows)}",
            })
            log_files = sorted(paths["manifest"].parent.glob("*.collector.log"))
            log_errors = sum(
                "error" in path.read_text(encoding="utf-8", errors="replace").lower()
                for path in log_files
            )
            validation_rows.append({
                "repeat_id": repeat_id,
                "check": "collector logs contain no errors",
                "status": "PASS" if len(log_files) == 3 and log_errors == 0 else "FAIL",
                "detail": f"logs={len(log_files)}, error_logs={log_errors}",
            })

            run_evaluator(paths, run_output)
            for row in read_rows(run_output / "independent_eps_metrics.csv"):
                per_run_metrics.append({"repeat_id": repeat_id, **row})
            normal_raw_rows = read_rows(paths["normal"])
            normal_raw_by_source_row = {
                str(source_row): row
                for source_row, row in enumerate(normal_raw_rows, start=2)
            }
            for row in read_rows(run_output / "independent_predictions.csv"):
                if math.isclose(float(row["eps"]), SELECTED_EPS):
                    selected_predictions.append({"repeat_id": str(repeat_id), **row})
                    if row["scenario"] == "normal" and row["predicted_anomaly"] == "1":
                        source = normal_raw_by_source_row[row["source_row"]]
                        packet_rate = float(source["packets_rate"])
                        rate_pattern = (
                            "low_polling_boundary"
                            if packet_rate < 0.75
                            else "high_polling_boundary"
                            if packet_rate > 1.25
                            else "central_rate"
                        )
                        false_positive_rows.append({
                            "repeat_id": repeat_id,
                            "source_row": row["source_row"],
                            "timestamp": row["timestamp"],
                            "packets_rate": f"{packet_rate:.8f}",
                            "bytes_rate": source["bytes_rate"],
                            "duration_sec": source["duration_sec"],
                            "nearest_core_distance": row["nearest_core_distance"],
                            "rate_pattern": rate_pattern,
                        })

        if any(row["status"] != "PASS" for row in validation_rows):
            raise RuntimeError("one or more confirmatory validation checks failed")

        aggregate_rows: list[dict[str, object]] = []
        for eps in EPS_VALUES:
            group = [row for row in per_run_metrics if math.isclose(float(row["eps"]), eps)]
            totals = {
                field: sum(int(row[field]) for row in group)
                for field in ("true_positive", "false_positive", "true_negative", "false_negative")
            }
            pooled = derived_metrics(
                totals["true_positive"], totals["false_positive"],
                totals["true_negative"], totals["false_negative"],
            )
            aggregate: dict[str, object] = {
                "eps": f"{eps:.2f}",
                "min_samples": MIN_SAMPLES,
                "confirmatory_repetitions": len(group),
                **totals,
            }
            for metric in METRIC_FIELDS:
                stats = summarize([float(row[metric]) for row in group])
                aggregate[f"{metric}_pooled"] = f"{pooled[metric]:.8f}"
                for stat_name, value in stats.items():
                    aggregate[f"{metric}_{stat_name}"] = f"{value:.8f}"
            aggregate_rows.append(aggregate)

        selected_rows = [
            row for row in per_run_metrics
            if math.isclose(float(row["eps"]), SELECTED_EPS)
        ]
        selected_aggregate = next(
            row for row in aggregate_rows
            if math.isclose(float(row["eps"]), SELECTED_EPS)
        )

        write_rows(args.output_dir / "raw_run_summary.csv", list(raw_summaries[0]), raw_summaries)
        write_rows(args.output_dir / "validation_summary.csv", list(validation_rows[0]), validation_rows)
        write_rows(args.output_dir / "per_run_eps_metrics.csv", list(per_run_metrics[0]), per_run_metrics)
        write_rows(args.output_dir / "aggregate_eps_metrics.csv", list(aggregate_rows[0]), aggregate_rows)
        write_rows(args.output_dir / "selected_eps_per_run_metrics.csv", list(selected_rows[0]), selected_rows)
        write_rows(
            args.output_dir / "selected_eps_predictions.csv",
            list(selected_predictions[0]),
            selected_predictions,
        )
        write_rows(
            args.output_dir / "false_positive_diagnostics.csv",
            list(false_positive_rows[0]),
            false_positive_rows,
        )

        save_aggregate_metrics_plot(aggregate_rows, args.output_dir / "aggregate_metrics_by_eps.png")
        save_per_repeat_plot(selected_rows, args.output_dir / "selected_eps_per_repeat.png")
        save_confusion_plot(
            int(selected_aggregate["true_positive"]),
            int(selected_aggregate["false_positive"]),
            int(selected_aggregate["true_negative"]),
            int(selected_aggregate["false_negative"]),
            args.output_dir / "pooled_confusion_matrix.png",
        )
        save_distance_plot(selected_predictions, args.output_dir / "selected_eps_distance_by_repeat.png")

        low_boundary_fp = sum(
            row["rate_pattern"] == "low_polling_boundary"
            for row in false_positive_rows
        )
        high_boundary_fp = sum(
            row["rate_pattern"] == "high_polling_boundary"
            for row in false_positive_rows
        )
        central_rate_fp = sum(
            row["rate_pattern"] == "central_rate"
            for row in false_positive_rows
        )

        report = f"""# گزارش ارزیابی تأییدی تکرارشده DBSCAN

## طراحی ارزیابی

اجرای ۰۱ به‌عنوان اجرای راهنما برای تثبیت پارامترهای `eps={SELECTED_EPS:.2f}` و `min_samples={MIN_SAMPLES}` در نظر گرفته شد. برآورد نهایی عملکرد فقط بر تکرارهای تأییدی {repeat_ids[0]:02d} تا {repeat_ids[-1]:02d} متکی است. در هر تکرار، استانداردساز و DBSCAN فقط با داده عادی کالیبراسیون همان تکرار برازش شدند و برچسب سناریو تنها پس از پیش‌بینی برای محاسبه معیارها استفاده شد.

## نتیجه تجمیعی در eps منتخب

- تعداد تکرارهای تأییدی: {len(selected_rows)}
- TP={selected_aggregate['true_positive']}، FP={selected_aggregate['false_positive']}، TN={selected_aggregate['true_negative']} و FN={selected_aggregate['false_negative']}
- Precision تجمیعی: {100*float(selected_aggregate['precision_pooled']):.2f} درصد
- Recall تجمیعی: {100*float(selected_aggregate['recall_pooled']):.2f} درصد
- F1 تجمیعی: {100*float(selected_aggregate['f1_score_pooled']):.2f} درصد
- FPR تجمیعی: {100*float(selected_aggregate['false_positive_rate_pooled']):.2f} درصد
- Specificity تجمیعی: {100*float(selected_aggregate['specificity_pooled']):.2f} درصد
- Balanced Accuracy تجمیعی: {100*float(selected_aggregate['balanced_accuracy_pooled']):.2f} درصد

## تحلیل مثبت‌های کاذب

از {len(false_positive_rows)} مثبت کاذب، {low_boundary_fp} نمونه نرخ نزدیک ۰٫۵ بسته‌برثانیه و {high_boundary_fp} نمونه نرخ نزدیک ۱٫۵ بسته‌برثانیه داشتند؛ تعداد مثبت کاذب در محدوده مرکزی نرخ عادی برابر {central_rate_fp} بود. این الگو با افتادن شمارش بسته در دو سوی مرزهای پولینگ دوسانیه‌ای سازگار است. این تحلیل پس از آزمون انجام شده و برای تغییر آستانه یا حذف نمونه‌ها استفاده نشده است.

نتایج باید به‌عنوان تفکیک ترافیک ICMP با نرخ بالا از ترافیک عادی در توپولوژی کنترل‌شده تفسیر شوند، نه اثبات تشخیص قطعی DDoS در شبکه واقعی. واحد تکرار آماری، اجرای مستقل است و نمونه‌های زمانی داخل هر اجرا جایگزین تکرار مستقل نیستند.
"""
        (args.output_dir / "repeated_analysis_report_fa.md").write_text(report, encoding="utf-8")

        manifest_rows: list[dict[str, object]] = []
        for path in sorted(args.output_dir.rglob("*")):
            if path.is_file() and path.name != "output_manifest.csv":
                manifest_rows.append({
                    "relative_path": str(path.relative_to(args.output_dir)),
                    "bytes": path.stat().st_size,
                    "sha256": digest(path),
                })
        write_rows(
            args.output_dir / "output_manifest.csv",
            ["relative_path", "bytes", "sha256"],
            manifest_rows,
        )

        print(f"Confirmatory repetitions: {repeat_ids}")
        print(f"Selected eps: {SELECTED_EPS:.2f}")
        print(
            "Pooled confusion: "
            f"TP={selected_aggregate['true_positive']} "
            f"FP={selected_aggregate['false_positive']} "
            f"TN={selected_aggregate['true_negative']} "
            f"FN={selected_aggregate['false_negative']}"
        )
        print(f"Created: {args.output_dir}")
    except BaseException:
        print(f"Partial results retained for diagnosis: {args.output_dir}")
        raise


if __name__ == "__main__":
    main()
