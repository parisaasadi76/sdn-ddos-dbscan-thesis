#!/usr/bin/env python3

"""Reproducible, post-hoc evaluation of the current DBSCAN experiments.

Ground-truth labels are used only after DBSCAN fitting. The script validates
row alignment, reproduces stored clusters, calculates density/anomaly and
cluster-structure metrics, and writes tables, figures, and a Markdown report.
"""

from __future__ import annotations

import csv
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)


BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
RESULTS_DIR = PROCESSED_DIR / "results"

METADATA_PATH = PROCESSED_DIR / "metadata_icmp.csv"
PROCESSED_PATH = PROCESSED_DIR / "processed_icmp.csv"
FEATURES_PATH = PROCESSED_DIR / "X_icmp_scaled.csv"
SCALER_STATS_PATH = PROCESSED_DIR / "scaler_stats.csv"
K_DISTANCE_PATH = PROCESSED_DIR / "k_distance_min5.csv"
DBSCAN_SUMMARY_PATH = PROCESSED_DIR / "dbscan_eps_comparison.csv"

NORMAL_RAW_PATH = RAW_DIR / "normal_01.csv"
HIGH_RATE_RAW_PATH = RAW_DIR / "high_rate_icmp_01.csv"

METRICS_PATH = RESULTS_DIR / "comprehensive_eps_metrics.csv"
CONFUSION_PATH = RESULTS_DIR / "confusion_matrices_noise_only.csv"
CROSSTAB_PATH = RESULTS_DIR / "scenario_cluster_crosstab.csv"
COMPOSITION_PATH = RESULTS_DIR / "cluster_composition.csv"
VALIDATION_PATH = RESULTS_DIR / "validation_summary.csv"
REPORT_PATH = RESULTS_DIR / "dbscan_analysis_report_fa.md"

NUMERIC_FIELDS = (
    "packets",
    "bytes",
    "packets_delta",
    "bytes_delta",
    "packets_rate",
    "bytes_rate",
    "duration_sec",
)

SCENARIO_ORDER = ("normal", "high_rate")
SCENARIO_COLORS = {
    "normal": "#2563EB",
    "high_rate": "#F97316",
}


def read_dict_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.exists():
        raise SystemExit(f"ERROR: required file not found: {path}")

    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        header = reader.fieldnames or []
        rows = list(reader)

    if not header:
        raise SystemExit(f"ERROR: missing header: {path}")

    return header, rows


def read_feature_matrix(path: Path) -> tuple[list[str], np.ndarray]:
    if not path.exists():
        raise SystemExit(f"ERROR: required file not found: {path}")

    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.reader(file)
        header = next(reader, None)
        rows = [row for row in reader if row]

    if not header:
        raise SystemExit(f"ERROR: missing feature header: {path}")

    if any(len(row) != len(header) for row in rows):
        raise SystemExit(f"ERROR: inconsistent feature width: {path}")

    try:
        matrix = np.asarray(rows, dtype=float)
    except ValueError as error:
        raise SystemExit(f"ERROR: non-numeric feature value: {error}")

    if matrix.ndim != 2 or matrix.shape[0] == 0:
        raise SystemExit("ERROR: feature matrix is empty or invalid.")

    if not np.isfinite(matrix).all():
        raise SystemExit("ERROR: feature matrix contains NaN or infinity.")

    return header, matrix


def write_dict_rows(
    path: Path,
    fieldnames: list[str],
    rows: list[dict[str, object]],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def divide(numerator: float, denominator: float) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def csv_number(value: float | None, digits: int = 8) -> str:
    if value is None or not math.isfinite(value):
        return ""
    return f"{value:.{digits}f}"


def display_number(value: float | None, digits: int = 4) -> str:
    if value is None or not math.isfinite(value):
        return "N/A"
    return f"{value:.{digits}f}"


def eps_tag(eps: float) -> str:
    return f"{eps:.2f}".replace(".", "p")


def protocol_name(match_value: str) -> str:
    text = (match_value or "").lower()
    if re.search(r"\bicmp\b", text):
        return "icmp"
    if re.search(r"\barp\b", text):
        return "arp"
    return "other"


def parse_numeric(value: str) -> float | None:
    text = (value or "").strip()
    if text.endswith("s"):
        text = text[:-1]
    try:
        number = float(text)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def add_check(
    checks: list[dict[str, object]],
    name: str,
    passed: bool,
    observed: object,
    expected: object,
    notes: str,
) -> None:
    checks.append(
        {
            "check": name,
            "status": "PASS" if passed else "FAIL",
            "observed": observed,
            "expected": expected,
            "notes": notes,
        }
    )


def internal_cluster_metrics(
    features: np.ndarray,
    labels: np.ndarray,
) -> tuple[float | None, float | None, float | None]:
    non_noise_mask = labels != -1
    selected_features = features[non_noise_mask]
    selected_labels = labels[non_noise_mask]
    unique_labels = np.unique(selected_labels)

    if not (2 <= len(unique_labels) < len(selected_labels)):
        return None, None, None

    try:
        silhouette = float(
            silhouette_score(
                selected_features,
                selected_labels,
                metric="euclidean",
            )
        )
        davies_bouldin = float(
            davies_bouldin_score(selected_features, selected_labels)
        )
        calinski_harabasz = float(
            calinski_harabasz_score(selected_features, selected_labels)
        )
    except ValueError:
        return None, None, None

    return silhouette, davies_bouldin, calinski_harabasz


def validate_raw_to_processed(
    processed_rows: list[dict[str, str]],
) -> tuple[bool, dict[str, Counter[str]], int]:
    raw_specs = (
        (NORMAL_RAW_PATH, "normal", "0"),
        (HIGH_RATE_RAW_PATH, "high_rate", "1"),
    )
    expected: list[tuple[object, ...]] = []
    protocol_counts: dict[str, Counter[str]] = {}

    for path, scenario, label in raw_specs:
        _, raw_rows = read_dict_rows(path)
        counts = Counter(protocol_name(row.get("match", "")) for row in raw_rows)
        protocol_counts[path.name] = counts

        for row in raw_rows:
            protocol = protocol_name(row.get("match", ""))
            if protocol != "icmp":
                continue

            numeric_values = [parse_numeric(row.get(field, "")) for field in NUMERIC_FIELDS]
            if any(value is None for value in numeric_values):
                continue

            expected.append(
                (
                    path.name,
                    scenario,
                    label,
                    row.get("timestamp", ""),
                    row.get("switch", ""),
                    protocol,
                    numeric_values,
                )
            )

    if len(expected) != len(processed_rows):
        return False, protocol_counts, len(expected)

    for expected_row, actual_row in zip(expected, processed_rows):
        expected_meta = expected_row[:6]
        actual_meta = (
            actual_row["source_file"],
            actual_row["scenario"],
            actual_row["label"],
            actual_row["timestamp"],
            actual_row["switch"],
            actual_row["protocol"],
        )

        if expected_meta != actual_meta:
            return False, protocol_counts, len(expected)

        actual_numeric = [float(actual_row[field]) for field in NUMERIC_FIELDS]
        if not np.allclose(
            np.asarray(expected_row[6], dtype=float),
            np.asarray(actual_numeric, dtype=float),
            rtol=0.0,
            atol=1e-12,
        ):
            return False, protocol_counts, len(expected)

    return True, protocol_counts, len(expected)


def plot_k_distance(k_distances: list[float], eps_values: list[float]) -> None:
    fig, axis = plt.subplots(figsize=(9, 5.2))
    ranks = np.arange(1, len(k_distances) + 1)
    axis.plot(ranks, k_distances, color="#1F2937", linewidth=2.0)
    axis.axhspan(
        0.30,
        0.43,
        color="#F59E0B",
        alpha=0.14,
        label="Previously inspected knee range (0.30–0.43)",
    )

    for eps in eps_values:
        axis.axhline(
            eps,
            color="#64748B",
            linewidth=0.8,
            linestyle="--",
            alpha=0.55,
        )

    axis.set_title("5-nearest-neighbor distance curve")
    axis.set_xlabel("Sorted sample rank")
    axis.set_ylabel("Distance to the 5th neighbor")
    axis.legend(loc="upper left", frameon=False)
    axis.grid(True, alpha=0.22)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "k_distance_plot.png", dpi=180)
    plt.close(fig)


def plot_noise(metric_rows: list[dict[str, object]]) -> None:
    eps_values = [float(row["eps"]) for row in metric_rows]
    total_noise = [int(row["noise_points"]) for row in metric_rows]
    normal_noise = [int(row["normal_noise_points"]) for row in metric_rows]
    high_rate_noise = [int(row["high_rate_noise_points"]) for row in metric_rows]

    fig, axis = plt.subplots(figsize=(8.8, 5.1))
    axis.plot(
        eps_values,
        total_noise,
        marker="s",
        markerfacecolor="white",
        linewidth=1.4,
        linestyle="--",
        color="#111827",
        label="All samples",
        zorder=4,
    )
    axis.plot(
        eps_values,
        normal_noise,
        marker="o",
        linewidth=2,
        label="Normal",
        color=SCENARIO_COLORS["normal"],
    )
    axis.plot(
        eps_values,
        high_rate_noise,
        marker="o",
        linewidth=2,
        label="High-rate ICMP",
        color=SCENARIO_COLORS["high_rate"],
    )
    axis.set_title("Noise points across eps values")
    axis.set_xlabel("eps")
    axis.set_ylabel("Number of noise points")
    axis.set_xticks(eps_values)
    axis.set_ylim(bottom=0)
    axis.legend(loc="upper right", frameon=False)
    axis.grid(True, alpha=0.22)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "noise_points_by_eps.png", dpi=180)
    plt.close(fig)


def plot_detection_metrics(metric_rows: list[dict[str, object]]) -> None:
    eps_values = [float(row["eps"]) for row in metric_rows]
    series = (
        ("noise_precision", "Precision", "#0F766E"),
        ("noise_recall", "Recall", "#DC2626"),
        ("noise_f1", "F1-score", "#7C3AED"),
        ("balanced_accuracy", "Balanced accuracy", "#2563EB"),
        ("false_positive_rate", "False-positive rate", "#64748B"),
    )

    fig, axis = plt.subplots(figsize=(9.2, 5.3))
    for key, label, color in series:
        values = [100.0 * float(row[key]) for row in metric_rows]
        axis.plot(
            eps_values,
            values,
            marker="o",
            linewidth=2,
            label=label,
            color=color,
        )

    axis.set_title("Noise-only anomaly metrics across eps values")
    axis.set_xlabel("eps")
    axis.set_ylabel("Percent")
    axis.set_xticks(eps_values)
    axis.set_ylim(-2, 103)
    axis.legend(loc="center right", frameon=False)
    axis.grid(True, alpha=0.22)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "noise_detection_metrics_by_eps.png", dpi=180)
    plt.close(fig)


def plot_scenario_distribution(
    eps_values: list[float],
    labels_by_eps: dict[float, np.ndarray],
    scenarios: list[str],
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 8.0), sharey=True)
    axes_flat = axes.ravel()

    for axis, eps in zip(axes_flat, eps_values):
        labels = labels_by_eps[eps]
        unique_labels = sorted(set(int(value) for value in labels))
        positions = np.arange(len(unique_labels))
        bottom = np.zeros(len(unique_labels), dtype=int)

        for scenario in SCENARIO_ORDER:
            counts = np.asarray(
                [
                    sum(
                        1
                        for index, value in enumerate(labels)
                        if int(value) == cluster_label
                        and scenarios[index] == scenario
                    )
                    for cluster_label in unique_labels
                ],
                dtype=int,
            )
            axis.bar(
                positions,
                counts,
                bottom=bottom,
                label=("Normal" if scenario == "normal" else "High-rate ICMP"),
                color=SCENARIO_COLORS[scenario],
            )
            bottom += counts

        axis.set_title(f"eps={eps:.2f}")
        axis.set_xticks(
            positions,
            ["Noise" if value == -1 else f"Cluster {value}" for value in unique_labels],
            rotation=18,
        )
        axis.set_ylim(0, 65)
        axis.grid(axis="y", alpha=0.22)

    axes_flat[0].set_ylabel("Number of samples")
    axes_flat[3].set_ylabel("Number of samples")
    handles, legend_labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.95),
    )
    fig.suptitle("Scenario composition of DBSCAN groups", y=0.99, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.89))
    fig.savefig(RESULTS_DIR / "scenario_cluster_distribution.png", dpi=180)
    plt.close(fig)


def plot_pca_clusters(
    features: np.ndarray,
    eps_values: list[float],
    labels_by_eps: dict[float, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    pca = PCA(n_components=2)
    projected = pca.fit_transform(features)
    variance = pca.explained_variance_ratio_

    fig, axes = plt.subplots(2, 3, figsize=(13.4, 8.2), sharex=True, sharey=True)
    axes_flat = axes.ravel()
    palette = {0: "#2563EB", 1: "#F97316", 2: "#16A34A", 3: "#7C3AED"}

    for axis, eps in zip(axes_flat, eps_values):
        labels = labels_by_eps[eps]
        for cluster_label in sorted(set(int(value) for value in labels)):
            mask = labels == cluster_label
            if cluster_label == -1:
                axis.scatter(
                    projected[mask, 0],
                    projected[mask, 1],
                    s=34,
                    marker="x",
                    linewidths=1.5,
                    color="#111827",
                    label="Noise",
                )
            else:
                axis.scatter(
                    projected[mask, 0],
                    projected[mask, 1],
                    s=28,
                    alpha=0.78,
                    color=palette.get(cluster_label, "#64748B"),
                    label=f"Cluster {cluster_label}",
                )
        axis.set_title(f"eps={eps:.2f}")
        axis.grid(True, alpha=0.2)

    axes_flat[3].set_xlabel(f"PC1 ({100.0 * variance[0]:.1f}% variance)")
    axes_flat[4].set_xlabel(f"PC1 ({100.0 * variance[0]:.1f}% variance)")
    axes_flat[5].set_xlabel(f"PC1 ({100.0 * variance[0]:.1f}% variance)")
    axes_flat[0].set_ylabel(f"PC2 ({100.0 * variance[1]:.1f}% variance)")
    axes_flat[3].set_ylabel(f"PC2 ({100.0 * variance[1]:.1f}% variance)")
    handles, legend_labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.95),
    )
    fig.suptitle(
        "DBSCAN assignments in a two-dimensional PCA view",
        y=0.995,
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(RESULTS_DIR / "pca_cluster_scatter_grid.png", dpi=180)
    plt.close(fig)
    return projected, variance


def plot_pca_scenarios(
    projected: np.ndarray,
    variance: np.ndarray,
    scenarios: list[str],
) -> None:
    fig, axis = plt.subplots(figsize=(8.4, 5.7))
    for scenario in SCENARIO_ORDER:
        mask = np.asarray([value == scenario for value in scenarios])
        axis.scatter(
            projected[mask, 0],
            projected[mask, 1],
            s=34,
            alpha=0.78,
            color=SCENARIO_COLORS[scenario],
            label=("Normal" if scenario == "normal" else "High-rate ICMP"),
        )
    axis.set_title("Observed scenarios in the same PCA projection")
    axis.set_xlabel(f"PC1 ({100.0 * variance[0]:.1f}% variance)")
    axis.set_ylabel(f"PC2 ({100.0 * variance[1]:.1f}% variance)")
    axis.legend(frameon=False)
    axis.grid(True, alpha=0.22)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "pca_scenario_scatter.png", dpi=180)
    plt.close(fig)


def write_report(
    feature_names: list[str],
    min_samples: int,
    protocol_counts: dict[str, Counter[str]],
    metric_rows: list[dict[str, object]],
    confusion_rows: list[dict[str, object]],
    validation_rows: list[dict[str, object]],
) -> None:
    lines: list[str] = []
    lines.extend(
        [
            "# گزارش راستی‌آزمایی و تحلیل DBSCAN",
            "",
            "تاریخ تحلیل: ۲۰۲۶-۰۹-۱۴",
            "",
            "## ۱. داده و تعداد نمونه‌ها",
            "",
            "این تحلیل روی ترافیک ICMP انجام شده است. فایل خام Normal شامل "
            f"{sum(protocol_counts['normal_01.csv'].values())} ردیف داده است: "
            f"{protocol_counts['normal_01.csv']['icmp']} ICMP و "
            f"{protocol_counts['normal_01.csv']['arp']} ARP. فایل High-Rate شامل "
            f"{sum(protocol_counts['high_rate_icmp_01.csv'].values())} ردیف داده است: "
            f"{protocol_counts['high_rate_icmp_01.csv']['icmp']} ICMP و "
            f"{protocol_counts['high_rate_icmp_01.csv']['arp']} ARP. بنابراین اختلاف "
            "تعداد خام و پردازش‌شده ناشی از حذف ARP طبق تعریف پیش‌پردازش است و "
            "هیچ ردیف ICMP معتبر گم نشده است.",
            "",
            "تعداد نهایی نمونه‌ها ۹۶ است: ۶۰ Normal و ۳۶ High-Rate ICMP.",
            "",
            "## ۲. ویژگی‌ها و مقیاس‌بندی",
            "",
            "ویژگی‌های مدل: `" + "`, `".join(feature_names) + "`.",
            "",
            "هر ویژگی با میانگین و انحراف معیار جمعیت روی کل ۹۶ نمونه "
            "استاندارد شده است. برچسب واقعی در محاسبهٔ scaling یا fit کردن DBSCAN "
            "استفاده نشده است. با این حال، fit شدن scaler روی کل داده برای ادعای "
            "تعمیم‌پذیری مناسب نیست و در آزمایش مستقل باید scaler فقط روی مجموعهٔ "
            "کالیبراسیون fit شود.",
            "",
            "## ۳. تنظیمات DBSCAN",
            "",
            f"مقدار `min_samples` برابر {min_samples} است. مقادیر بررسی‌شدهٔ `eps`: "
            + ", ".join(f"{float(row['eps']):.2f}" for row in metric_rows)
            + ".",
            "",
            "## ۴. نتایج جامع",
            "",
            "| eps | خوشه | core | border | noise | Silhouette | DBI | CH | ARI کل | خلوص غیرنویز |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )

    for row in metric_rows:
        lines.append(
            "| "
            f"{float(row['eps']):.2f} | {row['clusters']} | {row['core_points']} | "
            f"{row['border_points']} | {row['noise_points']} | "
            f"{display_number(float(row['silhouette_non_noise']) if row['silhouette_non_noise'] != '' else None)} | "
            f"{display_number(float(row['davies_bouldin_non_noise']) if row['davies_bouldin_non_noise'] != '' else None)} | "
            f"{display_number(float(row['calinski_harabasz_non_noise']) if row['calinski_harabasz_non_noise'] != '' else None, 2)} | "
            f"{display_number(float(row['adjusted_rand_all']))} | "
            f"{100.0 * float(row['cluster_purity_non_noise']):.2f}% |"
        )

    lines.extend(
        [
            "",
            "Silhouette، Davies–Bouldin و Calinski–Harabasz فقط روی نقاط غیرنویز "
            "محاسبه شده‌اند. این معیارها جدایی خوشه‌های پذیرفته‌شده را اندازه می‌گیرند، "
            "نه توان تشخیص حمله را.",
            "",
            "## ۵. ماتریس اغتشاش با قاعدهٔ نویز",
            "",
            "قاعدهٔ ارزیابی در این بخش: `label=-1` ناهنجاری و هر برچسب دیگر غیرناهنجاری.",
            "",
            "| eps | TP | FP | TN | FN | Precision | Recall | F1 | FPR | Specificity | Balanced accuracy |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )

    for row in confusion_rows:
        lines.append(
            "| "
            f"{float(row['eps']):.2f} | {row['true_positive']} | {row['false_positive']} | "
            f"{row['true_negative']} | {row['false_negative']} | "
            f"{100.0 * float(row['precision']):.2f}% | "
            f"{100.0 * float(row['recall']):.2f}% | "
            f"{100.0 * float(row['f1_score']):.2f}% | "
            f"{100.0 * float(row['false_positive_rate']):.2f}% | "
            f"{100.0 * float(row['specificity']):.2f}% | "
            f"{100.0 * float(row['balanced_accuracy']):.2f}% |"
        )

    lines.extend(
        [
            "",
            "## ۶. تشخیص نویز و تفکیک خوشه‌ای",
            "",
            "در ارزیابی نویز، فقط نقاط `-1` ناهنجار تلقی می‌شوند. طبق این تعریف، "
            "کاهش eps نمونه‌های بیشتری را نویز می‌کند و recall را افزایش می‌دهد. "
            "در ارزیابی ساختاری، قرار گرفتن High-Rate ICMP در خوشه‌ای جدا نیز یک "
            "الگوی مهم است، حتی اگر آن نقاط نویز نباشند. خلوص، ARI و NMI در این "
            "گزارش صرفاً ارزیابی پسینی هستند و در fit مدل استفاده نشده‌اند.",
            "",
            "شماره‌های خوشه معنای ثابت ندارند. نسبت دادن مفهوم Normal یا High-Rate "
            "به یک شمارهٔ خوشه با استفاده از برچسب‌های همین مجموعه، ارزیابی خوش‌بینانه "
            "و نوعی data leakage است و نباید به‌عنوان عملکرد روی دادهٔ جدید گزارش شود.",
            "",
            "## ۷. وضعیت انتخاب eps",
            "",
            "دادهٔ فعلی برای اعلام eps نهایی کافی نیست. `eps=0.28` نامزد قاعدهٔ "
            "noise-only است، زیرا بیشترین recall را بدون false positive در همین داده "
            "دارد. `eps=0.31` و `eps=0.35` نامزدهای تحلیل ساختاری هستند، زیرا بیشتر "
            "نمونه‌های High-Rate را در خوشه‌ای جدا نگه می‌دارند. این‌ها نامزدهای موقت "
            "هستند، نه انتخاب نهایی.",
            "",
            "## ۸. محدودیت‌ها",
            "",
            "- مجموعه کوچک و مربوط به تعداد محدودی اجرای شبکه است.",
            "- ردیف‌های متوالی جریان ممکن است مستقل آماری نباشند.",
            "- scaler روی کل مجموعه fit شده است.",
            "- DBSCAN تابع predict استاندارد برای نمونهٔ جدید ندارد.",
            "- High-Rate ICMP الزاماً حملهٔ DDoS قطعی نیست.",
            "- عملکرد صفر false positive در همین مجموعه تضمین عملکرد روی اجرای جدید نیست.",
            "",
            "## ۹. آزمایش مستقل موردنیاز",
            "",
            "۱. یک اجرای Normal مستقل برای کالیبراسیون جمع‌آوری شود. ۲. scaler فقط "
            "روی کالیبراسیون fit شود. ۳. DBSCAN فقط روی کالیبراسیون fit شود. "
            "۴. یک اجرای Normal جدید و یک اجرای High-Rate ICMP جدید برای test نگه "
            "داشته شوند. ۵. نمونهٔ test زمانی غیرناهنجار تلقی شود که در فاصلهٔ eps "
            "از حداقل یک core point کالیبراسیون قرار گیرد؛ در غیر این صورت ناهنجار "
            "است. ۶. انتخاب eps بر اساس recall، FPR، balanced accuracy و پایداری در "
            "چند اجرای مستقل انجام شود.",
            "",
            "تقسیم باید بر اساس اجرای شبکه باشد، نه تقسیم تصادفی ردیف‌های متوالی.",
            "",
            "## ۱۰. کنترل‌های اعتبارسنجی",
            "",
            f"{sum(row['status'] == 'PASS' for row in validation_rows)} کنترل موفق و "
            f"{sum(row['status'] == 'FAIL' for row in validation_rows)} کنترل ناموفق ثبت شد.",
            "",
            "جزئیات در `validation_summary.csv` قرار دارد.",
            "",
            "## ۱۱. بازتولید تحلیل",
            "",
            "```bash",
            "python src/analyze_dbscan_full_v2.py",
            "```",
        ]
    )

    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    metadata_header, metadata_rows = read_dict_rows(METADATA_PATH)
    processed_header, processed_rows = read_dict_rows(PROCESSED_PATH)
    feature_names, features = read_feature_matrix(FEATURES_PATH)
    _, scaler_rows = read_dict_rows(SCALER_STATS_PATH)
    _, summary_rows = read_dict_rows(DBSCAN_SUMMARY_PATH)
    k_header, k_rows = read_dict_rows(K_DISTANCE_PATH)

    checks: list[dict[str, object]] = []
    expected_ids = list(range(1, len(metadata_rows) + 1))
    observed_ids = [int(row["row_id"]) for row in metadata_rows]
    add_check(
        checks,
        "metadata_row_id_sequential",
        observed_ids == expected_ids,
        f"{observed_ids[0]}..{observed_ids[-1]}" if observed_ids else "empty",
        f"1..{len(metadata_rows)}",
        "row_id must be contiguous and ordered.",
    )

    add_check(
        checks,
        "feature_row_count",
        len(features) == len(metadata_rows),
        len(features),
        len(metadata_rows),
        "X and metadata must contain the same observations.",
    )
    add_check(
        checks,
        "processed_row_count",
        len(processed_rows) == len(metadata_rows),
        len(processed_rows),
        len(metadata_rows),
        "Processed rows and metadata must remain aligned.",
    )

    metadata_fields = (
        "row_id",
        "timestamp",
        "source_file",
        "scenario",
        "label",
        "protocol",
    )
    processed_metadata_match = len(processed_rows) == len(metadata_rows) and all(
        all(processed[field] == metadata[field] for field in metadata_fields)
        for processed, metadata in zip(processed_rows, metadata_rows)
    )
    add_check(
        checks,
        "processed_metadata_order",
        processed_metadata_match,
        processed_metadata_match,
        True,
        "Metadata columns must match the processed source row by row.",
    )

    scaler_stats = {
        row["feature"]: (
            float(row["mean"]),
            float(row["standard_deviation"]),
        )
        for row in scaler_rows
    }
    reconstructed = np.asarray(
        [
            [
                (float(row[name]) - scaler_stats[name][0])
                / scaler_stats[name][1]
                for name in feature_names
            ]
            for row in processed_rows
        ],
        dtype=float,
    )
    reconstruction_error = float(np.max(np.abs(reconstructed - features)))
    add_check(
        checks,
        "scaled_feature_reconstruction",
        reconstruction_error < 1e-9,
        f"{reconstruction_error:.3e}",
        "<1e-9",
        "Confirms values and row order using processed rows and scaler statistics.",
    )

    raw_match, protocol_counts, selected_raw_rows = validate_raw_to_processed(
        processed_rows
    )
    add_check(
        checks,
        "raw_icmp_to_processed_order",
        raw_match,
        selected_raw_rows,
        len(processed_rows),
        "Replays ICMP filtering and verifies metadata, numeric values, and order.",
    )

    unique_min_samples = {int(row["min_samples"]) for row in summary_rows}
    if len(unique_min_samples) != 1:
        raise SystemExit(
            "ERROR: dbscan summary contains inconsistent min_samples values."
        )
    min_samples = unique_min_samples.pop()
    eps_values = [float(row["eps"]) for row in summary_rows]
    summary_by_eps = {float(row["eps"]): row for row in summary_rows}

    true_labels = np.asarray([int(row["label"]) for row in metadata_rows])
    scenarios = [row["scenario"] for row in metadata_rows]

    labels_by_eps: dict[float, np.ndarray] = {}
    metric_rows: list[dict[str, object]] = []
    confusion_rows: list[dict[str, object]] = []
    crosstab_rows: list[dict[str, object]] = []
    composition_rows: list[dict[str, object]] = []

    for eps in eps_values:
        label_path = PROCESSED_DIR / f"dbscan_labels_eps_{eps_tag(eps)}.csv"
        _, saved_label_rows = read_dict_rows(label_path)
        sample_indices = [int(row["sample_index"]) for row in saved_label_rows]
        saved_labels = np.asarray(
            [int(row["cluster_label"]) for row in saved_label_rows],
            dtype=int,
        )
        indices_valid = sample_indices == expected_ids
        add_check(
            checks,
            f"label_indices_eps_{eps:.2f}",
            indices_valid,
            f"{sample_indices[0]}..{sample_indices[-1]}" if sample_indices else "empty",
            f"1..{len(metadata_rows)}",
            "sample_index must match metadata row_id.",
        )

        model = DBSCAN(
            eps=eps,
            min_samples=min_samples,
            metric="euclidean",
            n_jobs=-1,
        ).fit(features)
        labels = np.asarray(model.labels_, dtype=int)
        labels_by_eps[eps] = labels
        labels_match = np.array_equal(labels, saved_labels)
        add_check(
            checks,
            f"stored_labels_reproduced_eps_{eps:.2f}",
            labels_match,
            labels_match,
            True,
            "Fresh DBSCAN fit must reproduce every stored assignment.",
        )

        core_mask = np.zeros(len(labels), dtype=bool)
        core_mask[model.core_sample_indices_] = True
        noise_mask = labels == -1
        non_noise_mask = ~noise_mask
        border_mask = non_noise_mask & ~core_mask
        cluster_labels = sorted(set(int(value) for value in labels if value != -1))
        cluster_sizes = Counter(int(value) for value in labels if value != -1)

        predicted_anomaly = noise_mask
        actual_anomaly = true_labels == 1
        tp = int(np.sum(actual_anomaly & predicted_anomaly))
        fp = int(np.sum(~actual_anomaly & predicted_anomaly))
        tn = int(np.sum(~actual_anomaly & ~predicted_anomaly))
        fn = int(np.sum(actual_anomaly & ~predicted_anomaly))

        precision = divide(tp, tp + fp)
        recall = divide(tp, tp + fn)
        specificity = divide(tn, tn + fp)
        false_positive_rate = divide(fp, fp + tn)
        accuracy = divide(tp + tn, len(labels))
        f1_score = (
            divide(2.0 * precision * recall, precision + recall)
            if precision is not None and recall is not None
            else None
        )
        balanced_accuracy = (
            (recall + specificity) / 2.0
            if recall is not None and specificity is not None
            else None
        )

        silhouette, davies_bouldin, calinski_harabasz = internal_cluster_metrics(
            features, labels
        )

        ari_all = float(adjusted_rand_score(true_labels, labels))
        nmi_all = float(normalized_mutual_info_score(true_labels, labels))
        ari_non_noise = float(
            adjusted_rand_score(true_labels[non_noise_mask], labels[non_noise_mask])
        )
        nmi_non_noise = float(
            normalized_mutual_info_score(
                true_labels[non_noise_mask], labels[non_noise_mask]
            )
        )

        majority_total = 0
        dedicated_high_rate = 0
        for cluster_label in cluster_labels:
            cluster_mask = labels == cluster_label
            normal_count = int(np.sum(cluster_mask & (true_labels == 0)))
            high_rate_count = int(np.sum(cluster_mask & (true_labels == 1)))
            majority_total += max(normal_count, high_rate_count)
            if normal_count == 0:
                dedicated_high_rate += high_rate_count

        cluster_purity = divide(majority_total, int(np.sum(non_noise_mask)))
        dedicated_high_rate_recall = divide(
            dedicated_high_rate,
            int(np.sum(true_labels == 1)),
        )

        normal_mask = true_labels == 0
        high_rate_mask = true_labels == 1
        normal_noise_points = int(np.sum(normal_mask & noise_mask))
        high_rate_noise_points = int(np.sum(high_rate_mask & noise_mask))

        metric_row: dict[str, object] = {
            "eps": f"{eps:.2f}",
            "min_samples": min_samples,
            "samples": len(labels),
            "features": len(feature_names),
            "clusters": len(cluster_labels),
            "core_points": int(np.sum(core_mask)),
            "border_points": int(np.sum(border_mask)),
            "noise_points": int(np.sum(noise_mask)),
            "non_noise_points": int(np.sum(non_noise_mask)),
            "noise_percent_total": csv_number(float(np.mean(noise_mask))),
            "normal_noise_points": normal_noise_points,
            "normal_noise_percent": csv_number(
                divide(normal_noise_points, int(np.sum(normal_mask)))
            ),
            "high_rate_noise_points": high_rate_noise_points,
            "high_rate_noise_percent": csv_number(
                divide(high_rate_noise_points, int(np.sum(high_rate_mask)))
            ),
            "largest_cluster": max(cluster_sizes.values(), default=0),
            "smallest_cluster": min(cluster_sizes.values(), default=0),
            "silhouette_non_noise": csv_number(silhouette),
            "davies_bouldin_non_noise": csv_number(davies_bouldin),
            "calinski_harabasz_non_noise": csv_number(calinski_harabasz),
            "adjusted_rand_all": csv_number(ari_all),
            "normalized_mutual_info_all": csv_number(nmi_all),
            "adjusted_rand_non_noise": csv_number(ari_non_noise),
            "normalized_mutual_info_non_noise": csv_number(nmi_non_noise),
            "cluster_purity_non_noise": csv_number(cluster_purity),
            "dedicated_high_rate_cluster_recall": csv_number(
                dedicated_high_rate_recall
            ),
            "noise_precision": csv_number(precision),
            "noise_recall": csv_number(recall),
            "noise_f1": csv_number(f1_score),
            "false_positive_rate": csv_number(false_positive_rate),
            "specificity": csv_number(specificity),
            "balanced_accuracy": csv_number(balanced_accuracy),
            "accuracy": csv_number(accuracy),
        }
        metric_rows.append(metric_row)

        confusion_rows.append(
            {
                "eps": f"{eps:.2f}",
                "prediction_rule": "cluster_label == -1",
                "true_positive": tp,
                "false_positive": fp,
                "true_negative": tn,
                "false_negative": fn,
                "precision": csv_number(precision),
                "recall": csv_number(recall),
                "f1_score": csv_number(f1_score),
                "false_positive_rate": csv_number(false_positive_rate),
                "specificity": csv_number(specificity),
                "balanced_accuracy": csv_number(balanced_accuracy),
                "accuracy": csv_number(accuracy),
            }
        )

        for scenario in SCENARIO_ORDER:
            scenario_mask = np.asarray([value == scenario for value in scenarios])
            for cluster_label in sorted(set(int(value) for value in labels)):
                crosstab_rows.append(
                    {
                        "eps": f"{eps:.2f}",
                        "scenario": scenario,
                        "cluster_label": cluster_label,
                        "group_type": "noise" if cluster_label == -1 else "cluster",
                        "count": int(
                            np.sum(scenario_mask & (labels == cluster_label))
                        ),
                    }
                )

        for cluster_label in sorted(set(int(value) for value in labels)):
            cluster_mask = labels == cluster_label
            normal_count = int(np.sum(cluster_mask & (true_labels == 0)))
            high_rate_count = int(np.sum(cluster_mask & (true_labels == 1)))
            total = normal_count + high_rate_count
            majority = (
                "high_rate"
                if high_rate_count > normal_count
                else "normal"
                if normal_count > high_rate_count
                else "tie"
            )
            composition_rows.append(
                {
                    "eps": f"{eps:.2f}",
                    "cluster_label": cluster_label,
                    "group_type": "noise" if cluster_label == -1 else "cluster",
                    "total": total,
                    "normal_count": normal_count,
                    "high_rate_count": high_rate_count,
                    "normal_fraction": csv_number(divide(normal_count, total)),
                    "high_rate_fraction": csv_number(
                        divide(high_rate_count, total)
                    ),
                    "posthoc_majority_scenario": majority,
                    "posthoc_warning": (
                        "Uses evaluation labels; do not use as a trained mapping."
                    ),
                }
            )

        summary = summary_by_eps[eps]
        summary_matches = (
            int(summary["rows"]) == len(labels)
            and int(summary["clusters"]) == len(cluster_labels)
            and int(summary["core_points"]) == int(np.sum(core_mask))
            and int(summary["border_points"]) == int(np.sum(border_mask))
            and int(summary["noise_points"]) == int(np.sum(noise_mask))
        )
        add_check(
            checks,
            f"legacy_summary_reproduced_eps_{eps:.2f}",
            summary_matches,
            summary_matches,
            True,
            "Fresh calculation must match dbscan_eps_comparison.csv.",
        )

    failed_checks = [row for row in checks if row["status"] == "FAIL"]
    if failed_checks:
        write_dict_rows(
            VALIDATION_PATH,
            ["check", "status", "observed", "expected", "notes"],
            checks,
        )
        failed_names = ", ".join(str(row["check"]) for row in failed_checks)
        raise SystemExit(f"ERROR: validation failed: {failed_names}")

    metric_fields = list(metric_rows[0])
    confusion_fields = list(confusion_rows[0])
    crosstab_fields = list(crosstab_rows[0])
    composition_fields = list(composition_rows[0])
    validation_fields = ["check", "status", "observed", "expected", "notes"]

    write_dict_rows(METRICS_PATH, metric_fields, metric_rows)
    write_dict_rows(CONFUSION_PATH, confusion_fields, confusion_rows)
    write_dict_rows(CROSSTAB_PATH, crosstab_fields, crosstab_rows)
    write_dict_rows(COMPOSITION_PATH, composition_fields, composition_rows)
    write_dict_rows(VALIDATION_PATH, validation_fields, checks)

    distance_column = f"distance_to_neighbor_{min_samples}"
    if distance_column not in k_header:
        raise SystemExit(
            f"ERROR: expected {distance_column} in {K_DISTANCE_PATH}"
        )
    k_distances = [float(row[distance_column]) for row in k_rows]
    add_check(
        checks,
        "k_distance_row_count",
        len(k_distances) == len(features),
        len(k_distances),
        len(features),
        "k-distance must contain one value per observation.",
    )
    add_check(
        checks,
        "k_distance_sorted",
        k_distances == sorted(k_distances),
        k_distances == sorted(k_distances),
        True,
        "The plotted k-distance sequence must be non-decreasing.",
    )
    write_dict_rows(VALIDATION_PATH, validation_fields, checks)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 10,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )
    plot_k_distance(k_distances, eps_values)
    plot_noise(metric_rows)
    plot_detection_metrics(metric_rows)
    plot_scenario_distribution(eps_values, labels_by_eps, scenarios)
    projected, variance = plot_pca_clusters(features, eps_values, labels_by_eps)
    plot_pca_scenarios(projected, variance, scenarios)

    write_report(
        feature_names,
        min_samples,
        protocol_counts,
        metric_rows,
        confusion_rows,
        checks,
    )

    print(f"Validated observations: {len(metadata_rows)}")
    print(f"Features: {', '.join(feature_names)}")
    print(f"min_samples: {min_samples}")
    print("eps values: " + ", ".join(f"{value:.2f}" for value in eps_values))
    print(f"Validation checks: {len(checks)} passed, 0 failed")
    print(f"Results directory: {RESULTS_DIR}")
    for path in (
        METRICS_PATH,
        CONFUSION_PATH,
        CROSSTAB_PATH,
        COMPOSITION_PATH,
        VALIDATION_PATH,
        REPORT_PATH,
    ):
        print(f"Created: {path}")


if __name__ == "__main__":
    main()
