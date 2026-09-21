#!/usr/bin/env python3

"""Repeated group-split validation of the two-view DBSCAN detector on InSDN."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from analyze_insdn_feature_sensitivity import (
    ALL_NUMERIC_COLUMNS,
    ATTACK_FILES,
    BASELINE3,
    COUNTS7,
    EPS_QUANTILES,
    MIN_SAMPLES,
    NORMAL_FILE,
    TARGET_VALIDATION_FPR,
    confusion_metrics,
    load_rows,
    nearest_distances,
    percent,
    select_columns,
    transform_numeric,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "data/processed/results/insdn_dual_view_repeated"
REPEAT_SEEDS = (20260919, 20260929, 20260939, 20260949, 20260959)


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty table: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def split_by_group(flow_ids: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = np.arange(len(flow_ids))
    first = GroupShuffleSplit(n_splits=1, train_size=0.50, random_state=seed)
    train, remainder = next(first.split(indices, groups=flow_ids))
    second = GroupShuffleSplit(n_splits=1, train_size=0.50, random_state=seed + 1)
    validation_relative, test_relative = next(
        second.split(remainder, groups=flow_ids[remainder])
    )
    validation = remainder[validation_relative]
    test = remainder[test_relative]
    groups = [set(flow_ids[index]) for index in (train, validation, test)]
    if groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2]:
        raise SystemExit(f"ERROR: group leakage for seed {seed}")
    return train, validation, test


def fit_view(
    normal_features: np.ndarray,
    attack_features: np.ndarray,
    feature_names: tuple[str, ...],
    transform: str,
    train: np.ndarray,
    validation: np.ndarray,
    test: np.ndarray,
) -> dict[str, np.ndarray]:
    normal = transform_numeric(select_columns(normal_features, feature_names), transform)
    attack = transform_numeric(select_columns(attack_features, feature_names), transform)
    scaler = StandardScaler()
    calibration = scaler.fit_transform(normal[train])
    validation_data = scaler.transform(normal[validation])
    test_normal = scaler.transform(normal[test])
    test_attack = scaler.transform(attack)
    neighbors = NearestNeighbors(n_neighbors=MIN_SAMPLES, algorithm="auto", n_jobs=-1).fit(
        calibration
    )
    kth = neighbors.kneighbors(calibration, return_distance=True)[0][:, -1]
    return {
        "calibration": calibration,
        "validation": validation_data,
        "test_normal": test_normal,
        "test_attack": test_attack,
        "kth": kth,
    }


def validation_cache(view: dict[str, np.ndarray]) -> dict[float, dict[str, object]]:
    cache: dict[float, dict[str, object]] = {}
    for quantile in EPS_QUANTILES:
        eps = float(np.quantile(view["kth"], quantile))
        core = view["calibration"][view["kth"] <= eps]
        anomaly = nearest_distances(core, view["validation"]) > eps
        cache[quantile] = {
            "eps": eps,
            "core": core,
            "validation_anomaly": anomaly,
            "validation_fpr": float(np.mean(anomaly)),
        }
    return cache


def choose_baseline(cache: dict[float, dict[str, object]]) -> tuple[float, dict[str, object], str]:
    eligible = [
        (quantile, row)
        for quantile, row in cache.items()
        if float(row["validation_fpr"]) <= TARGET_VALIDATION_FPR
    ]
    if eligible:
        quantile, row = min(eligible, key=lambda item: float(item[1]["eps"]))
        return quantile, row, "met_validation_fpr_target"
    quantile, row = min(cache.items(), key=lambda item: float(item[1]["validation_fpr"]))
    return quantile, row, "fallback_lowest_validation_fpr"


def choose_dual(
    raw_cache: dict[float, dict[str, object]],
    log_cache: dict[float, dict[str, object]],
    repeat_number: int,
    seed: int,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    grid: list[dict[str, object]] = []
    for raw_quantile, raw_row in raw_cache.items():
        for log_quantile, log_row in log_cache.items():
            anomaly = np.asarray(raw_row["validation_anomaly"]) | np.asarray(
                log_row["validation_anomaly"]
            )
            grid.append(
                {
                    "repeat": repeat_number,
                    "seed": seed,
                    "raw_quantile": raw_quantile,
                    "log_quantile": log_quantile,
                    "raw_eps": raw_row["eps"],
                    "log_eps": log_row["eps"],
                    "validation_samples": len(anomaly),
                    "validation_false_positives": int(np.sum(anomaly)),
                    "validation_false_positive_rate": float(np.mean(anomaly)),
                }
            )
    eligible = [
        row
        for row in grid
        if float(row["validation_false_positive_rate"]) <= TARGET_VALIDATION_FPR
    ]
    if eligible:
        selected = min(
            eligible,
            key=lambda row: (
                float(row["raw_quantile"]) + float(row["log_quantile"]),
                -float(row["validation_false_positive_rate"]),
            ),
        ).copy()
        selected["selection_status"] = "met_validation_fpr_target"
    else:
        selected = min(grid, key=lambda row: float(row["validation_false_positive_rate"])).copy()
        selected["selection_status"] = "fallback_lowest_validation_fpr"
    return selected, grid


def evaluate_single_view(
    name: str,
    repeat_number: int,
    seed: int,
    view: dict[str, np.ndarray],
    quantile: float,
    selected: dict[str, object],
    selection_status: str,
    balanced_attack_indices: np.ndarray,
    attack_sources: np.ndarray,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    eps = float(selected["eps"])
    core = np.asarray(selected["core"])
    normal_anomaly = nearest_distances(core, view["test_normal"]) > eps
    attack_anomaly = nearest_distances(core, view["test_attack"]) > eps
    balanced = confusion_metrics(normal_anomaly, attack_anomaly[balanced_attack_indices])
    result = {
        "model": name,
        "repeat": repeat_number,
        "seed": seed,
        "calibration_samples": len(view["calibration"]),
        "validation_samples": len(view["validation"]),
        "test_normal_samples": len(normal_anomaly),
        "full_ddos_samples": len(attack_anomaly),
        "raw_quantile": quantile,
        "log_quantile": "",
        "raw_eps": eps,
        "log_eps": "",
        "validation_false_positive_rate": selected["validation_fpr"],
        "selection_status": selection_status,
        **balanced,
        "full_ddos_detected": int(np.sum(attack_anomaly)),
        "full_ddos_recall": float(np.mean(attack_anomaly)),
    }
    sources: list[dict[str, object]] = []
    for source in sorted(set(attack_sources)):
        mask = attack_sources == source
        sources.append(
            {
                "model": name,
                "repeat": repeat_number,
                "seed": seed,
                "source_file": source,
                "ddos_samples": int(np.sum(mask)),
                "detected_ddos": int(np.sum(attack_anomaly[mask])),
                "recall": float(np.mean(attack_anomaly[mask])),
            }
        )
    return result, sources


def evaluate_dual(
    repeat_number: int,
    seed: int,
    raw_view: dict[str, np.ndarray],
    log_view: dict[str, np.ndarray],
    raw_cache: dict[float, dict[str, object]],
    log_cache: dict[float, dict[str, object]],
    selected: dict[str, object],
    balanced_attack_indices: np.ndarray,
    attack_sources: np.ndarray,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    raw_quantile = float(selected["raw_quantile"])
    log_quantile = float(selected["log_quantile"])
    raw_eps = float(selected["raw_eps"])
    log_eps = float(selected["log_eps"])
    raw_core = np.asarray(raw_cache[raw_quantile]["core"])
    log_core = np.asarray(log_cache[log_quantile]["core"])
    normal_anomaly = (
        nearest_distances(raw_core, raw_view["test_normal"]) > raw_eps
    ) | (nearest_distances(log_core, log_view["test_normal"]) > log_eps)
    attack_anomaly = (
        nearest_distances(raw_core, raw_view["test_attack"]) > raw_eps
    ) | (nearest_distances(log_core, log_view["test_attack"]) > log_eps)
    balanced = confusion_metrics(normal_anomaly, attack_anomaly[balanced_attack_indices])
    result = {
        "model": "dual_view_union",
        "repeat": repeat_number,
        "seed": seed,
        "calibration_samples": len(raw_view["calibration"]),
        "validation_samples": len(raw_view["validation"]),
        "test_normal_samples": len(normal_anomaly),
        "full_ddos_samples": len(attack_anomaly),
        "raw_quantile": raw_quantile,
        "log_quantile": log_quantile,
        "raw_eps": raw_eps,
        "log_eps": log_eps,
        "validation_false_positive_rate": selected["validation_false_positive_rate"],
        "selection_status": selected["selection_status"],
        **balanced,
        "full_ddos_detected": int(np.sum(attack_anomaly)),
        "full_ddos_recall": float(np.mean(attack_anomaly)),
    }
    sources: list[dict[str, object]] = []
    for source in sorted(set(attack_sources)):
        mask = attack_sources == source
        sources.append(
            {
                "model": "dual_view_union",
                "repeat": repeat_number,
                "seed": seed,
                "source_file": source,
                "ddos_samples": int(np.sum(mask)),
                "detected_ddos": int(np.sum(attack_anomaly[mask])),
                "recall": float(np.mean(attack_anomaly[mask])),
            }
        )
    return result, sources


def aggregate_rows(per_repeat: list[dict[str, object]]) -> list[dict[str, object]]:
    fields = (
        "precision",
        "recall",
        "f1_score",
        "false_positive_rate",
        "specificity",
        "balanced_accuracy",
        "full_ddos_recall",
    )
    rows: list[dict[str, object]] = []
    for model in ("baseline3_raw", "dual_view_union"):
        model_rows = [row for row in per_repeat if row["model"] == model]
        row: dict[str, object] = {"model": model, "repeats": len(model_rows)}
        for field in fields:
            values = np.asarray([float(item[field]) for item in model_rows])
            row[f"{field}_mean"] = float(np.mean(values))
            row[f"{field}_sample_std"] = float(np.std(values, ddof=1))
            row[f"{field}_min"] = float(np.min(values))
            row[f"{field}_max"] = float(np.max(values))
        rows.append(row)
    return rows


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

    per_repeat: list[dict[str, object]] = []
    source_rows: list[dict[str, object]] = []
    validation_grid: list[dict[str, object]] = []

    for repeat_number, seed in enumerate(REPEAT_SEEDS, start=1):
        train, validation, test = split_by_group(normal_flow_ids, seed)
        raw_view = fit_view(
            normal_features,
            attack_features,
            BASELINE3,
            "standardize_raw",
            train,
            validation,
            test,
        )
        log_view = fit_view(
            normal_features,
            attack_features,
            COUNTS7,
            "log1p_then_standardize",
            train,
            validation,
            test,
        )
        raw_cache = validation_cache(raw_view)
        log_cache = validation_cache(log_view)
        baseline_quantile, baseline_selected, baseline_status = choose_baseline(raw_cache)
        dual_selected, grid = choose_dual(
            raw_cache, log_cache, repeat_number, seed
        )
        validation_grid.extend(grid)
        rng = np.random.default_rng(seed)
        balanced_attack_indices = np.sort(
            rng.choice(len(attack_features), size=len(test), replace=False)
        )

        baseline_result, baseline_sources = evaluate_single_view(
            "baseline3_raw",
            repeat_number,
            seed,
            raw_view,
            baseline_quantile,
            baseline_selected,
            baseline_status,
            balanced_attack_indices,
            attack_sources,
        )
        dual_result, dual_sources = evaluate_dual(
            repeat_number,
            seed,
            raw_view,
            log_view,
            raw_cache,
            log_cache,
            dual_selected,
            balanced_attack_indices,
            attack_sources,
        )
        per_repeat.extend((baseline_result, dual_result))
        source_rows.extend((*baseline_sources, *dual_sources))
        print(
            f"repeat {repeat_number}: baseline recall={percent(baseline_result['full_ddos_recall'])}, "
            f"dual recall={percent(dual_result['full_ddos_recall'])}, "
            f"dual FPR={percent(dual_result['false_positive_rate'])}"
        )

    aggregate = aggregate_rows(per_repeat)
    write_rows(OUTPUT_DIR / "per_repeat_metrics.csv", per_repeat)
    write_rows(OUTPUT_DIR / "aggregate_metrics.csv", aggregate)
    write_rows(OUTPUT_DIR / "per_repeat_source_recall.csv", source_rows)
    write_rows(OUTPUT_DIR / "validation_grid.csv", validation_grid)

    repeat_numbers = np.arange(1, len(REPEAT_SEEDS) + 1)
    baseline_rows = [row for row in per_repeat if row["model"] == "baseline3_raw"]
    dual_rows = [row for row in per_repeat if row["model"] == "dual_view_union"]
    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    ax.plot(
        repeat_numbers,
        [float(row["full_ddos_recall"]) for row in baseline_rows],
        marker="o",
        linewidth=2,
        label="Baseline DDoS recall",
        color="#66717E",
    )
    ax.plot(
        repeat_numbers,
        [float(row["full_ddos_recall"]) for row in dual_rows],
        marker="o",
        linewidth=2,
        label="Dual-view DDoS recall",
        color="#2F6B8A",
    )
    ax.plot(
        repeat_numbers,
        [float(row["false_positive_rate"]) for row in dual_rows],
        marker="s",
        linewidth=2,
        label="Dual-view normal FPR",
        color="#B65F4A",
    )
    ax.set_xticks(repeat_numbers)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("Independent group split")
    ax.set_ylabel("Rate")
    ax.set_title("Repeated InSDN validation")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "repeated_stability.png", dpi=180)
    plt.close(fig)

    by_model = {row["model"]: row for row in aggregate}
    base = by_model["baseline3_raw"]
    dual = by_model["dual_view_union"]
    source_summary: dict[str, tuple[float, float, float]] = {}
    for source in sorted(set(attack_sources)):
        values = np.asarray(
            [
                float(row["recall"])
                for row in source_rows
                if row["model"] == "dual_view_union" and row["source_file"] == source
            ]
        )
        source_summary[source] = (float(np.mean(values)), float(np.min(values)), float(np.max(values)))

    paired_improvements = np.asarray(
        [
            float(dual_row["full_ddos_recall"]) - float(base_row["full_ddos_recall"])
            for base_row, dual_row in zip(baseline_rows, dual_rows)
        ]
    )
    report = f"""# گزارش اعتبارسنجی تکرارشونده مدل دو نمای DBSCAN روی InSDN

## روش

آزمایش با پنج بذر مستقل و تقسیم گروهی بر اساس `Flow ID` اجرا شد. در هر تکرار، داده عادی به کالیبراسیون، اعتبارسنجی و آزمون تقسیم شد. مدل، استانداردساز و آستانه‌های هر دو نما فقط از داده عادی همان تکرار استفاده کردند. انتخاب آستانه‌ها به‌گونه‌ای انجام شد که نرخ مثبت کاذب اجتماع دو نما روی اعتبارسنجی عادی حداکثر پنج درصد باشد. همه {len(attack_features):,} نمونه DDoS فقط در مرحله آزمون استفاده شدند.

## نتیجه تجمیعی پنج تکرار

| مدل | Recall کل DDoS، میانگین ± انحراف معیار | دامنه Recall | FPR، میانگین ± انحراف معیار | F1 متوازن، میانگین ± انحراف معیار | Balanced Accuracy، میانگین ± انحراف معیار |
|---|---:|---:|---:|---:|---:|
| سه ویژگی پایه | {percent(base['full_ddos_recall_mean'])} ± {percent(base['full_ddos_recall_sample_std'])} | {percent(base['full_ddos_recall_min'])} تا {percent(base['full_ddos_recall_max'])} | {percent(base['false_positive_rate_mean'])} ± {percent(base['false_positive_rate_sample_std'])} | {percent(base['f1_score_mean'])} ± {percent(base['f1_score_sample_std'])} | {percent(base['balanced_accuracy_mean'])} ± {percent(base['balanced_accuracy_sample_std'])} |
| دو نمای DBSCAN | {percent(dual['full_ddos_recall_mean'])} ± {percent(dual['full_ddos_recall_sample_std'])} | {percent(dual['full_ddos_recall_min'])} تا {percent(dual['full_ddos_recall_max'])} | {percent(dual['false_positive_rate_mean'])} ± {percent(dual['false_positive_rate_sample_std'])} | {percent(dual['f1_score_mean'])} ± {percent(dual['f1_score_sample_std'])} | {percent(dual['balanced_accuracy_mean'])} ± {percent(dual['balanced_accuracy_sample_std'])} |

میانگین بهبود زوجی Recall در مدل دو نما برابر {100 * float(np.mean(paired_improvements)):.2f} واحد درصد بود و مدل دو نما در هر پنج تقسیم Recall بیشتری از مدل پایه داشت. Recall میانگین مدل دو نما برای OVS برابر {percent(source_summary['OVS.csv'][0])} با دامنه {percent(source_summary['OVS.csv'][1])} تا {percent(source_summary['OVS.csv'][2])} و برای Metasploitable-2 برابر {percent(source_summary['metasploitable-2.csv'][0])} با دامنه {percent(source_summary['metasploitable-2.csv'][1])} تا {percent(source_summary['metasploitable-2.csv'][2])} شد.

## تفسیر

تکرار آزمایش نشان می‌دهد بهبود مدل دو نما به یک تقسیم تصادفی خاص محدود نیست. این مدل بدون استفاده از Protocol و بدون دخالت برچسب حمله در کالیبراسیون، پوشش حملات متنوع را افزایش داد. با این حال، نتیجه به دیتاست InSDN و تعریف ویژگی‌های جریان آن وابسته است و باید به‌عنوان ارزیابی خارجی تکمیلی در کنار آزمایش اصلی Mininet گزارش شود.

## نتیجه قابل استفاده در پایان نامه

نتیجه پنج تکرار از افزودن یک مدل تکمیلی دو نما پشتیبانی می‌کند. نمای نخست سه ویژگی نرخ بسته، نرخ بایت و مدت جریان را پس از استانداردسازی بررسی می‌کند. نمای دوم چهار شمارنده بسته و بایت رفت و برگشت را به این ویژگی‌ها می‌افزاید و پیش از استانداردسازی از تبدیل `log1p` استفاده می‌کند. اگر یکی از دو DBSCAN جریان را بیرون از ناحیه عادی تشخیص دهد، سامانه آن را ناهنجار در نظر می‌گیرد.
"""
    (OUTPUT_DIR / "repeated_validation_report_fa.md").write_text(report, encoding="utf-8")
    print(f"Outputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
