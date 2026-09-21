# SDN DDoS Detection with DBSCAN

This repository contains the implementation, evaluation summaries, and thesis documents for a master's research project on unsupervised DDoS detection in Software-Defined Networks using DBSCAN.

## خلاصه فارسی

این مخزن شامل کدهای جمع‌آوری آمار جریان OpenFlow، پیش‌پردازش، خوشه‌بندی DBSCAN، ارزیابی مستقل آزمایش Mininet و اعتبارسنجی تکمیلی روی InSDN است. پیاده‌سازی اصلی روی سناریوی کنترل‌شده حمله ICMP پرنرخ انجام شده است؛ بنابراین نتایج نباید به همه انواع DDoS و همه شبکه‌های عملیاتی تعمیم قطعی داده شوند.

## Research scope

- SDN testbed based on Mininet, Open vSwitch, OpenFlow 1.0, and a POX controller
- Lightweight flow features: packet rate, byte rate, and flow duration
- Normal-only DBSCAN calibration separated from attack evaluation
- Four independent laboratory repetitions with 896 test samples
- Five group-based InSDN splits for external validation
- A complementary dual-view DBSCAN detector for broader DDoS coverage

## Main results

| Evaluation | Recall | FPR | F1 | Balanced accuracy |
| --- | ---: | ---: | ---: | ---: |
| Mininet laboratory experiment | 100.00% | 11.16% | 94.71% | 94.42% |
| InSDN dual-view mean | 95.62% | 4.99% | 95.30% | 95.36% |

The InSDN result is a complementary external validation and does not replace the primary Mininet experiment.

## Repository layout

```text
src/                 Python collection and analysis scripts
data/                Data acquisition instructions only
results/lab/         Selected laboratory summaries and figures
results/external/    Repeated InSDN validation summaries and figures
docs/                Persian thesis, defense slides, and progress report
```

Raw traffic captures, the full InSDN dataset, virtual-machine images, Python environments, and temporary files are intentionally excluded.

## Python environment

Python 3.10 or later is recommended for the analysis scripts.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Mininet, Open vSwitch, and POX are system-level dependencies and are not installed by `requirements.txt`. The collection scripts must be executed on a Linux SDN testbed with the required privileges.

## Typical workflow

```bash
# Collect traffic after starting the POX controller and Mininet environment
sudo python3 src/collect_independent_runs.py

# Preprocess locally collected normal and high-rate ICMP flows
python src/preprocess_flows.py
python src/k_distance.py
python src/analyze_dbscan_full_v2.py

# Run the repeated external validation after obtaining InSDN
python src/analyze_insdn_dual_view_repeated.py
```

The scripts resolve paths relative to the repository root. See [`data/README.md`](data/README.md) for the expected input locations.

## Reproducibility notes

- Ground-truth attack labels are used only after fitting the unsupervised detector.
- Calibration, validation, and test partitions are separated.
- InSDN partitions are grouped by `Flow ID` to prevent flow leakage.
- External-validation thresholds are selected from normal validation data only.
- Repeat seeds and evaluation grids are recorded in the analysis scripts and result tables.

## Citation

Citation metadata are provided in [`CITATION.cff`](CITATION.cff). The external dataset reference is:

M. S. Elsayed, N.-A. Le-Khac, and A. D. Jurcut, “InSDN: A Novel SDN Intrusion Dataset,” *IEEE Access*, vol. 8, pp. 165263–165284, 2020. DOI: [10.1109/ACCESS.2020.3022633](https://doi.org/10.1109/ACCESS.2020.3022633).

## Access and license

This repository is initially intended to remain private while the thesis is under review. No permission to redistribute the thesis text, slides, or source code is granted by default. Review `LICENSE` and obtain approval from the supervisor and university before making the repository public.

