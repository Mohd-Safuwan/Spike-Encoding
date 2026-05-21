# Adaptive Spike Encoding for Energy-Efficient SNN-Based DAS Microseismic Event Detection

Code, labels, and supplementary material for the paper:

> Shahabudin, M.S., Jaafar, J., Paputungan, I.V., Nadri, M.H.
> *Adaptive Spike Encoding for Energy-Efficient Spiking Neural Network-Based
> Microseismic Event Detection using Distributed Acoustic Sensing Data.*

This repository contains everything needed to reproduce the ten-encoder
benchmark, the AMSTE component ablation, the parameter sweep, and all
statistical tests reported in the paper.

---

## 1. What is in this repository

```
.
├── README.md                  This file
├── LICENSE                    MIT license
├── requirements.txt           Pinned Python environment
├── config.py                  Single place to set all data/output paths
├── amste_lib.py               Reusable AMSTE encoder + SNN (used by nb 9)
├── notebooks/                 The full pipeline (run in numbered order)
│   ├── 1_preprocessing_single_file.ipynb
│   ├── 2_parameter_sweep.ipynb
│   ├── 3_preprocessing_full_dataset.ipynb
│   ├── 4_snn_benchmark.ipynb
│   ├── 5_statistical_tests.ipynb
│   ├── 6_ablation_single_file.ipynb
│   ├── 7_ablation_preprocessing.ipynb
│   ├── 8_ablation_benchmark.ipynb
│   └── 9_supplementary.ipynb   Supplementary tables and figures (S1-S6)
├── data/
│   └── labels.csv             Manual event/noise labels for the 2000 files
├── results/                   CSV outputs reproduced by the notebooks
└── supplementary/             Tables and figures for the supplementary file
```

## 2. Dataset

The raw Distributed Acoustic Sensing (DAS) data are **not redistributed
here**. They are publicly available from the Geothermal Data Repository:

> Dadi, S., Titov, A. (2024). *Cape EGS: Frisco 2-P Well Stimulation
> Microseismic Data.* Geothermal Data Repository, Fervo Energy.
> https://gdr.openei.org/submissions/1664 — doi:10.15121/2479174

This study uses a balanced subset of **2000 SEGY records (1000 event +
1000 noise)** selected and labelled by manual visual inspection from the
STA/LTA-triggered records in that archive. The exact subset and its labels
are given in `data/labels.csv`, which lists, for every file:

| column            | meaning                                            |
|-------------------|----------------------------------------------------|
| file_id           | internal identifier (file_00001 ...)               |
| original_filename | the SEGY filename in the GDR archive               |
| label             | 1 = microseismic event, 0 = noise                  |
| class_name        | "event" or "noise"                                 |
| n_traces          | number of traces (366)                             |
| n_samples         | number of time samples (2000)                      |
| dt_ms             | sample interval in ms (0.5)                        |

The noise class consists of **false (non-event) STA/LTA triggers**, not
synthetic noise. With `data/labels.csv` and the GDR archive, the entire
experiment is reconstructible.

## 3. Environment

```bash
python -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Tested with Python 3.10, PyTorch 2.2.0, snnTorch 0.9.4 on an
NVIDIA Jetson AGX Orin (32 GB unified memory, Ubuntu 22.04).
The notebooks also run on a standard CUDA GPU or on CPU (slower).

## 4. Paths

All input/output locations are set in **one file**, `config.py`. Edit it
once to point `RAW_EVENT_DIR` and `RAW_NOISE_DIR` at your copy of the GDR
data; every notebook imports its paths from there. No path is hardcoded
inside a notebook.

## 5. How to reproduce the paper

Run the notebooks in numbered order.

| Step | Notebook | Produces |
|------|----------|----------|
| 1 | `1_preprocessing_single_file.ipynb` | Sanity check of the 10 encoders on one file (optional) |
| 2 | `2_parameter_sweep.ipynb` | Hyperparameter sweep; writes `results/parameter_sweep/` (Supplementary Table S1) |
| 3 | `3_preprocessing_full_dataset.ipynb` | Encodes all 2000 files with the swept parameters; writes `encoded/` and `labels.csv` |
| 4 | `4_snn_benchmark.ipynb` | Trains 10 encoders x 5 folds x 3 seeds = 150 runs; writes the CV and test result CSVs |
| 5 | `5_statistical_tests.ipynb` | Wilcoxon tests, energy analysis, learned-beta analysis, paper Tables 5-8 |
| 6 | `6_ablation_single_file.ipynb` | Sanity check of the 6 AMSTE ablation variants on one file (optional) |
| 7 | `7_ablation_preprocessing.ipynb` | Encodes all 2000 files with the 6 AMSTE ablation variants |
| 8 | `8_ablation_benchmark.ipynb` | Trains the ablation variants; writes the component-ablation table (Table 8) |
| 9 | `9_supplementary.ipynb` | Produces all supplementary tables and figures (S1-S6): full sweep grid, per-run results, confusion matrices, AMSTE sensitivity analysis, training curves, extra rasters |

Notebook 1 references one event file for all three of its sanity-check
blocks. To exercise the noise path as well, point the third `FILE_PATH`
at `config.SINGLE_FILE_NOISE`. This notebook is exploratory and is not
required to reproduce any reported result.

Fixed seeds: the train/validation/test split uses seed 42; the three
cross-validation repeats use seeds 42, 123, 456. With the same data these
reproduce the reported numbers.

## 6. Encoder parameters

The sweep-confirmed operating point for every encoder is documented in
`3_preprocessing_full_dataset.ipynb` and in Table 3 of the paper. The
proposed AMSTE encoder uses: alpha = 3.0, temporal lags = {1, 3, 8}
samples, minimum votes = 2, spatial aperture = 16 channels, spatial
threshold = 0.5.

## 7. Citation

If you use this code or the label set, please cite the paper above and the
GDR dataset (Dadi and Titov, 2024).

## 8. License

Code released under the MIT License (see `LICENSE`). The raw DAS data are
subject to the Geothermal Data Repository terms.

## 9. Contact

Mohd Safuwan Shahabudin — mohd_22010924@utp.edu.my
Department of Computer and Information Sciences,
Universiti Teknologi PETRONAS, Malaysia.
