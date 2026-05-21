"""
Central configuration for the AMSTE DAS microseismic SNN benchmark.

EDIT THE PATHS IN SECTION 1 ONCE, then run the notebooks in numbered order.
Every notebook imports its paths from this file, so no path is hardcoded
inside any notebook.
"""

import os

# ============================================================================
# 1. PATHS  --  edit these to match your machine
# ============================================================================

# Root of this repository (auto-detected; usually no need to change).
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

# Raw DAS SEGY data downloaded from the Geothermal Data Repository
# (doi:10.15121/2479174). Point these at your local copy. The event/noise
# split corresponds to the labels in data/labels.csv.
RAW_EVENT_DIR = os.path.join(REPO_ROOT, "raw_data", "MS_Event")
RAW_NOISE_DIR = os.path.join(REPO_ROOT, "raw_data", "Noise")

# Where encoded .npy spike arrays are written by notebook 3.
ENCODED_DIR = os.path.join(REPO_ROOT, "encoded")

# Where ablation-variant encoded arrays are written by notebook 6.
ENCODED_ABLATION_DIR = os.path.join(REPO_ROOT, "encoded_ablation")

# Where CSV results, checkpoints, and tables are written.
RESULTS_DIR = os.path.join(REPO_ROOT, "results")
CKPT_DIR = os.path.join(RESULTS_DIR, "checkpoints")
SUPPLEMENTARY_DIR = os.path.join(REPO_ROOT, "supplementary")

# Committed label manifest (ships with the repository).
LABELS_CSV = os.path.join(REPO_ROOT, "data", "labels.csv")

# Notebook 1 -- single-file sanity check.
SINGLE_FILE_EVENT = os.path.join(RAW_EVENT_DIR, "REPLACE_WITH_ONE_EVENT.segy")
SINGLE_FILE_NOISE = os.path.join(RAW_NOISE_DIR, "REPLACE_WITH_ONE_NOISE.segy")
SINGLE_OUTPUT_DIR = os.path.join(RESULTS_DIR, "single_file_check")

# Notebook 2 -- parameter sweep output.
SWEEP_RESULTS_DIR = os.path.join(RESULTS_DIR, "parameter_sweep")

for _d in (ENCODED_DIR, ENCODED_ABLATION_DIR, RESULTS_DIR, CKPT_DIR,
           SUPPLEMENTARY_DIR, SINGLE_OUTPUT_DIR, SWEEP_RESULTS_DIR):
    os.makedirs(_d, exist_ok=True)

# ============================================================================
# 2. REPRODUCIBILITY  --  do not change; these define the published results
# ============================================================================

SPLIT_SEED = 42                 # train/val/test stratified split
CV_SEEDS = [42, 123, 456]       # three cross-validation repeats
NUM_FOLDS = 5                   # stratified k-fold
TEST_SIZE = 0.15                # held-out fraction (300 of 2000 files)

# ============================================================================
# 3. PREPROCESSING  --  identical across all encoders
# ============================================================================

FORCE_DT_MS = 0.5               # 2 kHz DAS interrogator
FILTER_LOW_HZ = 50
FILTER_HIGH_HZ = 250
WINDOW_START_MS = 0
WINDOW_END_MS = 1000
PRE_EVENT_END_MS = 200.0        # SFBE fixed LTA baseline window

# ============================================================================
# 4. SNN ARCHITECTURE AND TRAINING
# ============================================================================

NUM_INPUTS = 366                # DAS traces
NUM_HIDDEN1 = 128
NUM_HIDDEN2 = 64
NUM_OUTPUTS = 2
BIN_FACTOR = 4                  # 2000 samples -> 500 timesteps
NUM_STEPS = 500

BETA_INIT = (0.85, 0.90, 0.95)  # learnable membrane decay per layer
THRESHOLD = 1.0
SURROGATE_SLOPE = 25

BATCH_SIZE = 256
NUM_EPOCHS = 100
LEARNING_RATE = 2e-3
LR_FACTOR = 0.5
LR_PATIENCE = 5
EARLY_STOP_PATIENCE = 15
BURNIN_EPOCHS = 20
CORRECT_RATE = 0.8
INCORRECT_RATE = 0.2

# ============================================================================
# 5. ENERGY MODEL
# ============================================================================

ENERGY_PER_SYNOP_PJ = 23.6      # Intel Loihi-2 estimate (pJ per synaptic op)

# ============================================================================
# 6. ENCODERS
# ============================================================================

ENCODER_NAMES = [
    "Rate", "Phase", "TTFS",
    "Delta-Mod", "PDE", "ATDE", "MASTE",
    "ST-MW", "AMSTE", "SFBE",
]

ENCODER_TYPE = {
    "Rate": "Baseline", "Phase": "Baseline", "TTFS": "Baseline",
    "Delta-Mod": "Proposed", "PDE": "Proposed", "ATDE": "Proposed",
    "MASTE": "Proposed", "ST-MW": "Proposed", "AMSTE": "Proposed",
    "SFBE": "Proposed",
}
