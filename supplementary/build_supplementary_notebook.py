#!/usr/bin/env python3
"""
build_supplementary_notebook.py
===============================
Generates notebooks/9_supplementary.ipynb -- a single notebook that
produces every supplementary artifact a reviewer of the benchmark is
likely to ask for:

  S1  Full parameter-sweep grid (all configurations, winners flagged)
  S2  Raw per-fold / per-seed results for all 150 runs
  S3  Held-out confusion matrices (figure + CSV)
  S4  AMSTE sensitivity analysis (one-parameter-at-a-time)
  S5  Representative training curves
  S6  Extra spike-raster examples

It reuses the encoders, model, and training code already in the
repository notebooks; it does not redefine the science.
"""

import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))

def code(text):
    cells.append(nbf.v4.new_code_cell(text))


# ---------------------------------------------------------------- header
md("""# 9 Supplementary

Generates the supplementary material for the paper. Run **after**
notebooks 1-8 have completed, so that the sweep results and the 150-run
benchmark CSVs exist.

Each section writes one artifact into `supplementary/`:

| Section | Artifact | Reviewer question it answers |
|---|---|---|
| S1 | `S1_parameter_sweep_full.csv` | Was the operating point cherry-picked? |
| S2 | `S2_per_run_results.csv` | Are the Wilcoxon tests checkable? |
| S3 | `S3_confusion_matrices.{png,csv}` | What is the actual TP/TN/FP/FN split? |
| S4 | `S4_amste_sensitivity.{csv,png}` | Is AMSTE fragile to its parameters? |
| S5 | `S5_training_curves.png` | Does early stopping / the degeneracy filter work? |
| S6 | `S6_extra_rasters.png` | Does the encoder behaviour generalise beyond Fig. 6? |

Paths come from `config.py`; edit that file, not this notebook.""")

# ---------------------------------------------------------------- setup
md("## Setup")
code("""# --- repository configuration -------------------------------------
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '..')))
import config
# ------------------------------------------------------------------
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SUPP = config.SUPPLEMENTARY_DIR
os.makedirs(SUPP, exist_ok=True)
print('Supplementary output folder:', SUPP)

# Result-file locations written by notebooks 2, 4 and 5.
SWEEP_DIR  = config.SWEEP_RESULTS_DIR
CV_CSV     = os.path.join(config.RESULTS_DIR, 'mode1_within_cv_results.csv')
TEST_CSV   = os.path.join(config.RESULTS_DIR, 'mode1_test_results.csv')

ENCODER_NAMES = config.ENCODER_NAMES
ENCODER_TYPE  = config.ENCODER_TYPE
""")

# ---------------------------------------------------------------- S1
md("""## S1 -- Full parameter-sweep grid

Exports **every** configuration evaluated in the sweep (notebook 2), not
only the selected operating point, with the chosen configuration flagged.
This is the standard evidence that the operating point was not
cherry-picked.

If your sweep notebook already saved a full per-configuration CSV, this
cell simply consolidates it. If it only saved the winners, re-run
notebook 2 with `supplementary/export_sweep_table.py` (see that file's
docstring) to capture the full grid first.""")
code("""# Collect every CSV the sweep wrote and concatenate them.
sweep_files = sorted(glob.glob(os.path.join(SWEEP_DIR, '*.csv')))
print(f'Found {len(sweep_files)} sweep CSV(s) in {SWEEP_DIR}')

if not sweep_files:
    print('No sweep CSVs found. Run notebook 2 first '
          '(and export_sweep_table.py for the full grid).')
else:
    frames = []
    for f in sweep_files:
        d = pd.read_csv(f)
        d['source_file'] = os.path.basename(f)
        frames.append(d)
    sweep = pd.concat(frames, ignore_index=True)

    # If a 'selected' column is absent, the sweep saved winners only.
    if 'selected' not in sweep.columns:
        sweep['selected'] = True
        print('NOTE: no "selected" column -- this CSV appears to hold the '
              'winners only. For the full grid, re-run notebook 2 with '
              'export_sweep_table.py.')

    out = os.path.join(SUPP, 'S1_parameter_sweep_full.csv')
    sweep.to_csv(out, index=False)
    print(f'Wrote {out}  ({len(sweep)} rows)')
    display(sweep.head(12))
""")

# ---------------------------------------------------------------- S2
md("""## S2 -- Raw per-run results (all 150 runs)

The paper reports cross-validation metrics as mean +/- std over 15
matched runs per encoder. This section exports the underlying
per-fold/per-seed rows so the paired Wilcoxon tests are fully
reproducible by a reader.""")
code("""if not os.path.exists(CV_CSV):
    print(f'CV results not found: {CV_CSV}\\nRun notebook 4 first.')
else:
    cv = pd.read_csv(CV_CSV)

    metric_cols = [c for c in
                   ['f1', 'recall', 'fnr', 'fpr', 'auprc',
                    'accuracy', 'precision', 'specificity', 'auc', 'mcc']
                   if c in cv.columns]
    id_cols = [c for c in ['encoder', 'fold', 'seed'] if c in cv.columns]

    per_run = cv[id_cols + metric_cols].copy()
    per_run = per_run.sort_values(id_cols).reset_index(drop=True)

    out = os.path.join(SUPP, 'S2_per_run_results.csv')
    per_run.to_csv(out, index=False)
    print(f'Wrote {out}  ({len(per_run)} rows)')

    # Quick sanity check: 15 runs per encoder expected.
    counts = per_run.groupby('encoder').size()
    print('\\nRuns per encoder:')
    print(counts.to_string())
    if (counts != 15).any():
        print('\\nWARNING: not every encoder has 15 runs -- check notebook 4.')
    display(per_run.head(12))
""")

# ---------------------------------------------------------------- S3
md("""## S3 -- Held-out confusion matrices

The benchmark notebook stores the confusion matrix of every run (the
`cm` field returned by `_metrics`). This section reads the test-set
results and renders the confusion matrices for the headline encoders.

If the test CSV does not contain a `cm` column, the cell falls back to
reconstructing the matrix from accuracy/recall/precision and the known
class balance (150 event + 150 noise).""")
code("""import ast

FOCUS = ['AMSTE', 'ST-MW']   # headline encoders; add more if you wish

def cm_from_row(row):
    \"\"\"Return (tn, fp, fn, tp) for one test row.\"\"\"
    if 'cm' in row and pd.notna(row['cm']):
        cm = row['cm']
        if isinstance(cm, str):
            cm = ast.literal_eval(cm)
        cm = np.array(cm)
        tn, fp, fn, tp = cm.ravel()
        return int(tn), int(fp), int(fn), int(tp)
    # Fallback: reconstruct from metrics + class balance.
    n_pos = n_neg = 150
    rec  = row.get('recall', np.nan)
    prec = row.get('precision', np.nan)
    tp = int(round(rec * n_pos))
    fn = n_pos - tp
    fp = int(round(tp / prec - tp)) if prec and prec > 0 else 0
    tn = n_neg - fp
    return tn, fp, fn, tp

if not os.path.exists(TEST_CSV):
    print(f'Test results not found: {TEST_CSV}\\nRun notebook 4 first.')
else:
    test = pd.read_csv(TEST_CSV)
    rows = []
    fig, axes = plt.subplots(1, len(FOCUS), figsize=(4 * len(FOCUS), 4))
    if len(FOCUS) == 1:
        axes = [axes]

    for ax, enc in zip(axes, FOCUS):
        sub = test[test['encoder'] == enc]
        if len(sub) == 0:
            ax.set_title(f'{enc}: no data'); ax.axis('off'); continue
        tn, fp, fn, tp = cm_from_row(sub.iloc[0])
        cm = np.array([[tn, fp], [fn, tp]])
        rows.append({'encoder': enc, 'TN': tn, 'FP': fp,
                     'FN': fn, 'TP': tp})

        im = ax.imshow(cm, cmap='Blues')
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, str(v), ha='center', va='center',
                    fontsize=14,
                    color='white' if v > cm.max() / 2 else 'black')
        ax.set_xticks([0, 1]); ax.set_xticklabels(['Noise', 'Event'])
        ax.set_yticks([0, 1]); ax.set_yticklabels(['Noise', 'Event'])
        ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
        ax.set_title(f'{enc} (held-out test set)')

    plt.tight_layout()
    fig_path = os.path.join(SUPP, 'S3_confusion_matrices.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.show()

    cm_df = pd.DataFrame(rows)
    csv_path = os.path.join(SUPP, 'S3_confusion_matrices.csv')
    cm_df.to_csv(csv_path, index=False)
    print(f'Wrote {fig_path}')
    print(f'Wrote {csv_path}')
    display(cm_df)
""")

# ---------------------------------------------------------------- S4
md("""## S4 -- AMSTE sensitivity analysis

This is the most important supplementary item. It varies **one AMSTE
parameter at a time** around the operating point used in the paper
(alpha = 3.0, lags = {1,3,8}, v_min = 2, W_s = 16, theta_s = 0.5) and
records detection F1 and estimated energy for each setting.

It re-encodes a fixed evaluation subset for each parameter value and
trains the same SNN. To keep runtime manageable it uses a reduced
protocol (one seed, fewer folds); this is appropriate because the goal
is a *trend*, not a new headline number. Set `FULL_PROTOCOL = True` to
use the full 5-fold x 3-seed protocol.

Runtime note: this cell trains many short SNN runs. On the Jetson AGX
Orin used in the paper, the reduced protocol takes roughly 1-2 hours.
The grids below are deliberately small; widen them only if needed.""")
code("""# ---- sensitivity grids -- centred on the paper's operating point ----
SENS_GRID = {
    'alpha':     [2.0, 2.5, 3.0, 3.5, 4.0],          # MAD multiplier
    'lags':      [[1], [1, 3], [1, 3, 8], [1, 3, 8, 16]],
    'min_votes': [1, 2, 3],
    'ws':        [8, 12, 16, 24, 32],                # spatial aperture (ch)
    'thr_s':     [0.3, 0.4, 0.5, 0.6, 0.7],          # spatial gate
}
OPERATING_POINT = dict(alpha=3.0, lags=[1, 3, 8],
                       min_votes=2, ws=16, thr_s=0.5)

FULL_PROTOCOL = False        # True -> 5 folds x 3 seeds (much slower)
SENS_SEEDS    = config.CV_SEEDS if FULL_PROTOCOL else [config.CV_SEEDS[0]]
SENS_FOLDS    = config.NUM_FOLDS if FULL_PROTOCOL else 3

print('Sensitivity protocol:',
      f'{SENS_FOLDS} folds x {len(SENS_SEEDS)} seed(s)')
print('Operating point:', OPERATING_POINT)
""")
md("""**Implementation note.** The next cell is a template. It assumes the
AMSTE encoder function and the SNN training routine are importable from
the repository. The cleanest way is to factor the AMSTE encoder out of
notebook 3 and the `train_one_run` / `DAS_SNN` code out of notebook 4
into a small `amste_lib.py` at the repository root, then
`from amste_lib import amste_encode, train_eval`. Until that refactor is
done, paste the AMSTE encoder and the training helper into the cell
where indicated.""")
code("""# ===================================================================
# S4 sensitivity sweep -- TEMPLATE
# ===================================================================
# Requires two callables:
#   amste_encode(panel, dt_ms, alpha, lags, min_votes, ws, thr_s) -> spikes
#   train_eval(encoded_dir, folds, seeds) -> dict(f1=..., energy_uj=...)
#
# Both already exist inside notebooks 3 and 4. Factor them into
# amste_lib.py (recommended) or paste them here, then remove the
# `raise` below.
# -------------------------------------------------------------------
raise NotImplementedError(
    'Provide amste_encode() and train_eval() -- see the markdown above. '
    'Then delete this raise and run the sweep.')

# Reference implementation of the sweep loop (runs once the two
# callables are available):
#
# records = []
# for param, values in SENS_GRID.items():
#     for val in values:
#         cfg = dict(OPERATING_POINT)
#         cfg[param] = val
#         enc_dir = os.path.join(config.ENCODED_DIR,
#                                f'_sens_{param}_{val}')
#         # 1. re-encode the evaluation subset with cfg
#         #    (loop over SEGY files -> amste_encode -> save .npy)
#         # 2. train + evaluate
#         res = train_eval(enc_dir, folds=SENS_FOLDS, seeds=SENS_SEEDS)
#         records.append(dict(parameter=param, value=str(val),
#                              is_operating_point=(val == OPERATING_POINT[param]),
#                              f1=res['f1'], energy_uj=res['energy_uj']))
#
# sens = pd.DataFrame(records)
# sens.to_csv(os.path.join(SUPP, 'S4_amste_sensitivity.csv'), index=False)
""")
md("""Once `S4_amste_sensitivity.csv` exists, the cell below renders it.
You can run this part independently of the sweep above.""")
code("""sens_csv = os.path.join(SUPP, 'S4_amste_sensitivity.csv')
if not os.path.exists(sens_csv):
    print('Run the S4 sweep first to create', sens_csv)
else:
    sens = pd.read_csv(sens_csv)
    params = sens['parameter'].unique()
    fig, axes = plt.subplots(1, len(params),
                             figsize=(3.2 * len(params), 3.2))
    if len(params) == 1:
        axes = [axes]
    for ax, p in zip(axes, params):
        sub = sens[sens['parameter'] == p]
        ax.plot(range(len(sub)), sub['f1'], 'o-', label='F1')
        ax.set_xticks(range(len(sub)))
        ax.set_xticklabels(sub['value'], rotation=45, ha='right')
        op = sub[sub['is_operating_point']]
        if len(op):
            idx = list(sub['value']).index(op.iloc[0]['value'])
            ax.axvline(idx, color='red', ls='--', lw=1,
                       label='operating point')
        ax.set_title(p); ax.set_ylabel('CV F1'); ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    plt.tight_layout()
    out = os.path.join(SUPP, 'S4_amste_sensitivity.png')
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.show()
    print('Wrote', out)
""")

# ---------------------------------------------------------------- S5
md("""## S5 -- Representative training curves

Plots loss and F1 against epoch for a representative run, showing the
20-epoch burn-in, early stopping, and that the degeneracy filter keeps
the optimiser away from the all-positive collapse (F1 = 0.667).

This requires per-epoch logging. If notebook 4 wrote a per-epoch history
(e.g. one CSV per run, or a TensorBoard log), point `HISTORY_CSV` at it.
If not, add a few lines to `train_one_run` in notebook 4 to append
`(epoch, train_loss, val_loss, val_f1)` to a list and save it.""")
code("""HISTORY_CSV = os.path.join(config.RESULTS_DIR, 'training_history.csv')

if not os.path.exists(HISTORY_CSV):
    print(f'No per-epoch history at {HISTORY_CSV}.')
    print('Add epoch logging to train_one_run() in notebook 4 -- save '
          'columns: encoder, fold, seed, epoch, train_loss, val_loss, '
          'val_f1.')
else:
    hist = pd.read_csv(HISTORY_CSV)
    # Pick one representative AMSTE run.
    one = hist[(hist['encoder'] == 'AMSTE') &
               (hist['fold'] == hist['fold'].min()) &
               (hist['seed'] == config.CV_SEEDS[0])]
    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(one['epoch'], one['train_loss'], label='train loss')
    ax1.plot(one['epoch'], one['val_loss'],   label='val loss')
    ax1.set_xlabel('epoch'); ax1.set_ylabel('loss'); ax1.legend(loc='upper right')
    ax2 = ax1.twinx()
    ax2.plot(one['epoch'], one['val_f1'], color='green', ls=':',
             label='val F1')
    ax2.axhline(0.667, color='red', ls='--', lw=1,
                label='degenerate F1 (0.667)')
    ax2.axvline(config.BURNIN_EPOCHS, color='grey', ls='--', lw=1,
                label='burn-in end')
    ax2.set_ylabel('val F1'); ax2.legend(loc='lower right')
    plt.title('Representative AMSTE training run')
    plt.tight_layout()
    out = os.path.join(SUPP, 'S5_training_curves.png')
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.show()
    print('Wrote', out)
""")

# ---------------------------------------------------------------- S6
md("""## S6 -- Extra spike-raster examples

Figure 6 in the paper shows one event panel and one noise panel. This
section renders the spike rasters for a few additional matched pairs
(e.g. a weak event, a strong event, a noisy false trigger) to show the
encoder behaviour generalises beyond a single hand-picked example.

Pick the file_ids you want from `data/labels.csv` and list them below.""")
code("""# file_ids to visualise -- choose a spread of difficulty from labels.csv
EXTRA_EXAMPLES = {
    'weak event':       'file_00007',   # <- replace with real ids
    'strong event':     'file_00042',
    'noisy false trig': 'file_01510',
}
RASTER_ENCODERS = ['Rate', 'ST-MW', 'AMSTE']   # one dense, one spatial, ours

missing = []
fig, axes = plt.subplots(len(RASTER_ENCODERS), len(EXTRA_EXAMPLES),
                         figsize=(4 * len(EXTRA_EXAMPLES),
                                  2.6 * len(RASTER_ENCODERS)),
                         squeeze=False)
for j, (label, fid) in enumerate(EXTRA_EXAMPLES.items()):
    for i, enc in enumerate(RASTER_ENCODERS):
        npy = os.path.join(config.ENCODED_DIR, enc, f'{fid}.npy')
        ax = axes[i][j]
        if not os.path.exists(npy):
            missing.append(npy); ax.axis('off')
            ax.set_title(f'{enc} / {fid}\\n(missing)'); continue
        spikes = np.load(npy)
        ch, t = np.where(spikes > 0)
        ax.scatter(t, ch, s=0.4, marker='.', color='black')
        ax.set_xlim(0, spikes.shape[1]); ax.set_ylim(0, spikes.shape[0])
        ax.invert_yaxis()
        if i == 0:
            ax.set_title(f'{label}\\n({fid})')
        if j == 0:
            ax.set_ylabel(f'{enc}\\nchannel')
        ax.set_xlabel('time sample')
plt.tight_layout()
out = os.path.join(SUPP, 'S6_extra_rasters.png')
plt.savefig(out, dpi=300, bbox_inches='tight')
plt.show()
if missing:
    print('Missing encoded files (run notebook 3, or fix file_ids):')
    for m in missing:
        print('  ', m)
else:
    print('Wrote', out)
""")

# ---------------------------------------------------------------- close
md("""## Done

`supplementary/` now holds the artifacts. Assemble them into a single
supplementary PDF (or a zip) for the journal submission, and reference
them from the manuscript as Tables S1-S2 and Figures S1-S4.

The S4 sensitivity sweep is the only part that needs the encoder and
training code wired in; see the note in section S4. Everything else runs
directly from the result CSVs produced by notebooks 2, 4 and 5.""")

nb['cells'] = cells
nb['metadata'] = {
    'kernelspec': {'display_name': 'Python 3', 'language': 'python',
                   'name': 'python3'},
    'language_info': {'name': 'python', 'version': '3.10'},
}

out_path = '/home/claude/repo/notebooks/9_supplementary.ipynb'
with open(out_path, 'w') as f:
    nbf.write(nb, f)
print('Wrote', out_path, 'with', len(cells), 'cells')
