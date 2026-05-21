"""
amste_lib.py
============
Reusable building blocks for the AMSTE DAS microseismic benchmark.

This module factors out three things that the notebooks otherwise repeat,
so that the supplementary sensitivity sweep (notebook 9, section S4) can
import them instead of copy-pasting:

  - amste_encode(...)   the AMSTE spike encoder (parameterised)
  - preprocess(...)     the shared SEGY preprocessing pipeline
  - DAS_SNN             the three-layer LIF-SNN classifier
  - train_eval(...)     train + evaluate the SNN on one encoded folder

The encoder and preprocessing code is identical to notebook 3; the SNN
and training code is identical to notebook 4. Keeping a single copy here
avoids drift between the notebooks and the supplement.

All hyperparameters default to the values in config.py.
"""

import os
import copy
import numpy as np

import config

# ---------------------------------------------------------------------------
# 1. PREPROCESSING  (identical to notebook 3)
# ---------------------------------------------------------------------------

def _bandpass(data, dt_ms):
    from scipy.signal import butter, filtfilt
    fs = 1000.0 / dt_ms
    nyq = 0.5 * fs
    b, a = butter(4, [np.clip(config.FILTER_LOW_HZ / nyq, 1e-6, 0.99),
                      np.clip(config.FILTER_HIGH_HZ / nyq, 1e-6, 0.99)],
                  btype="band")
    out = np.zeros_like(data)
    for i in range(data.shape[0]):
        out[i] = filtfilt(b, a, data[i])
    return out


def _window(data, dt_ms):
    s = max(0, int(config.WINDOW_START_MS / dt_ms))
    e = min(data.shape[1], int(config.WINDOW_END_MS / dt_ms))
    return data[:, s:e] if s < e else data


def _normalize_signed(data):
    """Per-trace signed normalisation to [-1, +1] (preserves polarity)."""
    out = np.zeros_like(data, dtype=np.float64)
    for i in range(data.shape[0]):
        t = data[i] - np.mean(data[i])
        m = np.max(np.abs(t))
        out[i] = t / m if m > 0 else t
    return out


def load_segy(filepath):
    """Load a SEGY file as (C, T) float array plus the sample interval."""
    import segyio
    with segyio.open(filepath, mode="r", ignore_geometry=True) as f:
        data = np.array([f.trace[i] for i in range(len(f.trace))])
        try:
            dt_ms = f.bin[segyio.BinField.Interval] / 1000.0
            if not (0.001 <= dt_ms <= 10.0):
                dt_ms = config.FORCE_DT_MS
        except Exception:
            dt_ms = config.FORCE_DT_MS
    return data, dt_ms


def preprocess_signed(filepath):
    """Full preprocessing -> signed [-1,+1] panel and dt_ms (AMSTE input)."""
    data, dt_ms = load_segy(filepath)
    filtered = _window(_bandpass(data, dt_ms), dt_ms)
    return _normalize_signed(filtered), dt_ms


# ---------------------------------------------------------------------------
# 2. AMSTE ENCODER  (identical to notebook 3, fully parameterised)
# ---------------------------------------------------------------------------

def amste_encode(panel, dt_ms=None,
                 alpha=3.0, lags=(1, 3, 8), min_votes=2,
                 ws=16, thr_s=0.5):
    """
    Adaptive Multi-Scale Spatio-Temporal Encoder.

    panel : signed-normalised DAS panel, shape (C, T), values in [-1, +1].
    Returns a binary float32 spike tensor of the same shape.

    The parameters are the four DAS-aware mechanisms; varying them one at
    a time is the basis of the section S4 sensitivity analysis.
    """
    from scipy.ndimage import maximum_filter1d, minimum_filter1d
    n_tr, n_smp = panel.shape

    # Step 1 -- per-channel MAD threshold.
    med_c = np.median(panel, axis=1, keepdims=True)
    mad_c = np.median(np.abs(panel - med_c), axis=1, keepdims=True)
    theta_c = np.maximum(alpha * 1.4826 * mad_c, 1e-9)

    # Steps 2-3 -- multi-scale bidirectional polarity vote.
    pos = np.zeros((n_tr, n_smp), dtype=np.int32)
    neg = np.zeros((n_tr, n_smp), dtype=np.int32)
    for lag in lags:
        if lag >= n_smp:
            continue
        diff = panel[:, lag:] - panel[:, :-lag]
        pos[:, lag:] += (diff > theta_c).astype(np.int32)
        neg[:, lag:] += (diff < -theta_c).astype(np.int32)
    candidate = (pos >= min_votes) | (neg >= min_votes)

    # Step 4 -- spatial coherence gate.
    abs_p = np.abs(panel)
    spat_max = maximum_filter1d(abs_p, size=ws, axis=0, mode="nearest")
    spat_min = minimum_filter1d(abs_p, size=ws, axis=0, mode="nearest")
    da_s = spat_max - spat_min

    return (candidate & (da_s > thr_s)).astype(np.float32)


# ---------------------------------------------------------------------------
# 3. SNN MODEL  (identical to notebook 4)
# ---------------------------------------------------------------------------

def build_model(num_inputs=None):
    """Construct the three-layer LIF-SNN. Imports torch lazily."""
    import torch.nn as nn
    import snntorch as snn
    from snntorch import surrogate

    num_inputs = num_inputs or config.NUM_INPUTS
    b1, b2, b3 = config.BETA_INIT

    class DAS_SNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.num_steps = config.NUM_STEPS
            grad = surrogate.fast_sigmoid(slope=config.SURROGATE_SLOPE)
            self.drop_in = nn.Dropout(p=0.1)
            self.fc1 = nn.Linear(num_inputs, config.NUM_HIDDEN1)
            self.lif1 = snn.Leaky(beta=b1, learn_beta=True, spike_grad=grad,
                                  threshold=config.THRESHOLD,
                                  reset_mechanism="subtract")
            self.fc2 = nn.Linear(config.NUM_HIDDEN1, config.NUM_HIDDEN2)
            self.lif2 = snn.Leaky(beta=b2, learn_beta=True, spike_grad=grad,
                                  threshold=config.THRESHOLD,
                                  reset_mechanism="subtract")
            self.fc3 = nn.Linear(config.NUM_HIDDEN2, config.NUM_OUTPUTS)
            self.lif3 = snn.Leaky(beta=b3, learn_beta=True, spike_grad=grad,
                                  threshold=config.THRESHOLD,
                                  reset_mechanism="subtract")
            for m in (self.fc1, self.fc2, self.fc3):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

        def forward(self, x):
            import torch
            m1, m2, m3 = (self.lif1.init_leaky(), self.lif2.init_leaky(),
                          self.lif3.init_leaky())
            out = []
            for t in range(x.shape[0]):
                c = self.drop_in(x[t])
                s1, m1 = self.lif1(self.fc1(c), m1)
                s2, m2 = self.lif2(self.fc2(s1), m2)
                s3, m3 = self.lif3(self.fc3(s2), m3)
                out.append(s3)
            return torch.stack(out), None

    return DAS_SNN()


# ---------------------------------------------------------------------------
# 4. TRAIN + EVALUATE  (reduced wrapper around notebook 4's protocol)
# ---------------------------------------------------------------------------

def train_eval(encoded_dir, encoder_subdir, labels_csv=None,
               folds=3, seeds=None, num_inputs=None):
    """
    Train the SNN on one encoded folder and return mean detection F1 and
    estimated inference energy.

    encoded_dir    : root containing per-encoder subfolders of .npy files.
    encoder_subdir : the subfolder name to train on.
    folds, seeds   : reduced protocol for the sensitivity sweep.

    Returns dict(f1=..., energy_uj=...). This is a deliberately compact
    wrapper; for the full headline numbers use notebook 4.
    """
    import torch
    from torch.utils.data import Dataset, DataLoader
    import snntorch.functional as SF
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import f1_score

    labels_csv = labels_csv or config.LABELS_CSV
    seeds = seeds or [config.CV_SEEDS[0]]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    import pandas as pd
    df = pd.read_csv(labels_csv)
    file_ids = df["file_id"].values
    y = df["label"].values.astype(int)
    enc_dir = os.path.join(encoded_dir, encoder_subdir)

    def _load(fid):
        s = np.load(os.path.join(enc_dir, f"{fid}.npy")).astype(np.float32)
        n_tr, n_smp = s.shape
        nb = n_smp // config.BIN_FACTOR
        b = s[:, :nb * config.BIN_FACTOR].reshape(
            n_tr, nb, config.BIN_FACTOR).sum(axis=2)
        return (b > 0).astype(np.float32).T          # (T, C)

    class _DS(Dataset):
        def __init__(self, ids, lab):
            self.ids, self.lab = ids, lab

        def __len__(self):
            return len(self.ids)

        def __getitem__(self, i):
            return (torch.tensor(_load(self.ids[i])),
                    torch.tensor(int(self.lab[i])))

    f1s, energies = [], []
    for seed in seeds:
        skf = StratifiedKFold(n_splits=config.NUM_FOLDS, shuffle=True,
                              random_state=seed)
        for fi, (tr, va) in enumerate(skf.split(file_ids, y)):
            if fi >= folds:
                break
            torch.manual_seed(seed)
            np.random.seed(seed)
            model = build_model(num_inputs).to(device)
            loss_fn = SF.mse_count_loss(correct_rate=config.CORRECT_RATE,
                                        incorrect_rate=config.INCORRECT_RATE)
            opt = torch.optim.Adam(model.parameters(),
                                   lr=config.LEARNING_RATE)
            tl = DataLoader(_DS(file_ids[tr], y[tr]),
                            batch_size=config.BATCH_SIZE, shuffle=True)
            vl = DataLoader(_DS(file_ids[va], y[va]),
                            batch_size=config.BATCH_SIZE)

            best_f1 = -1.0
            for _ in range(config.NUM_EPOCHS):
                model.train()
                for xb, yb in tl:
                    xb = xb.permute(1, 0, 2).to(device)
                    yb = yb.to(device)
                    out, _ = model(xb)
                    loss = loss_fn(out, yb)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                # validation
                model.eval()
                preds, tgts, in_spk, l1_spk = [], [], 0, 0
                with torch.no_grad():
                    for xb, yb in vl:
                        xb = xb.permute(1, 0, 2).to(device)
                        out, _ = model(xb)
                        cnt = out.sum(dim=0)
                        preds.extend(cnt.argmax(1).cpu().numpy())
                        tgts.extend(yb.numpy())
                        in_spk += float(xb.sum().item())
                f1 = f1_score(tgts, preds, zero_division=0)
                best_f1 = max(best_f1, f1)
            f1s.append(best_f1)
            # energy: SynOps estimate from input spike count
            synops = in_spk * config.NUM_HIDDEN1
            energies.append(synops * config.ENERGY_PER_SYNOP_PJ * 1e-6)

    return {"f1": float(np.mean(f1s)),
            "energy_uj": float(np.mean(energies))}
