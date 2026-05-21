"""
export_sweep_table.py
=====================
Drop-in helper to export the FULL parameter sweep (all configurations
tried, not only the winners) as Supplementary Table S1.

Reviewers of a benchmark paper routinely ask to see the whole grid, so
that the chosen operating point is visibly not cherry-picked. Your sweep
notebook (notebook 2) already evaluates every configuration; this script
just makes sure every row is written to disk, with the selected operating
point flagged.

USAGE
-----
Inside notebook 2, after the sweep loop has built its list of result
dicts (one dict per configuration evaluated), call:

    from export_sweep_table import export_full_sweep
    export_full_sweep(all_sweep_results, selected_params)

where
    all_sweep_results : list[dict]
        every configuration tried, each dict containing at least
        'encoder', the swept parameter values, 'sp_ev', 'sp_no', 'snr'.
    selected_params : dict[str, dict]
        encoder name -> the winning parameter dict (so the chosen row
        can be flagged in the 'selected' column).

It writes results/parameter_sweep_full.csv.
"""

import os
import pandas as pd


def export_full_sweep(all_sweep_results, selected_params, results_dir="results"):
    """Write the complete sweep grid to a single CSV (Supplementary Table S1)."""
    df = pd.DataFrame(all_sweep_results)

    # Flag the configuration that was selected for each encoder.
    def _is_selected(row):
        sel = selected_params.get(row.get("encoder"))
        if not sel:
            return False
        for k, v in sel.items():
            if k in row and row[k] != v:
                return False
        return True

    df["selected"] = df.apply(_is_selected, axis=1)

    # Tidy ordering: encoder, then selected flag, then everything else.
    front = [c for c in ("encoder", "selected") if c in df.columns]
    rest = [c for c in df.columns if c not in front]
    df = df[front + rest]
    df = df.sort_values(["encoder", "selected"], ascending=[True, False])

    os.makedirs(results_dir, exist_ok=True)
    out = os.path.join(results_dir, "parameter_sweep_full.csv")
    df.to_csv(out, index=False)

    n_total = len(df)
    n_enc = df["encoder"].nunique() if "encoder" in df else 0
    print(f"Supplementary Table S1 written: {out}")
    print(f"  {n_total} configurations across {n_enc} encoders")
    print(f"  {int(df['selected'].sum())} rows flagged as the selected "
          f"operating point")
    return df
