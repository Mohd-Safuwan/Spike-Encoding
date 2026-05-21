# data/

## labels.csv

The manual event/noise label manifest for the 2000-file subset used in
the paper. This is the single most important data artifact for
reproducibility: with it plus the public GDR archive, the exact
experiment can be rebuilt.

**This file is not yet in the repository.** Generate it once, then commit
it:

1. Set `RAW_EVENT_DIR` and `RAW_NOISE_DIR` in `config.py` to your local
   copy of the GDR data.
2. Run `notebooks/3_preprocessing_full_dataset.ipynb`. It writes a
   `labels.csv` into the encoded-output folder.
3. Copy that `labels.csv` into this `data/` folder and commit it.

Do **not** commit the encoded `.npy` arrays or the raw SEGY files (the
`.gitignore` already excludes them). Only `labels.csv` belongs here.

### Columns

| column            | meaning                                         |
|-------------------|-------------------------------------------------|
| file_id           | internal identifier (file_00001 ...)            |
| label             | 1 = microseismic event, 0 = noise               |
| class_name        | "event" or "noise"                              |
| original_filename | the SEGY filename in the GDR archive            |
| n_traces          | number of traces (366)                          |
| n_samples         | number of time samples (2000)                   |
| dt_ms             | sample interval in ms (0.5)                     |

`original_filename` is what lets a reader map each labelled record back
to the public GDR archive, so keep that column.
