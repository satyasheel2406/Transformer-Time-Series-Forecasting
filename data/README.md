# Dataset

This project uses the **PJM Hourly Energy Consumption** dataset (AEP zone),
which contains real hourly electricity consumption readings (in megawatts)
collected by PJM Interconnection, a regional transmission organization in
the United States.

## Source

- Original source: PJM Interconnection LLC (https://www.pjm.com)
- Public Kaggle mirror: https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption
- File used: `AEP_hourly.csv` (American Electric Power zone), covering
  hourly readings from 2004-10-01 to 2018-08-03 (121,273 rows).

## Where to place it

Download `AEP_hourly.csv` from the Kaggle link above (or any mirror of the
same public dataset) and place it at:

```text
data/AEP_hourly.csv
```

with the original two columns:

```text
Datetime,AEP_MW
2004-12-31 01:00:00,13478.0
2004-12-31 02:00:00,12865.0
...
```

`src/data_loader.py` automatically renames these to the project's generic
`timestamp` / `target_value` schema, so any similarly-shaped hourly
energy/demand CSV (`Datetime,<value>`) can be dropped in as a drop-in
replacement.

## Note on dataset size

The full raw file spans almost 14 years (~121,000 hourly rows). To keep
training time reasonable on a normal laptop with a single CPU core, the
pipeline (`src/data_loader.py`, see `N_HOURS_USED` in `src/config.py`)
keeps only the most recent ~3 years (26,280 hourly rows) of real,
unmodified readings. No values in this dataset are fabricated or
synthetically generated - only a chronological subset is used.
