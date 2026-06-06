# Data

Raw futures market data is intentionally not committed.

To rerun the full audit, place OHLCV CSV files under `data/raw/` or pass explicit paths with environment variables:

```bash
NQ_RAW_1M_CSV=/path/to/databento_nq_1m.csv python src/overfit_replication_audit.py
```

Expected columns:

- `timestamp`
- `open`
- `high`
- `low`
- `close`
- `volume`

The committed reports and tables are enough to reproduce the packaged audit summary with:

```bash
python src/reproduce_metrics.py
```

