# Data Notes

This directory contains small benchmark feeder files used by the validation scripts.

- `case33bw.m` and `case69.m` are MATPOWER distribution test cases cached for reproducibility.
- `ieee123/` contains the IEEE 123-bus OpenDSS model files used by `simulations/ieee123_validation.py`.
- `public_ev/` is reserved for local public EV charging-session data. Large public-session CSV files are intentionally not committed.

To run the Palo Alto public-session replay, place the City of Palo Alto ChargePoint CSV at:

```text
data/public_ev/palo_alto_chargepoint.csv
```

or set:

```bash
PUBLIC_EV_DATA_PATH=/path/to/palo_alto_chargepoint.csv
```
