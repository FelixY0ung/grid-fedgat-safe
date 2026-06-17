# Grid-FedGAT-Safe

Safety-projected federated graph-feature learning for electric-vehicle charging coordination.

This repository contains the simulation code, manuscript source, and reproducibility artifacts for the ICPICN 2026 paper draft `Safety-Projected Federated Graph Learning for EV Charging Coordination`.

The core idea is to keep learning and safety enforcement separate: a federated graph-feature model proposes station-level charging power, then a convex projection and AC feasibility filter enforce charger, service, transformer, and voltage constraints before actuation.

## License

This code is released for noncommercial research and educational use under the PolyForm Noncommercial License 1.0.0. This is source-available, not OSI-approved open source, because the license intentionally restricts commercial use.

Commercial use requires separate permission from the copyright holder.

## Repository Layout

```text
simulations/   Reproducible experiment and validation scripts
tools/         Manuscript build/postprocess and readiness checks
data/          Small benchmark feeder files and data instructions
results/       Generated CSV and Markdown result summaries
*.md           Manuscript source and submission notes
```

Generated Word/PDF artifacts under `build/` and large third-party public-session CSV files are intentionally excluded from version control.

## Installation

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The IEEE 123-bus validator requires OpenDSSDirect.py and an OpenDSS-compatible runtime. The manuscript build tools additionally require Pandoc, LibreOffice, Poppler tools such as `pdfinfo` and `pdftotext`, and the ICPICN Word template if you want to regenerate DOCX/PDF submission files.

## Reproducing Experiments

Run commands from the repository root.

```bash
python simulations/pilot_grid_fedgat.py
python simulations/robust_grid_fedgat.py
python simulations/ac_validation.py
python simulations/ieee33_validation.py
python simulations/matpower_validation.py
python simulations/ieee123_validation.py
python simulations/reserve_pareto_sensitivity.py
```

The public Palo Alto ChargePoint replay expects the source CSV at `data/public_ev/palo_alto_chargepoint.csv`. You can also point to another local copy:

```bash
PUBLIC_EV_DATA_PATH=/path/to/palo_alto_chargepoint.csv \
  python simulations/public_ev_validation.py
```

ACN-Data ingestion is credential-dependent and is kept separate:

```bash
ACN_API_TOKEN=... python simulations/acn_data_validation.py
ACN_DATA_JSON=/path/to/export.json python simulations/acn_data_validation.py
```

## Manuscript Checks

The current submission candidate is generated outside version control under `build/`. After building the DOCX/PDF artifacts, run:

```bash
python tools/check_submission_readiness.py
```

The checker validates author metadata, page count, file size, expected labels, equation numbering, absence of arXiv reference text, and absence of known draft-note phrases.

## Current Evidence Boundary

The repository supports a research prototype, not a production charger controller. The reported results support feasibility, service, and communication-efficiency claims under the tested synthetic, public-feeder, and public-session scenarios. They do not establish universal optimality, formal differential privacy, or universal charging-cost dominance.

## Citation

If you use this code, cite the accompanying paper draft and this repository. A `CITATION.cff` file is included for citation managers.
