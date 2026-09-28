# Reservoir Lab

A local Streamlit web app for exploring open reservoir geology, simplified oil–gas–water flow, supervised ML/deep-learning surrogates, and a separate reinforcement-learning teaching experiment.

## Start the app

The prepared Windows environment in this workspace can be started with:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open http://localhost:8501. Alternatively run `launch.ps1` from PowerShell.

On a fresh machine with Python 3.12:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Linux/macOS use `.venv/bin/python` in place of `.venv\Scripts\python.exe`. The bundled derived rock and training data allow normal app use without downloading external data or supplying API keys. Initial model training happens locally and is cached in memory. The exact development environment is recorded in `requirements-lock.txt`; `requirements.txt` pins the direct dependencies.

## Workspaces

1. **3D reservoir** — original OPM SPE10 cell properties, directional permeability, porosity, layer slices, rotation, hover and vertical exaggeration.
2. **Dynamic modelling** — rate-controlled injection, gas/water injection mixture, oil/water/gas saturation snapshots, pressure, phase production, recovery and mass balance. Download CSV history, NumPy states, input settings and provenance together.
3. **ML / Deep learning** — random forest and three-hidden-layer MLP, mean baseline, held-out scenario metrics, bounded prediction inputs, raw physical-consistency diagnostics and a direct simulator comparison.
4. **RL experiment** — seeded Q-learning in a separate perfectly mixed oil–water tank, compared with fixed-rate policies. This is not a three-phase field controller.
5. **Data & methods** — source, licences, assumptions, numerical methods and limitations.

## Reproduce data and checks

```powershell
.\.venv\Scripts\python.exe scripts/fetch_data.py
.\.venv\Scripts\python.exe scripts/build_training.py
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts/validate.py
```

The downloader uses a pinned OPM commit once a provenance file exists. Original permeability/porosity inputs total about 77 MB and are cached under `data/raw`; normal startup needs only the small derived subset. Training generation runs 32 independent three-phase simulations and may take several minutes. Tests do not require re-downloading the data.

## Project layout

- `app.py` — Streamlit interface
- `reservoir/data.py` — checksum validation and explicit grid reduction
- `reservoir/physics.py` — sparse pressure solver and conservative transport
- `reservoir/learning.py` — grouped training, evaluation and prediction bounds
- `reservoir/control.py` — separate reduced RL environment
- `data/` — attributed open geology, generated targets and provenance
- `tests/` — numerical, learning and app checks
- `METHODS.md` — scientific assumptions and validity boundaries
- `VALIDATION.md` — test evidence and observed model results

## Scope and licensing

This is a research demonstrator, not a field-qualified reservoir simulator. Gas is incompressible in the numerical model; black-oil PVT and phase changes are not implemented. Production histories are generated locally, not observed field data. See `METHODS.md` before interpreting results.

Original OPM deck: copyright Statoil (2015), ODbL 1.0; individual contents DbCL 1.0. The derived geological subset is made available under the same database/content terms, with source provenance and the method for recreating it. Licence texts are in `data/raw/odbl-10.txt` and `data/raw/dbcl-10.txt`. Application code is MIT licensed; this does not replace the data licences.

The app is configured for local use. Public hosting has not been provisioned. A Streamlit-compatible Python host can run `streamlit run app.py`; include the bundled `data` files and install `requirements.txt`.
