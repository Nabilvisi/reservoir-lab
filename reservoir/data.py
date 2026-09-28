from pathlib import Path
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load_rock():
    path = ROOT / "data/spe10_subset.npz"
    metadata = json.loads((ROOT / "data/provenance.json").read_text())
    if hashlib.sha256(path.read_bytes()).hexdigest() != metadata["derived_sha256"]:
        raise ValueError("Dataset checksum mismatch. Restore or re-download the dataset.")
    with np.load(path, allow_pickle=False) as archive:
        rock = {key: archive[key].copy() for key in archive.files}
    for key, values in rock.items():
        if values.shape != tuple(metadata["shape_zyx"]) or not np.isfinite(values).all():
            raise ValueError(f"Invalid data in {key}")
    rock["spacing"] = np.asarray(metadata["cell_size_m_xyz"])[::-1]
    return rock, metadata


def simulation_grid(rock):
    """2x2x2 block averages; geometric permeability is only an approximation."""
    result = {}
    nz, ny, nx = rock["porosity"].shape
    for key in ("porosity", "permx", "permy", "permz"):
        a = rock[key]
        if key != "porosity":
            a = np.log(a)
        a = a.reshape(nz // 2, 2, ny // 2, 2, nx // 2, 2).mean(axis=(1, 3, 5))
        result[key] = np.exp(a) if key != "porosity" else a
    # An explicit numerical assumption, never applied to the original data.
    result["porosity"] = np.maximum(result["porosity"], 0.02)
    result["spacing"] = rock["spacing"] * 2
    return result


def load_training():
    import pandas as pd
    path = ROOT / 'data/training.csv'
    provenance = json.loads((ROOT / 'data/training_provenance.json').read_text())
    if hashlib.sha256(path.read_bytes()).hexdigest() != provenance['training_sha256']:
        raise ValueError('Training data checksum mismatch. Rebuild the training dataset.')
    frame = pd.read_csv(path)
    if len(frame) != provenance['scenarios'] * provenance['snapshots_per_scenario'] or not np.isfinite(frame).all().all():
        raise ValueError('Invalid training data dimensions or non-finite values')
    return frame
