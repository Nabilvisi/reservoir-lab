"""Download pinned OPM SPE10 rock data and create a small, attributed teaching grid."""
from pathlib import Path
import hashlib
import json
import re
import urllib.request
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def download(url):
    with urllib.request.urlopen(url, timeout=180) as response:
        return response.read()


def parse(text, keyword):
    clean = re.sub(r"--[^\n]*", "", text)
    block = re.search(r"\b" + keyword + r"\b(.*?)/", clean, re.S)
    if block is None:
        raise ValueError(f"Missing {keyword}")
    values = []
    for token in block.group(1).split():
        if "*" in token:
            count, value = token.split("*")
            values.extend([float(value)] * int(count))
        else:
            values.append(float(token.replace("D", "E")))
    arr = np.asarray(values, dtype=np.float64)
    if arr.size != 60 * 220 * 85 or not np.isfinite(arr).all():
        raise ValueError(f"Invalid {keyword} count or values: {arr.size}")
    return arr.reshape(85, 220, 60)


def main():
    DATA.mkdir(exist_ok=True)
    raw = DATA / "raw"
    raw.mkdir(exist_ok=True)
    manifest_path = DATA / "provenance.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        commit = existing["commit"]
    else:
        existing = {}
        commit = json.loads(download("https://api.github.com/repos/OPM/opm-data/commits/master"))["sha"]
    base = f"https://raw.githubusercontent.com/OPM/opm-data/{commit}/"
    files = ["spe10model2/SPE10MODEL2_PERM.INC", "spe10model2/SPE10MODEL2_PHI.INC",
             "spe10model2/SPE10_MODEL2.DATA", "odbl-10.txt", "dbcl-10.txt", "README.md"]
    sources = []
    for name in files:
        path = raw / Path(name).name
        if not path.exists():
            print("Downloading", name, flush=True)
            payload = download(base + name)
            path.write_bytes(payload)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        previous = next((s for s in existing.get('sources', []) if s['file'] == name), None)
        if previous and digest != previous['sha256']:
            raise ValueError(f'Source checksum mismatch for {name}; restore the pinned source before proceeding.')
        sources.append({"file": name, "url": base + name, "sha256": digest})
    perm_text = (raw / "SPE10MODEL2_PERM.INC").read_text()
    poro = parse((raw / "SPE10MODEL2_PHI.INC").read_text(), "PORO")
    # Contiguous subset preserves original rock properties, without upscaling.
    # zero-based x=0:18, y=0:30, z=0:6; source arrays use x fastest.
    selection = np.s_[0:6, 0:30, 0:18]
    arrays = {key.lower(): parse(perm_text, key)[selection] for key in ["PERMX", "PERMY", "PERMZ"]}
    arrays["porosity"] = poro[selection]
    if not (arrays["permx"] > 0).all() or not ((arrays["porosity"] >= 0) & (arrays["porosity"] <= 1)).all():
        raise ValueError("Rock properties outside valid range")
    np.savez_compressed(DATA / "spe10_subset.npz", **arrays)
    manifest = {"dataset": "OPM SPE10 Model 2: contiguous 18 x 30 x 6 subset",
                "commit": commit, "repository": "https://github.com/OPM/opm-data",
                "database_license": "ODbL-1.0", "content_license": "DbCL-1.0",
                "sources": sources, "shape_zyx": [6, 30, 18], "cell_size_m_xyz": [6.096, 3.048, 0.6096],
                "selection_zero_based": {"x": [0, 18], "y": [0, 30], "z": [0, 6]},
                "derived_sha256": hashlib.sha256((DATA / "spe10_subset.npz").read_bytes()).hexdigest(),
                "notes": "Original rock values retained. Cropped domain; new teaching wells and fluid assumptions. Not the full SPE10 benchmark simulation."}
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print("Saved", arrays["porosity"].shape, "cells; porosity range", poro[selection].min(), poro[selection].max())


if __name__ == "__main__":
    main()
