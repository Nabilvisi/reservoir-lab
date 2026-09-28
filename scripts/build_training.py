"""Generate reproducible simulation targets from the bundled open geological data."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
import hashlib
import time
import numpy as np
import pandas as pd
from reservoir.data import load_rock, simulation_grid, ROOT
from reservoir.physics import FlowModel, Scenario


def main():
    rock, metadata = load_rock()
    model = FlowModel(simulation_grid(rock))
    rng = np.random.default_rng(42)
    rows = []
    start = time.time()
    # Stratified ranges plus fixed endpoints; independent scenario holdout in learning.py.
    for i in range(32):
        scenario = Scenario(rate=float(rng.uniform(5, 30)), gas_fraction=float(rng.uniform(0, 1)),
                            initial_gas=float(rng.uniform(0, 0.2)), days=180)
        result = model.run(scenario, snapshots=25)
        frame = result["history"]
        frame["rate"], frame["gas_fraction"], frame["initial_gas"] = scenario.rate, scenario.gas_fraction, scenario.initial_gas
        frame["scenario_id"] = i
        rows.append(frame)
        print(f"Scenario {i+1}/32, {result['steps']} steps, {time.time()-start:.1f}s", flush=True)
    training = pd.concat(rows, ignore_index=True)
    training.to_csv(ROOT / "data/training.csv", index=False)
    (ROOT / "data/training_provenance.json").write_text(json.dumps({
        "target_source": "Local simplified three-phase finite-volume simulator, NOT observed production",
        "rock_sha256": metadata["derived_sha256"], "seed": 42, "scenarios": 32, "snapshots_per_scenario": 25,
        "training_sha256": hashlib.sha256((ROOT / 'data/training.csv').read_bytes()).hexdigest(),
        "physics_sha256": hashlib.sha256((ROOT / 'reservoir/physics.py').read_bytes()).hexdigest(),
        "grid_code_sha256": hashlib.sha256((ROOT / 'reservoir/data.py').read_bytes()).hexdigest(),
        "simulation_grid_zyx": list(model.shape), "max_balance_error_m3": float(training.balance_error_m3.max())}, indent=2))


if __name__ == "__main__":
    main()
