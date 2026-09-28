"""Write observed numerical and surrogate metrics without changing model choices."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datetime import datetime, timezone
import hashlib
import json
import numpy as np
import pandas as pd
from reservoir.data import ROOT, load_rock, simulation_grid
from reservoir.physics import FlowModel, Scenario
from reservoir.learning import fit_models
from reservoir.control import learn_policy


def main():
    rock, source = load_rock()
    model = FlowModel(simulation_grid(rock))
    result = model.run(Scenario())
    bundle = fit_models(pd.read_csv(ROOT / 'data/training.csv'))
    policies, _ = learn_policy(result['pore_volume_m3'])
    history = result['history']
    metrics = {'verified_at_utc': datetime.now(timezone.utc).isoformat(),
        'source_commit': source['commit'], 'default_scenario': {
            'pore_volume_m3': float(result['pore_volume_m3']), 'transport_steps': result['steps'],
            'phase_balance_error_m3': float(history.balance_error_m3.max()),
            'saturation_closure_error': float(np.abs(result['saturation'].sum(axis=-1)-1).max()),
            'pressure_residual_m3_day': float(result['pressure_residual']),
            'final_oil_recovery': float(history.iloc[-1].oil_recovery)},
        'training_sha256': hashlib.sha256((ROOT / 'data/training.csv').read_bytes()).hexdigest(),
        'model_scores': bundle['scores'].to_dict(orient='records'),
        'model_warnings': bundle['warnings'], 'train_ids': bundle['train_ids'], 'test_ids': bundle['test_ids'],
        'rl_rewards': policies.groupby('Policy').reward.sum().to_dict()}
    (ROOT / 'data/validation.json').write_text(json.dumps(metrics, indent=2))
    lines = ['# Validation evidence', '', 'Generated: ' + metrics['verified_at_utc'], '',
             'These are software checks and simulator-surrogate metrics, not physical or field validation.', '',
             '## Default 180-day simulation', '']
    lines.extend(f'- {k}: {v}' for k, v in metrics['default_scenario'].items())
    lines += ['', '## Held-out whole-scenario evaluation', '',
              '24 training scenarios; 8 test scenarios. All targets are locally simulated. MAE is averaged across oil, gas and water.', '',
              '| Model | MAE (m³/day) | R² |', '|---|---:|---:|']
    lines.extend(f"| {row['Model']} | {row['MAE (m3/day)']:.6f} | {row['R2']:.6f} |" for row in metrics['model_scores'])
    lines += ['', 'The random forest has negative mean R² on this split, despite beating the global-mean baseline on MAE. '
              'It is not presented as a high-quality forecaster. The MLP also has nonzero error and violates exact phase-rate closure.',
              '', '## RL evaluation', '', 'Reward units are illustrative utility units in the same training tank environment.', '']
    lines.extend(f'- {k}: {v:.3f}' for k, v in metrics['rl_rewards'].items())
    lines += ['', '## Automated checks', '', 'Run `.venv/Scripts/python.exe -m pytest -q` for the current numerical, ML, RL and Streamlit interaction checks.',
              'UI rendering is also inspected separately in a running local browser.', '',
              '## Remaining limitations', '', 'See METHODS.md. No external simulator cross-validation, grid-convergence qualification, real production history matching, '
              'field validation, calibrated prediction intervals or public-host deployment has been completed.']
    (ROOT / 'VALIDATION.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
