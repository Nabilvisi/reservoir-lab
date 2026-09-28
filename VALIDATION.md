# Validation evidence

Generated: 2026-09-28T04:26:57.108663+00:00

These are software checks and simulator-surrogate metrics, not physical or field validation.

## Default 180-day simulation

- pore_volume_m3: 6029.690357248379
- transport_steps: 1298
- phase_balance_error_m3: 6.330935775622493e-12
- saturation_closure_error: 3.852473895449293e-14
- pressure_residual_m3_day: 3.765876499528531e-13
- final_oil_recovery: 0.351010690129604

## Held-out whole-scenario evaluation

24 training scenarios; 8 test scenarios. All targets are locally simulated. MAE is averaged across oil, gas and water.

| Model | MAE (m³/day) | R² |
|---|---:|---:|
| Mean baseline | 3.690319 | -1.217960 |
| Random forest | 1.764872 | -0.321334 |
| Deep MLP (64 / 64 / 32) | 0.550732 | 0.920142 |

The random forest has negative mean R² on this split, despite beating the global-mean baseline on MAE. It is not presented as a high-quality forecaster. The MLP also has nonzero error and violates exact phase-rate closure.

## RL evaluation

Reward units are illustrative utility units in the same training tank environment.

- Fixed 10: 676.520
- Fixed 20: 158.763
- Fixed 30: -1016.366
- Q-learning: 676.520
- Shut in: 0.000

## Automated checks

On 2026-09-28, `.venv/Scripts/python.exe -m pytest -q` passed all 14 tests in 18.53 seconds. After the final export-metadata and chart-label changes, the full Streamlit page-interaction test passed again in 13.19 seconds.
The numerical checks include an analytical homogeneous pressure case, phase conservation and bounds, zero flow and invalid inputs. Learning checks cover whole-scenario separation, baseline comparisons and prediction-range rejection. RL checks cover seeded reproducibility and tank balance.
UI rendering was separately inspected in the running local browser, including the original 3D rock view and default dynamic simulation. The browser displayed 35.1% oil recovery and a 6.3e-12 m³ phase-balance error for the default simulation. A screenshot is saved in `artifacts/reservoir-preview.png`.

A further boundary-input run used the maximum UI rate (30 m³/day), maximum duration (360 days), pure gas injection and zero initial gas. It completed 5,838 transport steps, with maximum phase-balance error 2.27e-11 m³ and saturations between 0 and 0.90334. The local Streamlit health endpoint returned `ok`.

## Remaining limitations

See METHODS.md. No external simulator cross-validation, grid-convergence qualification, real production history matching, field validation, calibrated prediction intervals or public-host deployment has been completed.
