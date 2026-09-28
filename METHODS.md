### What the data represents

The geological input is **OPM SPE10 Model 2**, a synthetic reservoir benchmark. It is not an observed production dataset. The source deck credits Statoil (2015) and the SPE comparative study by Christie and Blunt (2001).

- [OPM source repository and licensing](https://github.com/OPM/opm-data)
- [SPE10 geometry and units documentation](https://www.sintef.no/contentassets/2551f5f85547478590ceca14bc13ad51/spe10.html)
- Source database: ODbL 1.0. Individual contents: DbCL 1.0. Licence texts are included under `data/raw`.
- The distributed subset retains the original porosity and directional permeability values at zero-based x=0:18, y=0:30, z=0:6. Array order is z, y, x; x varies fastest in the original deck.
- Original cell dimensions are 20 × 10 × 2 ft, converted to 6.096 × 3.048 × 0.6096 m. Permeability is in mD. Depth is relative to the cropped top; no absolute depth or well survey is implied.
- Pinned repository commit, source URLs, source SHA-256 values, and the subset checksum are included in the provenance file. The app checks the subset checksum when it loads.

### Dynamic model

The original 18 × 30 × 6 subset is reduced to a **9 × 15 × 3 simulation grid**. Porosity uses arithmetic block averages; directional permeability uses geometric block averages. This is an approximate reduction, not a qualified flow-based upscaling method. A porosity floor of 0.02 applies only to the simulation grid. These changes can alter storage and flow behaviour.

The solver uses a Cartesian two-point finite-volume pressure discretization and first-order upwind explicit phase transport. Outer boundaries are sealed. An injector and producer occupy opposite x/y corners, with equal prescribed rate per layer. Total production equals injection. One producer cell fixes the pressure reference to 200 bar; this is a mathematical datum, not a wellbore-pressure calculation.

For phase α, relative permeability is assumed to be Sα². Assumed constant viscosities are water 1 cP, oil 3 cP, and gas 0.08 cP. Total mobility is the sum of phase mobilities, and fractional flow is each phase mobility divided by total mobility. Harmonic directional permeability and harmonic total mobility define the face transmissibility. Pressure and mobility are recomputed at each transport substep. The timestep is limited by outgoing phase flux and available phase volume. Saturations are not clipped to hide numerical errors.

Initial water saturation is 0.2; initial gas saturation is user-selected; oil fills the remainder. The injector supplies a chosen mixture of water and gas. Phase inventory changes are checked against cumulative injection and production. Reported volumes and rates are **reservoir volumes**, including gas; they are not stock-tank volumes or standard-condition gas rates.

This is an **incompressible, immiscible teaching model**, including an incompressible approximation for gas. It omits PVT, dissolution, vaporization, gravity, capillarity, residual saturations, calibrated relative permeability, fractures, geomechanics, well indices, facility constraints and history matching. It is not a black-oil or compositional simulator, and it does not reproduce the full SPE10 benchmark. Software conservation tests do not establish physical accuracy or suitability for field decisions.

### Machine learning and deep learning

Thirty-two seeded simulation scenarios vary injection rate, injection gas fraction and initial gas saturation on one fixed geological grid. Each has 25 snapshots from day 0 to day 180. Targets are simulated water, oil and gas production rates. Features are time and the three scenario inputs.

A fixed group split assigns 24 entire scenarios to training and 8 to testing. No scenario appears in both groups. The mean baseline, random forest and three-hidden-layer MLP (64, 64, 32 units) are evaluated on the same held-out scenarios. Feature and target standardization for the neural network is fitted only on the training set. Hyperparameters are fixed, not selected against the displayed test set. Scores include MAE, R² and phase-rate closure error. A convergence warning is shown if neural-network optimization does not converge.

These are surrogate-model predictions, not field forecasts. Input ranges are bounded to the training range, but a bounding box does not guarantee local training support. There is no validation on a new geological realization, no long-horizon extrapolation claim, and no calibrated uncertainty interval. Raw predictions may violate non-negativity or conservation; the app reports these limitations and lets users compare a prediction with the simulator.

### Reinforcement learning

The RL experiment is intentionally a **separate oil–water tank model**, with equal mobility and perfect mixing. It inherits only the derived grid's total pore volume. It contains no gas and no spatial geology. Oil fraction evolves exactly as So(next) = So × exp(−q Δt / PV) under pure water injection. Initial oil fraction is 0.7; the rest is water.

Tabular Q-learning chooses 0, 10, 20 or 30 m³/day every 30 days for 12 periods. It uses 4,000 episodes, a fixed seed, discretized oil-fraction states, learning rate 0.12 and undiscounted finite-horizon return. Reward is oil volume minus 0.4 times produced water volume minus 0.15 times injected volume. These are illustrative weights, not financial valuation.

The greedy learned policy is compared with shut-in and fixed-rate baselines in the same deterministic environment. This evaluates learning behaviour, not transfer to the 3D model. State discretization and finite training can produce a policy worse than a fixed baseline. There is no optimality guarantee, no field-control interface, and no operational recommendation.

### Verification and remaining validation

Automated tests cover an analytical homogeneous pressure-gradient case, phase conservation, saturation closure and bounds, zero-flow behaviour, invalid inputs, data checksums and dimensions, grouped learning splits, baseline scores, reproducible RL, and Streamlit page interactions. A validation report records results from the current run.

Still required for engineering use: comparison against an established multiphase simulator, timestep and grid convergence studies, calibrated rock/fluid inputs, well-model validation, history matching, independent reservoir-engineering review and appropriate uncertainty analysis.
