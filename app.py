from dataclasses import asdict
import io
import hashlib
import json
import zipfile
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from reservoir.data import ROOT, load_rock, simulation_grid, load_training
from reservoir.physics import FlowModel, Scenario
from reservoir.charts import volume, production
from reservoir.learning import FEATURES, TARGETS, fit_models, predict
from reservoir.control import learn_policy

st.set_page_config(page_title="Reservoir Lab · Oil / Gas / Water", page_icon="🧊", layout="wide")
st.markdown("""<style>
.block-container {padding-top:4rem; max-width:1550px}
h1 {font-weight:650!important; letter-spacing:-1.4px}
[data-testid="stMetric"] {background:#131f31;padding:18px;border-radius:10px;border:1px solid #25354c}
[data-testid="stSidebar"] {border-right:1px solid #25354c}
.eyebrow {color:#33d6c5;letter-spacing:2px;font-size:13px;font-weight:600}
</style>""", unsafe_allow_html=True)


@st.cache_data
def rocks():
    return load_rock()


@st.cache_data(max_entries=8)
def simulate(parameters):
    return FlowModel(simulation_grid(rocks()[0])).run(Scenario(**parameters))


@st.cache_resource
def trained():
    return fit_models(load_training())


@st.cache_data
def control(pv):
    return learn_policy(pv)


try:
    rock, metadata = rocks()
    grid = simulation_grid(rock)
except (OSError, ValueError, KeyError) as exc:
    st.error(f"Dataset could not be loaded: {exc}")
    st.code("python scripts/fetch_data.py")
    st.stop()

with st.sidebar:
    st.markdown("### ◈ RESERVOIR LAB")
    st.caption("OIL · GAS · WATER")
    page = st.radio("Workspace", ["3D reservoir", "Dynamic modelling", "ML / Deep learning", "RL experiment", "Data & methods"], label_visibility="collapsed")
    st.divider()
    st.markdown("**SPE10 · Tarbert subset**")
    st.caption("3,240 source cells · 405 simulation cells\n\nOpen geological benchmark · local computation")
    st.info("Research demonstrator. Simplified three-phase physics; no field validation.")
    st.caption("All flow volumes are at reservoir conditions. Gas is approximated as incompressible.")

st.markdown('<div class="eyebrow">SUBSURFACE MODELLING WORKSPACE</div>', unsafe_allow_html=True)
st.title(page)

if page == "3D reservoir":
    st.write("Explore the original open rock properties. Rotate the model, inspect individual cells, and move through geological layers.")
    a, b, c, d = st.columns(4)
    a.metric("Source cells", f"{rock['porosity'].size:,}")
    b.metric("Mean porosity", f"{rock['porosity'].mean():.1%}")
    c.metric("Median Kx", f"{np.median(rock['permx']):.1f} mD")
    d.metric("Domain thickness", f"{rock['spacing'][0]*6:.2f} m")
    left, right = st.columns([3, 1])
    with right:
        field = st.selectbox("Rock property", ["Permeability X", "Permeability Y", "Permeability Z", "Porosity"])
        exaggeration = st.slider("Vertical exaggeration", 1, 20, 8)
        layer = st.slider("Source layer", 1, 6, 3)
        st.caption("Depth is relative to the cropped model top. Points represent cell centres; thickness is visually exaggerated.")
        st.caption("Rock values are unchanged from the OPM source. This view does not display measured field production.")
    key = {"Permeability X": "permx", "Permeability Y": "permy", "Permeability Z": "permz", "Porosity": "porosity"}[field]
    values = rock[key] if key == "porosity" else np.log10(rock[key])
    label = "Porosity" if key == "porosity" else "log10(mD)"
    with left:
        st.plotly_chart(volume(values, rock["spacing"], label, exaggeration), width="stretch")
    with st.expander("Layer cross-section", expanded=True):
        dz, dy, dx = rock["spacing"]
        fig = px.imshow(values[layer - 1], x=(np.arange(18)+.5)*dx, y=(np.arange(30)+.5)*dy,
                        color_continuous_scale="Viridis", labels={"x": "X (m)", "y": "Y (m)", "color": label}, aspect="equal")
        fig.update_layout(height=330, margin=dict(t=15, b=15))
        st.plotly_chart(fig, width="stretch")

elif page == "Dynamic modelling":
    st.write("Run a balanced injector–producer experiment and inspect the evolving phase saturations.")
    with st.form("simulation"):
        a, b, c, d = st.columns(4)
        rate = a.slider("Injection rate (m³/day)", 0., 30., 15., 1.)
        gas = b.slider("Gas fraction in injection", 0., 1., 0., .05)
        days = c.slider("Duration (days)", 30, 360, 180, 30)
        initial_gas = d.slider("Initial gas saturation", 0., .2, .1, .01)
        submitted = st.form_submit_button("Run simulation", type="primary")
    if "scenario" not in st.session_state:
        st.session_state.scenario = asdict(Scenario())
    if submitted:
        st.session_state.scenario = asdict(Scenario(rate=rate, gas_fraction=gas, days=days, initial_gas=initial_gas))
    parameters = st.session_state.scenario
    st.caption(f"Displayed run: {parameters['rate']:g} m³/day · {parameters['gas_fraction']:.0%} injected gas · {parameters['days']:g} days · initial gas {parameters['initial_gas']:.0%}. Submit changes to update results.")
    try:
        with st.spinner("Solving pressure and conservative phase transport…"):
            result = simulate(parameters)
    except (RuntimeError, ValueError) as exc:
        st.error(str(exc))
        st.stop()
    hist = result["history"]
    a, b, c, d = st.columns(4)
    a.metric("Oil recovery", f"{hist.iloc[-1].oil_recovery:.1%}")
    b.metric("Produced oil", f"{hist.iloc[-1].cumulative_oil_m3:,.0f} m³")
    c.metric("Final water cut", f"{hist.iloc[-1].water_cut:.1%}")
    d.metric("Phase balance error", f"{hist.balance_error_m3.max():.0e} m³")
    left, right = st.columns([3, 1])
    with right:
        index = st.slider("Snapshot", 0, len(hist)-1, len(hist)-1, format="%d")
        st.write(f"**Day {hist.day.iloc[index]:.0f}**")
        property_name = st.selectbox("Simulated property", ["Water saturation", "Oil saturation", "Gas saturation", "Pressure"])
        st.caption("405 cells · approximate geometric permeability averaging · simulation porosity floor 0.02.")
        st.caption("Pressure reference: 200 bar in one producer cell. No gravity, PVT or dissolved gas. These are model assumptions.")
    with left:
        vals = result["pressure"][index] if property_name == "Pressure" else result["saturation"][index, ..., ["Water saturation", "Oil saturation", "Gas saturation"].index(property_name)]
        st.plotly_chart(volume(vals, grid["spacing"], "bar" if property_name == "Pressure" else "Saturation", wells=True), width="stretch")
    st.plotly_chart(production(hist), width="stretch")
    if hist.max_pressure_bar.max() > 500:
        st.warning("The rate-controlled toy model requires pressure above 500 bar. There is no fracture-pressure constraint; this scenario is not an operational recommendation.")
    package = io.BytesIO()
    with zipfile.ZipFile(package, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("history.csv", hist.to_csv(index=False))
        z.writestr("scenario.json", json.dumps(parameters, indent=2))
        z.writestr("provenance.json", json.dumps(metadata, indent=2))
        arrays = io.BytesIO()
        np.savez_compressed(arrays, saturation=result["saturation"], pressure_bar=result["pressure"], days=hist.day.values,
                            spacing_m_zyx=grid['spacing'], porosity=grid['porosity'],
                            permx_md=grid['permx'], permy_md=grid['permy'], permz_md=grid['permz'])
        z.writestr("states.npz", arrays.getvalue())
        z.writestr("run_metadata.json", json.dumps({'origin': 'local simplified simulation',
            'phase_order': ['water', 'oil', 'gas'], 'state_array_order': ['snapshot', 'z', 'y', 'x', 'phase'],
            'volumes': 'reservoir m3', 'pressure': 'bar', 'time': 'days',
            'solver_sha256': hashlib.sha256((ROOT / 'reservoir/physics.py').read_bytes()).hexdigest()}, indent=2))
        z.writestr("METHODS.md", (ROOT / "METHODS.md").read_text())
    st.download_button("Download run, 3D states & provenance", package.getvalue(), "reservoir_run.zip", "application/zip")

elif page == "ML / Deep learning":
    st.write("Compare a random forest and a three-hidden-layer neural network against a mean baseline.")
    st.info("Targets are generated by this app’s simplified simulator. Test scores measure simulator emulation, not accuracy on a real reservoir. All models use one fixed geology.")
    if not (ROOT / "data/training.csv").exists():
        st.error("Training dataset is missing. Run python scripts/build_training.py.")
        st.stop()
    try:
        with st.spinner("Fitting reproducible models and evaluating held-out scenarios…"):
            bundle = trained()
    except (ValueError, OSError, KeyError) as exc:
        st.error(f"Training data could not be used: {exc}")
        st.stop()
    a, b, c = st.columns(3)
    a.metric("Training scenarios", len(bundle["train_ids"]))
    b.metric("Held-out scenarios", len(bundle["test_ids"]))
    c.metric("Prediction targets", "Oil / Gas / Water")
    st.dataframe(bundle["scores"], hide_index=True, width="stretch")
    st.caption("MAE is in reservoir m³/day; lower is better. R² can be negative. The baseline is a training-set mean. Test scenarios are never used to fit scaling or model weights.")
    for warning in bundle["warnings"]:
        st.warning(warning)
    model_name = st.selectbox("Prediction model", list(bundle["models"])[1:] )
    bounds = bundle["bounds"]
    a, b, c = st.columns(3)
    rate = a.slider("Prediction injection (m³/day)", *bounds["rate"], float(np.mean(bounds["rate"])))
    gas = b.slider("Prediction injected gas fraction", *bounds["gas_fraction"], float(np.mean(bounds["gas_fraction"])))
    initial = c.slider("Prediction initial gas saturation", *bounds["initial_gas"], float(np.mean(bounds["initial_gas"])))
    inputs = pd.DataFrame({"day": np.linspace(0, 180, 31), "rate": rate, "gas_fraction": gas, "initial_gas": initial})
    output = predict(bundle["models"][model_name], inputs, bounds)
    forecast = pd.DataFrame(output, columns=TARGETS)
    forecast["day"] = inputs.day
    st.plotly_chart(production(forecast), width="stretch")
    closure = float(np.abs(output.sum(axis=1)-rate).max())
    st.caption(f"Raw predictions, without clipping. Maximum phase-rate closure error: {closure:.3f} m³/day. Predictions are not constrained by conservation; simulator results should be used to check them.")
    if (output < 0).any():
        st.warning("This prediction contains a negative phase rate and is physically invalid. Use the dynamic simulator for this scenario.")
    if st.button("Compare prediction with simulator"):
        with st.spinner("Running matching scenario…"):
            truth = simulate(asdict(Scenario(rate=rate, gas_fraction=gas, initial_gas=initial)))["history"]
        comparison = pd.DataFrame({"day": inputs.day, "Predicted oil": output[:, 1], "Simulated oil": truth.oil_m3_day})
        st.line_chart(comparison.set_index("day"), y_label="Oil (m³/day)")
        st.write(f"Oil MAE for this scenario: {np.abs(output[:, 1]-truth.oil_m3_day).mean():.3f} m³/day")
    with st.expander("Held-out predictions and split audit"):
        test = bundle["test"]
        predicted = bundle["predictions"][model_name]
        fig = px.scatter(x=test.oil_m3_day, y=predicted[:, 1], color=test.scenario_id.astype(str),
                         labels={"x": "Simulated oil (m³/day)", "y": "Predicted oil (m³/day)", "color": "Held-out scenario"})
        upper = float(max(test.oil_m3_day.max(), predicted[:, 1].max()))
        fig.add_trace(go.Scatter(x=[0, upper], y=[0, upper], mode="lines", name="Perfect agreement", line=dict(dash="dash")))
        st.plotly_chart(fig, width="stretch")
        st.json({"train_scenario_ids": bundle["train_ids"], "test_scenario_ids": bundle["test_ids"], "input_bounds": bounds})
    st.download_button("Download predicted phase rates", forecast.to_csv(index=False), "predictions.csv", "text/csv")

elif page == "RL experiment":
    st.write("Train a Q-learning policy to select monthly water-injection rates in a reduced reservoir tank.")
    st.warning("Separate teaching environment: a perfectly mixed oil–water tank with equal mobility, no gas, and pore volume taken from the derived grid. This policy has not been validated in the 3D simulator and cannot control equipment.")
    pv = float(np.sum(grid["porosity"]) * np.prod(grid["spacing"]))
    with st.spinner("Training 4,000 seeded episodes…"):
        policies, rewards = control(pv)
    a, b, c = st.columns(3)
    a.metric("Episodes", "4,000")
    b.metric("Decision interval", "30 days")
    c.metric("Actions", "0 / 10 / 20 / 30")
    st.caption("Actions in m³/day. Reward = produced oil − 0.4 × produced water − 0.15 × injected volume. Coefficients are illustrative utility weights, not prices or NPV. No pressure or facility limits are modelled.")
    totals = policies.groupby("Policy", sort=False).agg(Reward=("reward", "sum"), Oil_m3=("oil_m3", "sum"), Water_m3=("water_m3", "sum"))
    st.dataframe(totals, width="stretch")
    learned = totals.loc["Q-learning", "Reward"]
    best_fixed = totals.drop(index="Q-learning").Reward.max()
    st.write(f"Learned policy reward relative to the best fixed policy: **{learned-best_fixed:+.2f}**. A negative result means Q-learning underperformed that baseline.")
    fig = px.line(policies, x="day", y="rate_m3_day", color="Policy", line_shape="hv", labels={"day": "Day", "rate_m3_day": "Injection (m³/day)"})
    st.plotly_chart(fig, width="stretch")
    st.line_chart(rewards.set_index("episode").rolling(100).mean().dropna(), y_label="Training reward · 100-episode mean")
    st.caption("Evaluation uses a greedy policy with exploration off, in the same deterministic tank used for training. This is not out-of-environment validation or evidence of an optimal policy.")
    st.download_button("Download policy comparison", policies.to_csv(index=False), "rl_policies.csv", "text/csv")

else:
    st.markdown((ROOT / "METHODS.md").read_text())
    with st.expander("Exact source versions and checksums"):
        st.json(metadata)
    st.download_button("Download source provenance", json.dumps(metadata, indent=2), "provenance.json", "application/json")
    st.download_button("Download open rock subset", (ROOT / "data/spe10_subset.npz").read_bytes(), "spe10_subset.npz", "application/octet-stream")

st.divider()
st.caption("Reservoir Lab · OPM SPE10 open geological data · Simulation and AI outputs are explicitly model-derived · Research use")
