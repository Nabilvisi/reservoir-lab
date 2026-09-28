"""Grouped evaluation of simulator surrogates; no field-data claims."""
import warnings
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = ["day", "rate", "gas_fraction", "initial_gas"]
TARGETS = ["water_m3_day", "oil_m3_day", "gas_m3_day"]


def fit_models(frame):
    split = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    train, test = next(split.split(frame, groups=frame.scenario_id))
    x, y = frame[FEATURES], frame[TARGETS]
    models = {
        "Mean baseline": DummyRegressor(),
        "Random forest": RandomForestRegressor(n_estimators=160, min_samples_leaf=2, random_state=42, n_jobs=2),
        "Deep MLP (64 / 64 / 32)": TransformedTargetRegressor(
            regressor=make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(64, 64, 32),
                max_iter=1500, random_state=42, early_stopping=False, learning_rate_init=0.001)),
            transformer=StandardScaler()),
    }
    scores, predictions, diagnostics = [], {}, []
    for name, model in models.items():
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(x.iloc[train], y.iloc[train])
        diagnostics.extend(f"{name}: {str(w.message)}" for w in caught)
        predicted = model.predict(x.iloc[test])
        predictions[name] = predicted
        scores.append({"Model": name, "MAE (m3/day)": mean_absolute_error(y.iloc[test], predicted),
                       "R2": r2_score(y.iloc[test], predicted),
                       "Water MAE": mean_absolute_error(y.iloc[test, 0], predicted[:, 0]),
                       "Oil MAE": mean_absolute_error(y.iloc[test, 1], predicted[:, 1]),
                       "Gas MAE": mean_absolute_error(y.iloc[test, 2], predicted[:, 2]),
                       "Max rate closure error": float(np.max(np.abs(predicted.sum(axis=1) - x.iloc[test].rate)))})
    return {"models": models, "scores": pd.DataFrame(scores), "predictions": predictions,
            "train_ids": sorted(frame.iloc[train].scenario_id.unique().tolist()),
            "test_ids": sorted(frame.iloc[test].scenario_id.unique().tolist()),
            "test": frame.iloc[test].copy(), "warnings": diagnostics,
            "bounds": {c: [float(x.iloc[train][c].min()), float(x.iloc[train][c].max())] for c in FEATURES}}


def predict(model, inputs, bounds):
    for feature in FEATURES:
        lo, hi = bounds[feature]
        if not np.isfinite(inputs[feature]).all() or not inputs[feature].between(lo, hi).all():
            raise ValueError(f"{feature} outside training range [{lo:g}, {hi:g}]. Prediction blocked.")
    return model.predict(inputs[FEATURES])
