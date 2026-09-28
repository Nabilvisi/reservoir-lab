"""Tabular Q-learning in a separate, deliberately reduced tank environment."""
import numpy as np
import pandas as pd

ACTIONS = np.array([0., 10., 20., 30.])
HORIZON = 12


def tank_step(oil, rate, pv):
    # Perfect mixing, equal mobility, pure water injection. Exact tank update.
    next_oil = oil * np.exp(-rate * 30 / pv)
    oil_volume = pv * (oil - next_oil)
    water_volume = rate * 30 - oil_volume
    reward = oil_volume - 0.4 * water_volume - 0.15 * rate * 30
    return next_oil, reward, oil_volume, water_volume


def learn_policy(pore_volume, episodes=4000, seed=42):
    if not np.isfinite(pore_volume) or pore_volume <= 0 or episodes < 1:
        raise ValueError("Positive pore volume and episode count required")
    rng = np.random.default_rng(seed)
    q = np.zeros((HORIZON, 41, len(ACTIONS)))
    rewards = []
    for episode in range(episodes):
        oil, total = 0.7, 0.
        epsilon = max(0.05, 1 - episode / (episodes * 0.8))
        for t in range(HORIZON):
            state = min(40, int(oil * 40))
            action = rng.integers(4) if rng.random() < epsilon else int(np.argmax(q[t, state]))
            new_oil, reward, _, _ = tank_step(oil, ACTIONS[action], pore_volume)
            next_state = min(40, int(new_oil * 40))
            future = q[t + 1, next_state].max() if t < HORIZON - 1 else 0.
            q[t, state, action] += 0.12 * (reward + future - q[t, state, action])
            oil, total = new_oil, total + reward
        rewards.append(total)
    records = []
    for name, fixed in [("Q-learning", None), ("Shut in", 0.), ("Fixed 10", 10.), ("Fixed 20", 20.), ("Fixed 30", 30.)]:
        oil, total = 0.7, 0.
        for t in range(HORIZON):
            rate = ACTIONS[np.argmax(q[t, min(40, int(oil * 40))])] if fixed is None else fixed
            oil, reward, produced, water = tank_step(oil, rate, pore_volume)
            total += reward
            records.append({"Policy": name, "day": (t + 1) * 30, "rate_m3_day": rate,
                            "oil_m3": produced, "water_m3": water, "reward": reward, "total_reward": total})
    return pd.DataFrame(records), pd.DataFrame({"episode": np.arange(1, episodes + 1), "reward": rewards})
