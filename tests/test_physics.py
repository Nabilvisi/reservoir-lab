import numpy as np
import pytest
from reservoir.data import load_rock, simulation_grid
from reservoir.physics import FlowModel, Scenario, CONVERSION


def uniform():
    return {**{k: np.full((2, 2, 4), 100.) for k in ['permx', 'permy', 'permz']},
            'porosity': np.full((2, 2, 4), .2), 'spacing': np.array([10., 10., 10.])}


def test_analytical_pressure():
    model = FlowModel(uniform())
    ids = np.arange(model.n).reshape(model.shape)
    model.injectors, model.producers = ids[:, :, 0].ravel(), ids[:, :, -1].ravel()
    s = np.tile([1., 0., 0.], (model.n, 1))
    p, _, _, _, residual = model.pressure(s, 40.)
    expected = np.broadcast_to((200 + np.arange(3, -1, -1) * (10 / (CONVERSION * 100 * 10))), model.shape)
    np.testing.assert_allclose(p.reshape(model.shape), expected, atol=1e-9)
    assert residual < 1e-9


@pytest.mark.parametrize('gas', [0., .5, 1.])
def test_phase_conservation(gas):
    result = FlowModel(uniform()).run(Scenario(rate=15, gas_fraction=gas, days=90), snapshots=7)
    s, h = result['saturation'], result['history']
    assert s.min() >= -1e-10 and s.max() <= 1+1e-10
    np.testing.assert_allclose(s.sum(axis=-1), 1, atol=1e-9)
    assert h.balance_error_m3.max() < 1e-8
    np.testing.assert_allclose(h[['water_m3_day', 'oil_m3_day', 'gas_m3_day']].sum(axis=1), 15.)
    assert h.cumulative_oil_m3.is_monotonic_increasing


def test_zero_flow():
    result = FlowModel(uniform()).run(Scenario(rate=0, days=30), snapshots=3)
    np.testing.assert_allclose(result['saturation'][0], result['saturation'][-1])
    np.testing.assert_allclose(result['pressure'], 200.)
    assert result['history'].cumulative_oil_m3.max() == 0


@pytest.mark.parametrize('scenario', [Scenario(rate=-1), Scenario(gas_fraction=1.1), Scenario(days=0), Scenario(rate=float('nan'))])
def test_invalid_scenarios(scenario):
    with pytest.raises(ValueError):
        FlowModel(uniform()).run(scenario)


def test_source_and_reduction():
    rock, metadata = load_rock()
    assert rock['porosity'].shape == (6, 30, 18)
    assert metadata['content_license'] == 'DbCL-1.0'
    grid = simulation_grid(rock)
    assert grid['porosity'].shape == (3, 15, 9)
    assert grid['porosity'].min() >= .02
    np.testing.assert_allclose(grid['spacing'], rock['spacing']*2)
    assert (grid['permx'] > 0).all()
