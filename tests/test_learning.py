import numpy as np
import pandas as pd
import pytest
from reservoir.data import ROOT, load_training
from reservoir.learning import fit_models, predict, FEATURES
from reservoir.control import learn_policy, tank_step


@pytest.fixture(scope='module')
def bundle():
    return fit_models(load_training())


def test_holdout_and_models(bundle):
    assert not set(bundle['train_ids']) & set(bundle['test_ids'])
    assert len(bundle['train_ids']) == 24 and len(bundle['test_ids']) == 8
    assert np.isfinite(bundle['scores'].select_dtypes('number')).all().all()
    assert not bundle['warnings']
    scores = bundle['scores'].set_index('Model')['MAE (m3/day)']
    assert scores['Random forest'] < scores['Mean baseline']
    assert scores['Deep MLP (64 / 64 / 32)'] < scores['Mean baseline']


def test_out_of_range_blocked(bundle):
    frame = pd.DataFrame({c: [np.mean(bundle['bounds'][c])] for c in FEATURES})
    frame['day'] = 999
    with pytest.raises(ValueError, match='outside training range'):
        predict(bundle['models']['Random forest'], frame, bundle['bounds'])


def test_rl_reproducibility_and_balance():
    a, _ = learn_policy(7000, episodes=100)
    b, _ = learn_policy(7000, episodes=100)
    pd.testing.assert_frame_equal(a, b)
    assert set(a.rate_m3_day) <= {0., 10., 20., 30.}
    oil, reward, produced, water = tank_step(.7, 20., 7000)
    assert oil < .7 and water >= 0
    assert produced + water == pytest.approx(600)
    assert produced == pytest.approx(7000*(.7-oil))
    assert reward == pytest.approx(produced-.4*water-.15*600)
