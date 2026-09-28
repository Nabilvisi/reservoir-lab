from streamlit.testing.v1 import AppTest
from reservoir.data import ROOT


def test_all_pages_and_controls():
    app = AppTest.from_file(ROOT / 'app.py', default_timeout=120).run()
    assert not app.exception
    app.selectbox[0].select('Porosity').run()
    assert not app.exception
    app.radio[0].set_value('Dynamic modelling').run()
    assert not app.exception
    app.slider[0].set_value(0.)
    app.button[0].click().run()
    assert not app.exception
    assert app.metric[1].value == '0 m³'
    app.selectbox[0].select('Pressure').run()
    assert not app.exception
    app.radio[0].set_value('ML / Deep learning').run()
    assert not app.exception
    app.selectbox[0].select('Deep MLP (64 / 64 / 32)').run()
    assert not app.exception
    app.radio[0].set_value('RL experiment').run()
    assert not app.exception
    app.radio[0].set_value('Data & methods').run()
    assert not app.exception
