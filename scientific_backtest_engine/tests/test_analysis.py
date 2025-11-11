import pandas as pd
import numpy as np

from scientific_backtest_engine.analysis.overfitting_test import OverfittingProbabilityTest
from scientific_backtest_engine.core.walk_forward import WalkForwardAnalyzer
from scientific_backtest_engine.config.settings import WalkForwardConfig


def simple_strategy(data: pd.DataFrame, params):
    n = params.get('n', 5)
    d = data.copy()
    d['ret'] = d['close'].pct_change().fillna(0)
    d['ma'] = d['close'].rolling(n).mean()
    d['sig'] = 0
    d.loc[d['close'] > d['ma'], 'sig'] = 1
    d.loc[d['close'] < d['ma'], 'sig'] = -1
    d['strategy_returns'] = d['sig'].shift(1).fillna(0) * d['ret']
    return d['strategy_returns'].dropna()


def gen_data(n: int = 300) -> pd.DataFrame:
    dates = pd.date_range('2021-01-01', periods=n, freq='D')
    r = np.random.normal(0.0004, 0.015, n)
    price = 100 * np.cumprod(1 + r)
    df = pd.DataFrame({'date': dates, 'close': price, 'open': price, 'high': price*1.01, 'low': price*0.99, 'volume': np.random.lognormal(10, 1, n)})
    df.set_index('date', inplace=True)
    return df


def test_overfitting_probability_smoke():
    data = gen_data(250)
    tester = OverfittingProbabilityTest(n_monte_carlo=10)
    param_space = {'n': [5, 10, 15]}
    res = tester.calculate_pbo(simple_strategy, data, param_space)
    assert 'pbo' in res
    report = tester.get_detailed_report()
    assert 'overfitting_probability' in report


def test_walk_forward_analyzer_smoke():
    data = gen_data(300)
    cfg = WalkForwardConfig(window_size=100, step_size=30, min_periods=20, method='rolling')
    wfa = WalkForwardAnalyzer(cfg)
    param_space = {'n': [5, 10]}
    results = wfa.analyze(simple_strategy, data, param_space)
    assert isinstance(results, list)
    summary = wfa.get_summary_stats()
    assert isinstance(summary, dict)
