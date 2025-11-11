import pandas as pd
import numpy as np

from scientific_backtest_engine import ScientificBacktestEngine, BacktestConfig


def moving_average_crossover_strategy(data: pd.DataFrame, params):
    short_window = params.get('short_window', 10)
    long_window = params.get('long_window', 30)
    data = data.copy()
    data['short_ma'] = data['close'].rolling(short_window).mean()
    data['long_ma'] = data['close'].rolling(long_window).mean()
    data['signal'] = 0
    data.loc[data['short_ma'] > data['long_ma'], 'signal'] = 1
    data.loc[data['short_ma'] < data['long_ma'], 'signal'] = -1
    data['strategy_returns'] = data['signal'].shift(1).fillna(0) * data['returns'].fillna(0)
    return data['strategy_returns'].dropna()


def generate_sample_data(n: int = 600) -> pd.DataFrame:
    dates = pd.date_range('2020-01-01', periods=n, freq='D')
    ret = np.random.normal(0.0005, 0.02, n)
    price = 100 * np.cumprod(1 + ret)
    df = pd.DataFrame({
        'date': dates,
        'open': price * (1 + np.random.normal(0, 0.001, n)),
        'high': price * (1 + np.abs(np.random.normal(0, 0.005, n))),
        'low': price * (1 - np.abs(np.random.normal(0, 0.005, n))),
        'close': price,
        'volume': np.random.lognormal(10, 1, n)
    })
    df.set_index('date', inplace=True)
    return df


def test_engine_comprehensive_analysis_smoke():
    data = generate_sample_data(400)
    engine = ScientificBacktestEngine(BacktestConfig())
    engine.set_strategy(moving_average_crossover_strategy)
    engine.load_data(data)
    param_space = {
        'short_window': [10, 15],
        'long_window': [30, 40],
    }
    results = engine.run_comprehensive_analysis(param_space)
    assert 'base_backtest' in results
    assert 'walk_forward_analysis' in results
    assert 'overfitting_test' in results
    assert 'stress_tests' in results
    assert 'final_assessment' in results
    base_metrics = results['base_backtest']['performance_metrics']
    assert isinstance(base_metrics, dict)
    # sharpe 可能為 0，但鍵應存在
    assert 'sharpe_ratio' in base_metrics
