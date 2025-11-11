import numpy as np
import pandas as pd
from typing import Dict, Any, Callable

from scientific_backtest_engine.utils.metrics import PerformanceMetrics


class RobustnessTester:
    """魯棒性測試器"""

    def __init__(self, strategy: Callable, data: pd.DataFrame):
        self.strategy = strategy
        self.data = data

    def parameter_sensitivity_analysis(self, base_params: Dict[str, Any]) -> Dict[str, float]:
        """對連續型參數做±10%擾動，觀察夏普比率變化。"""
        results: Dict[str, float] = {}
        for k, v in base_params.items():
            if isinstance(v, (int, float)) and v != 0:
                sharpes = []
                for scale in (0.9, 1.1):
                    params2 = dict(base_params)
                    params2[k] = v * scale
                    try:
                        ret = self.strategy(self.data.copy(), params2)
                        sharpes.append(PerformanceMetrics.calculate_sharpe_ratio(ret))
                    except Exception:
                        sharpes.append(0.0)
                if sharpes:
                    mu = float(np.mean(sharpes))
                    sigma = float(np.std(sharpes))
                    results[k] = 0.0 if mu == 0 else float(sigma / (mu + 1e-8))
        return results

    def transaction_cost_analysis(self, scenarios: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """模擬不同手續費/滑點對策略表現的影響（簡化：以成本直接扣減日收益）。"""
        impacts: Dict[str, float] = {}
        for name, costs in scenarios.items():
            commission = float(costs.get('commission', 0.0))
            slippage = float(costs.get('slippage', 0.0))
            try:
                base_returns = self.strategy(self.data.copy(), {})
                adj = base_returns - (commission + slippage) / 252.0
                impacts[name] = PerformanceMetrics.calculate_sharpe_ratio(adj)
            except Exception:
                impacts[name] = 0.0
        return impacts
