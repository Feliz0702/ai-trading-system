import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Callable
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

from scientific_backtest_engine.config.settings import WalkForwardConfig
from scientific_backtest_engine.utils.metrics import PerformanceMetrics


@dataclass
class WalkForwardResult:
    in_sample_performance: Dict[str, float]
    out_of_sample_performance: Dict[str, float]
    parameters: Dict[str, Any]
    decay_metrics: Dict[str, float]
    stability_scores: Dict[str, float]


class WalkForwardAnalyzer:
    def __init__(self, config: WalkForwardConfig):
        self.config = config
        self.results: List[WalkForwardResult] = []

    def rolling_window_split(self, data: pd.DataFrame) -> List[Tuple[int, int, int, int]]:
        splits = []
        n = len(data)
        for i in range(0, n - self.config.window_size, self.config.step_size):
            train_start = i
            train_end = i + self.config.window_size
            test_start = train_end
            test_end = min(test_start + self.config.step_size, n)
            if test_end - test_start >= self.config.min_periods:
                splits.append((train_start, train_end, test_start, test_end))
        return splits

    def expanding_window_split(self, data: pd.DataFrame) -> List[Tuple[int, int, int, int]]:
        splits = []
        n = len(data)
        for i in range(self.config.min_periods, n - self.config.min_periods, self.config.step_size):
            train_start = 0
            train_end = i
            test_start = i
            test_end = min(i + self.config.step_size, n)
            splits.append((train_start, train_end, test_start, test_end))
        return splits

    def analyze(self, strategy: Callable, data: pd.DataFrame, parameter_space: Dict[str, List[Any]]) -> List[WalkForwardResult]:
        if self.config.method == "rolling":
            splits = self.rolling_window_split(data)
        else:
            splits = self.expanding_window_split(data)
        results: List[WalkForwardResult] = []
        for train_start, train_end, test_start, test_end in splits:
            train_data = data.iloc[train_start:train_end]
            test_data = data.iloc[test_start:test_end]
            if len(train_data) < self.config.min_periods or len(test_data) < self.config.min_periods:
                continue
            best_params, best_performance = self._optimize_parameters(strategy, train_data, parameter_space)
            test_performance = self._evaluate_strategy(strategy, test_data, best_params)
            decay_metrics = self._calculate_performance_decay(best_performance, test_performance)
            stability_scores = self._calculate_stability_scores(train_data, test_data, strategy, best_params)
            results.append(WalkForwardResult(best_performance, test_performance, best_params, decay_metrics, stability_scores))
        self.results = results
        return results

    def _optimize_parameters(self, strategy: Callable, data: pd.DataFrame, parameter_space: Dict[str, List[Any]]) -> Tuple[Dict[str, Any], Dict[str, float]]:
        best_params = None
        best_performance = None
        from itertools import product
        for params in product(*parameter_space.values()):
            param_dict = dict(zip(parameter_space.keys(), params))
            try:
                performance = self._evaluate_strategy(strategy, data, param_dict)
                if best_performance is None or performance.get('sharpe_ratio', 0) > best_performance.get('sharpe_ratio', 0):
                    best_performance = performance
                    best_params = param_dict
            except Exception:
                continue
        return best_params or {}, best_performance or {}

    def _evaluate_strategy(self, strategy: Callable, data: pd.DataFrame, parameters: Dict[str, Any]) -> Dict[str, float]:
        returns = strategy(data.copy(), parameters)
        return PerformanceMetrics.calculate_all_metrics(returns)

    def _calculate_performance_decay(self, in_sample: Dict[str, float], out_of_sample: Dict[str, float]) -> Dict[str, float]:
        decay: Dict[str, float] = {}
        for metric in in_sample.keys():
            if metric in out_of_sample:
                a = in_sample[metric]
                b = out_of_sample[metric]
                decay[metric] = (b / a) if a not in (0, None) else 0
        return decay

    def _calculate_stability_scores(self, train_data: pd.DataFrame, test_data: pd.DataFrame, strategy: Callable, parameters: Dict[str, Any]) -> Dict[str, float]:
        # 參數敏感度（簡化版）
        sensitivities = []
        for k, v in parameters.items():
            if isinstance(v, (int, float)) and v != 0:
                for scale in (0.9, 1.1):
                    params2 = dict(parameters)
                    params2[k] = v * scale
                    try:
                        perf = self._evaluate_strategy(strategy, train_data, params2)
                        sensitivities.append(perf.get('sharpe_ratio', 0))
                    except Exception:
                        pass
        param_sensitivity = 0
        if len(sensitivities) >= 2 and np.mean(sensitivities) != 0:
            param_sensitivity = np.std(sensitivities) / (np.mean(sensitivities) + 1e-8)
        # 時間穩定性（簡化版）
        n_splits = 5
        split_size = max(1, len(train_data) // n_splits)
        shard_sharpes = []
        for i in range(n_splits):
            subset = train_data.iloc[i*split_size:(i+1)*split_size]
            if len(subset) < 3:
                continue
            try:
                perf = self._evaluate_strategy(strategy, subset, parameters)
                shard_sharpes.append(perf.get('sharpe_ratio', 0))
            except Exception:
                pass
        time_stability = 0
        if shard_sharpes and np.mean(shard_sharpes) > 0:
            cv = np.std(shard_sharpes) / (np.mean(shard_sharpes) + 1e-8)
            time_stability = 1 / (1 + cv)
        return {"parameter_sensitivity": 1 - param_sensitivity, "time_stability": time_stability}

    def get_summary_stats(self) -> Dict[str, Any]:
        if not self.results:
            return {}
        return {
            'avg_in_sample_sharpe': np.mean([r.in_sample_performance.get('sharpe_ratio', 0) for r in self.results]),
            'avg_out_of_sample_sharpe': np.mean([r.out_of_sample_performance.get('sharpe_ratio', 0) for r in self.results]),
            'avg_performance_decay': np.mean([r.decay_metrics.get('sharpe_ratio', 0) for r in self.results]),
            'success_rate': np.mean([1 if r.out_of_sample_performance.get('sharpe_ratio', 0) > 0 else 0 for r in self.results]),
            'parameter_stability': np.mean([r.stability_scores.get('parameter_sensitivity', 0) for r in self.results]),
            'time_stability': np.mean([r.stability_scores.get('time_stability', 0) for r in self.results])
        }
