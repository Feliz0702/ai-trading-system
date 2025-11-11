import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Callable
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

from scientific_backtest_engine.utils.metrics import PerformanceMetrics


class OverfittingProbabilityTest:
    """過擬合概率檢驗"""

    def __init__(self, n_monte_carlo: int = 1000, confidence_level: float = 0.95):
        self.n_monte_carlo = n_monte_carlo
        self.confidence_level = confidence_level
        self.results: Dict[str, Any] = {}

    def calculate_pbo(self, strategy: Callable, data: pd.DataFrame, parameter_space: Dict[str, List[Any]]) -> Dict[str, Any]:
        monte_carlo_results = self._monte_carlo_simulation(strategy, data, parameter_space)
        rankings = self._calculate_performance_rankings(monte_carlo_results)
        pbo = self._calculate_probability_backtest_overfitting(rankings)
        kde_results = self._probability_density_estimation(monte_carlo_results)
        stability_metrics = self._performance_stability_test(monte_carlo_results)
        self.results = {
            'pbo': pbo,
            'monte_carlo_results': monte_carlo_results,
            'rankings': rankings,
            'kde_results': kde_results,
            'stability_metrics': stability_metrics,
            'confidence_interval': self._calculate_confidence_intervals(monte_carlo_results),
        }
        return self.results

    def _monte_carlo_simulation(self, strategy: Callable, data: pd.DataFrame, parameter_space: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for i in range(self.n_monte_carlo):
            train_data, test_data = self._random_split_data(data)
            random_params = self._sample_random_parameters(parameter_space)
            try:
                train_returns = strategy(train_data.copy(), random_params)
                train_perf = self._calculate_performance_metrics(train_returns)
                test_returns = strategy(test_data.copy(), random_params)
                test_perf = self._calculate_performance_metrics(test_returns)
                results.append({
                    'iteration': i,
                    'parameters': random_params,
                    'train_performance': train_perf,
                    'test_performance': test_perf,
                    'performance_ratio': (test_perf.get('sharpe_ratio', 0) / (train_perf.get('sharpe_ratio', 0) + 1e-8)),
                })
            except Exception:
                continue
        return results

    def _random_split_data(self, data: pd.DataFrame, train_ratio: float = 0.7) -> Tuple[pd.DataFrame, pd.DataFrame]:
        n = len(data)
        split = int(n * train_ratio)
        idx = np.random.permutation(n)
        train_idx = np.sort(idx[:split])
        test_idx = np.sort(idx[split:])
        return data.iloc[train_idx], data.iloc[test_idx]

    def _sample_random_parameters(self, parameter_space: Dict[str, List[Any]]) -> Dict[str, Any]:
        return {k: np.random.choice(v) for k, v in parameter_space.items()}

    def _calculate_performance_metrics(self, returns: pd.Series) -> Dict[str, float]:
        return PerformanceMetrics.calculate_all_metrics(returns)

    def _calculate_performance_rankings(self, results: List[Dict[str, Any]]) -> np.ndarray:
        if not results:
            return np.array([])
        train_sharpes = [r['train_performance'].get('sharpe_ratio', 0) for r in results]
        test_sharpes = [r['test_performance'].get('sharpe_ratio', 0) for r in results]
        train_ranks = stats.rankdata(train_sharpes)
        corresponding_test = [test_sharpes[i] for i in np.argsort(train_ranks)]
        return np.array(corresponding_test)

    def _calculate_probability_backtest_overfitting(self, rankings: np.ndarray) -> float:
        if rankings.size == 0:
            return 0.0
        threshold = np.percentile(rankings, 50)
        overfit = np.sum(rankings < threshold)
        return float(overfit / len(rankings))

    def _probability_density_estimation(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not results:
            return {'train_density': np.array([]), 'test_density': np.array([])}
        train = np.array([r['train_performance'].get('sharpe_ratio', 0) for r in results]).reshape(-1, 1)
        test = np.array([r['test_performance'].get('sharpe_ratio', 0) for r in results]).reshape(-1, 1)
        try:
            from sklearn.neighbors import KernelDensity
            kde_train = KernelDensity(kernel='gaussian', bandwidth=0.1).fit(train)
            kde_test = KernelDensity(kernel='gaussian', bandwidth=0.1).fit(test)
            return {
                'train_density': np.exp(kde_train.score_samples(train)),
                'test_density': np.exp(kde_test.score_samples(test)),
            }
        except Exception:
            return {'train_density': np.array([]), 'test_density': np.array([])}

    def _performance_stability_test(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        ratios = [r['performance_ratio'] for r in results] if results else [1.0]
        train = [r['train_performance'].get('sharpe_ratio', 0) for r in results] if results else [0]
        test = [r['test_performance'].get('sharpe_ratio', 0) for r in results] if results else [0]
        return {
            'sharpe_stability': float(np.corrcoef(train, test)[0, 1]) if len(train) > 1 else 0.0,
            'performance_consistency': float(np.std(ratios) / (np.mean(ratios) + 1e-8)),
            'success_consistency': float(np.mean([1 if r > 0 else 0 for r in test])),
        }

    def _calculate_confidence_intervals(self, results: List[Dict[str, Any]]) -> Dict[str, Tuple[float, float]]:
        if not results:
            return {'train_sharpe_ci': (0.0, 0.0), 'test_sharpe_ci': (0.0, 0.0)}
        train = [r['train_performance'].get('sharpe_ratio', 0) for r in results]
        test = [r['test_performance'].get('sharpe_ratio', 0) for r in results]
        alpha = 1 - self.confidence_level
        ci_train = np.percentile(train, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        ci_test = np.percentile(test, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        return {'train_sharpe_ci': tuple(ci_train), 'test_sharpe_ci': tuple(ci_test)}

    def get_detailed_report(self) -> Dict[str, Any]:
        if not self.results:
            return {}
        res = self.results
        pbo = res.get('pbo', 0.0)
        mc = res.get('monte_carlo_results', [])
        return {
            'overfitting_probability': pbo,
            'monte_carlo_iterations': len(mc),
            'average_train_performance': float(np.mean([r['train_performance'].get('sharpe_ratio', 0) for r in mc])) if mc else 0.0,
            'average_test_performance': float(np.mean([r['test_performance'].get('sharpe_ratio', 0) for r in mc])) if mc else 0.0,
            'performance_degradation': float(np.mean([r['performance_ratio'] for r in mc])) if mc else 0.0,
            'stability_assessment': res.get('stability_metrics', {}),
            'risk_assessment': {
                'risk_level': 'low' if pbo <= 0.3 else ('medium' if pbo <= 0.5 else 'high')
            },
        }
