import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Callable
import warnings
warnings.filterwarnings('ignore')

from scientific_backtest_engine.config.settings import BacktestConfig, WalkForwardConfig
from scientific_backtest_engine.core.walk_forward import WalkForwardAnalyzer
from scientific_backtest_engine.data.data_processor import DataProcessor
from scientific_backtest_engine.data.gans.stress_test import StressTestGenerator
from scientific_backtest_engine.analysis.overfitting_test import OverfittingProbabilityTest
from scientific_backtest_engine.utils.metrics import PerformanceMetrics


class ScientificBacktestEngine:
    """科學回測引擎主類"""

    def __init__(self, config: Optional[BacktestConfig] = None):
        self.config = config or BacktestConfig()
        self.data_processor = DataProcessor()
        self.strategy: Optional[Callable] = None
        self.data: Optional[pd.DataFrame] = None
        self.results: Dict[str, Any] = {}

        self.walk_forward_analyzer: Optional[WalkForwardAnalyzer] = None
        self.overfitting_tester: Optional[OverfittingProbabilityTest] = None
        self.stress_test_generator: Optional[StressTestGenerator] = None

    def load_data(self, data: pd.DataFrame, preprocess: bool = True) -> 'ScientificBacktestEngine':
        self.data = data.copy()
        if preprocess:
            self.data = self._preprocess_data(self.data)
        return self

    def set_strategy(self, strategy: Callable) -> 'ScientificBacktestEngine':
        self.strategy = strategy
        return self

    def run_comprehensive_analysis(self, parameter_space: Dict[str, List[Any]]) -> Dict[str, Any]:
        if self.strategy is None or self.data is None:
            raise ValueError("請先設置策略和加載數據")

        base_results = self._run_base_backtest(parameter_space)
        walk_forward_results = self._run_walk_forward_analysis(parameter_space)
        overfitting_results = self._run_overfitting_test(parameter_space)
        stress_test_results = self._run_stress_tests()
        performance_analysis = self._run_performance_analysis()

        self.results = {
            'base_backtest': base_results,
            'walk_forward_analysis': walk_forward_results,
            'overfitting_test': overfitting_results,
            'stress_tests': stress_test_results,
            'performance_analysis': performance_analysis,
            'final_assessment': self._generate_final_assessment()
        }
        return self.results

    def _preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        processor = DataProcessor()
        processed_data = (
            processor
            .add_cleaner(DataProcessor.remove_duplicates)
            .add_cleaner(DataProcessor.fill_missing_values)
            .add_transformer(DataProcessor.add_returns)
            .add_transformer(DataProcessor.add_technical_indicators)
            .add_validator(DataProcessor.validate_data)
            .process(data)
        )
        return processed_data

    def _run_base_backtest(self, parameter_space: Dict[str, List[Any]]) -> Dict[str, Any]:
        best_params = self._find_optimal_parameters(parameter_space)
        returns = self.strategy(self.data.copy(), best_params)
        metrics = PerformanceMetrics.calculate_all_metrics(returns)
        return {
            'optimal_parameters': best_params,
            'returns': returns,
            'performance_metrics': metrics,
        }

    def _run_walk_forward_analysis(self, parameter_space: Dict[str, List[Any]]) -> Dict[str, Any]:
        wf_config = WalkForwardConfig()
        self.walk_forward_analyzer = WalkForwardAnalyzer(wf_config)
        results = self.walk_forward_analyzer.analyze(self.strategy, self.data, parameter_space)
        summary = self.walk_forward_analyzer.get_summary_stats()
        return {'detailed_results': results, 'summary_statistics': summary}

    def _run_overfitting_test(self, parameter_space: Dict[str, List[Any]]) -> Dict[str, Any]:
        self.overfitting_tester = OverfittingProbabilityTest(n_monte_carlo=100)
        results = self.overfitting_tester.calculate_pbo(self.strategy, self.data, parameter_space)
        report = self.overfitting_tester.get_detailed_report()
        return {'pbo_results': results, 'detailed_report': report}

    def _run_stress_tests(self) -> Dict[str, Any]:
        if self.stress_test_generator is None:
            self.stress_test_generator = StressTestGenerator(self.data)
        stress_results: Dict[str, Any] = {}
        black_swan_scenarios = self.stress_test_generator.generate_black_swan(10)
        stress_results['black_swan'] = self._evaluate_stress_scenarios(black_swan_scenarios)
        flash_crash_scenarios = self.stress_test_generator.generate_flash_crash(10)
        stress_results['flash_crash'] = self._evaluate_stress_scenarios(flash_crash_scenarios)
        liquidity_scenarios = self.stress_test_generator.generate_liquidity_crisis(10)
        stress_results['liquidity_crisis'] = self._evaluate_stress_scenarios(liquidity_scenarios)
        return stress_results

    def _run_performance_analysis(self) -> Dict[str, Any]:
        returns = self.results['base_backtest']['returns'] if 'base_backtest' in self.results else pd.Series(dtype=float)
        return PerformanceMetrics.calculate_all_metrics(returns)

    def _find_optimal_parameters(self, parameter_space: Dict[str, List[Any]]) -> Dict[str, Any]:
        best_params: Optional[Dict[str, Any]] = None
        best_perf = -float('inf')
        from itertools import product
        for params in product(*parameter_space.values()):
            param_dict = dict(zip(parameter_space.keys(), params))
            try:
                returns = self.strategy(self.data.copy(), param_dict)
                perf = PerformanceMetrics.calculate_sharpe_ratio(returns)
                if perf > best_perf:
                    best_perf = perf
                    best_params = param_dict
            except Exception:
                continue
        return best_params or {k: v[0] for k, v in parameter_space.items()}

    def _evaluate_stress_scenarios(self, scenarios: List[pd.DataFrame]) -> Dict[str, Any]:
        performances: List[float] = []
        best_params = self.results.get('base_backtest', {}).get('optimal_parameters')
        if not best_params:
            return {'avg_performance': 0, 'min_performance': 0, 'max_performance': 0, 'survival_rate': 0, 'performance_distribution': []}
        for scenario in scenarios:
            try:
                returns = self.strategy(scenario.copy(), best_params)
                perf = PerformanceMetrics.calculate_sharpe_ratio(returns)
                performances.append(perf)
            except Exception:
                performances.append(-float('inf'))
        arr = np.array(performances) if performances else np.array([0.0])
        return {
            'avg_performance': float(np.mean(arr)),
            'min_performance': float(np.min(arr)),
            'max_performance': float(np.max(arr)),
            'survival_rate': float(np.mean(arr > 0)),
            'performance_distribution': performances,
        }

    def _generate_final_assessment(self) -> Dict[str, Any]:
        assessment: Dict[str, Any] = {}
        base_metrics = self.results.get('base_backtest', {}).get('performance_metrics', {})
        assessment['base_performance'] = {
            'sharpe_ratio': base_metrics.get('sharpe_ratio', 0.0),
            'max_drawdown': base_metrics.get('max_drawdown', 0.0),
            'annual_return': base_metrics.get('annual_return', 0.0),
        }
        wf_summary = self.results.get('walk_forward_analysis', {}).get('summary_statistics', {})
        assessment['stability'] = {
            'out_of_sample_sharpe': wf_summary.get('avg_out_of_sample_sharpe', 0.0),
            'performance_decay': wf_summary.get('avg_performance_decay', 0.0),
            'success_rate': wf_summary.get('success_rate', 0.0),
        }
        overfit_report = self.results.get('overfitting_test', {}).get('detailed_report', {})
        assessment['overfitting_risk'] = {
            'pbo': overfit_report.get('overfitting_probability', 0.0),
            'risk_level': 'unknown',
        }
        stress = self.results.get('stress_tests', {})
        assessment['stress_resilience'] = {
            'black_swan_survival': stress.get('black_swan', {}).get('survival_rate', 0.0),
            'flash_crash_survival': stress.get('flash_crash', {}).get('survival_rate', 0.0),
            'liquidity_crisis_survival': stress.get('liquidity_crisis', {}).get('survival_rate', 0.0),
        }
        assessment['composite_score'] = self._calculate_composite_score(assessment)
        return assessment

    def _calculate_composite_score(self, assessment: Dict[str, Any]) -> float:
        scores: List[float] = []
        base_sharpe = assessment['base_performance']['sharpe_ratio']
        scores.append(min(max(base_sharpe, 0), 3) / 3 * 100 * 0.4)
        decay = assessment['stability']['performance_decay']
        scores.append(max(0.0, 100 * (1 - abs(1 - decay))) * 0.3)
        pbo = assessment['overfitting_risk']['pbo']
        scores.append(100 * (1 - pbo) * 0.2)
        stress_scores = [
            assessment['stress_resilience']['black_swan_survival'],
            assessment['stress_resilience']['flash_crash_survival'],
            assessment['stress_resilience']['liquidity_crisis_survival'],
        ]
        scores.append(float(np.mean(stress_scores)) * 100 * 0.1)
        return float(sum(scores))
