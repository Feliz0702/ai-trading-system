import numpy as np
import pandas as pd
from typing import Dict
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


class PerformanceMetrics:
    """性能指標計算類"""

    @staticmethod
    def calculate_all_metrics(returns: pd.Series, risk_free_rate: float = 0.02) -> Dict[str, float]:
        if returns is None or len(returns) == 0:
            return {}
        returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
        if len(returns) == 0:
            return {}
        metrics: Dict[str, float] = {}
        metrics.update(PerformanceMetrics._basic(returns, risk_free_rate))
        metrics.update(PerformanceMetrics._risk_adjusted(returns, risk_free_rate))
        metrics.update(PerformanceMetrics._downside(returns, risk_free_rate))
        metrics.update(PerformanceMetrics._drawdown(returns))
        metrics.update(PerformanceMetrics._stats_tests(returns))
        return metrics

    @staticmethod
    def _basic(returns: pd.Series, rf: float) -> Dict[str, float]:
        return {
            'total_return': float(returns.sum()),
            'annual_return': float(returns.mean() * 252),
            'annual_volatility': float(returns.std(ddof=0) * np.sqrt(252)),
            'cumulative_return': float((1 + returns).prod() - 1),
            'sharpe_ratio': float(PerformanceMetrics.calculate_sharpe_ratio(returns, rf)),
            'sortino_ratio': float(PerformanceMetrics.calculate_sortino_ratio(returns, rf)),
            'calmar_ratio': float(PerformanceMetrics.calculate_calmar_ratio(returns)),
        }

    @staticmethod
    def _risk_adjusted(returns: pd.Series, rf: float) -> Dict[str, float]:
        excess = returns - rf / 252
        return {
            'omega_ratio': float(PerformanceMetrics.calculate_omega_ratio(returns, rf)),
            'information_ratio': float(PerformanceMetrics.calculate_information_ratio(returns, excess)),
            'treynor_ratio': float(PerformanceMetrics.calculate_treynor_ratio(returns, rf, beta=1.0)),
            'appraisal_ratio': float(PerformanceMetrics.calculate_appraisal_ratio(returns, excess)),
        }

    @staticmethod
    def _downside(returns: pd.Series, rf: float) -> Dict[str, float]:
        return {
            'var_95': float(PerformanceMetrics.calculate_var(returns, 0.95)),
            'cvar_95': float(PerformanceMetrics.calculate_cvar(returns, 0.95)),
            'ulcer_index': float(PerformanceMetrics.calculate_ulcer_index(returns)),
            'pain_ratio': float(PerformanceMetrics.calculate_pain_ratio(returns)),
        }

    @staticmethod
    def _drawdown(returns: pd.Series) -> Dict[str, float]:
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return {
            'max_drawdown': float(drawdown.min()),
            'avg_drawdown': float(drawdown.mean()),
            'drawdown_std': float(drawdown.std(ddof=0)),
            'recovery_time': float(PerformanceMetrics.calculate_avg_recovery_time(drawdown)),
        }

    @staticmethod
    def _stats_tests(returns: pd.Series) -> Dict[str, float]:
        jb = stats.jarque_bera(returns)
        norm = stats.normaltest(returns)
        return {
            'skewness': float(returns.skew()),
            'kurtosis': float(returns.kurtosis()),
            'jarque_bera_stat': float(jb[0]),
            'jarque_bera_pvalue': float(jb[1]),
            'normality_pvalue': float(norm[1]),
        }

    @staticmethod
    def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        excess = returns - risk_free_rate / 252
        denom = excess.std(ddof=0)
        if len(excess) < 2 or denom == 0 or np.isnan(denom):
            return 0.0
        return float(np.sqrt(252) * excess.mean() / denom)

    @staticmethod
    def calculate_sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.02, target_return: float = 0.0) -> float:
        downside = returns[returns < target_return / 252]
        denom = downside.std(ddof=0) * np.sqrt(252)
        if len(downside) < 2 or denom == 0 or np.isnan(denom):
            return 0.0
        return float((returns.mean() * 252 - risk_free_rate) / denom)

    @staticmethod
    def calculate_calmar_ratio(returns: pd.Series, period: int = 252) -> float:
        max_dd = PerformanceMetrics.calculate_max_drawdown(returns)
        annual = returns.mean() * period
        return 0.0 if max_dd == 0 else float(annual / abs(max_dd))

    @staticmethod
    def calculate_max_drawdown(returns: pd.Series) -> float:
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return float(drawdown.min())

    @staticmethod
    def calculate_omega_ratio(returns: pd.Series, risk_free_rate: float = 0.02, target_return: float = 0.0) -> float:
        threshold = target_return / 252
        gains = returns[returns > threshold].sum()
        losses = returns[returns <= threshold].sum()
        if losses == 0:
            return float('inf')
        return float(gains / abs(losses))

    @staticmethod
    def calculate_var(returns: pd.Series, confidence_level: float = 0.95) -> float:
        if len(returns) == 0:
            return 0.0
        return float(np.percentile(returns, 100 * (1 - confidence_level)))

    @staticmethod
    def calculate_cvar(returns: pd.Series, confidence_level: float = 0.95) -> float:
        var = PerformanceMetrics.calculate_var(returns, confidence_level)
        tail = returns[returns <= var]
        return float(var if len(tail) == 0 else tail.mean())

    @staticmethod
    def calculate_ulcer_index(returns: pd.Series) -> float:
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        if len(drawdown) == 0:
            return 0.0
        return float(np.sqrt(np.mean(drawdown ** 2)))

    @staticmethod
    def calculate_pain_ratio(returns: pd.Series) -> float:
        ulcer = PerformanceMetrics.calculate_ulcer_index(returns)
        annual = returns.mean() * 252
        return 0.0 if ulcer == 0 else float(annual / ulcer)

    @staticmethod
    def calculate_avg_recovery_time(drawdown: pd.Series) -> float:
        recovery_times = []
        in_dd = False
        start = 0
        for i, dd in enumerate(drawdown):
            if dd < -0.05 and not in_dd:
                in_dd = True; start = i
            elif dd >= -0.01 and in_dd:
                in_dd = False; recovery_times.append(i - start)
        return 0.0 if not recovery_times else float(np.mean(recovery_times))

    @staticmethod
    def calculate_information_ratio(returns: pd.Series, benchmark_returns: pd.Series) -> float:
        active = returns - benchmark_returns
        denom = active.std(ddof=0)
        if len(active) < 2 or denom == 0 or np.isnan(denom):
            return 0.0
        return float(np.sqrt(252) * active.mean() / denom)

    @staticmethod
    def calculate_treynor_ratio(returns: pd.Series, risk_free_rate: float, beta: float) -> float:
        excess = returns.mean() * 252 - risk_free_rate
        return 0.0 if beta == 0 else float(excess / beta)

    @staticmethod
    def calculate_appraisal_ratio(returns: pd.Series, benchmark_returns: pd.Series) -> float:
        active = returns - benchmark_returns
        te = active.std(ddof=0) * np.sqrt(252)
        if te == 0 or np.isnan(te):
            return 0.0
        return float(active.mean() * 252 / te)
