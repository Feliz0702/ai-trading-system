import numpy as np
import pandas as pd
from typing import Dict
from scientific_backtest_engine.utils.metrics import PerformanceMetrics


class PerformanceAnalyzer:
    """性能深度分析器"""

    def comprehensive_analysis(self, returns: pd.Series) -> Dict[str, float]:
        returns = returns.copy().replace([np.inf, -np.inf], np.nan).dropna()
        if returns.empty:
            return {}
        metrics = PerformanceMetrics.calculate_all_metrics(returns)
        # 其他分析可在此擴充（例如 rolling 指標、回撤分佈等）
        return metrics

# 增強的可視化分析器
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Optional, Any


class EnhancedPerformanceAnalyzer:
    """增強的性能分析器，提供交互式圖表儀表板"""

    def __init__(self):
        self.figures = {}

    def create_comprehensive_dashboard(self, returns: pd.Series, benchmark_returns: Optional[pd.Series] = None) -> go.Figure:
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                '累計收益曲線', '滾動夏普比率',
                '回撤分析', '月度收益熱力圖',
                '收益分佈', '壓力測試情景'
            ),
            specs=[[{"secondary_y": True}, {}],
                   [{"secondary_y": True}, {}],
                   [{}, {}]]
        )
        cum = self._calculate_cumulative_returns(returns)
        if benchmark_returns is not None:
            bench_cum = self._calculate_cumulative_returns(benchmark_returns)
            fig.add_trace(go.Scatter(x=bench_cum.index, y=bench_cum, name='基準', line=dict(dash='dash')), row=1, col=1)
        fig.add_trace(go.Scatter(x=cum.index, y=cum, name='策略', line=dict(color='blue')), row=1, col=1)
        rolling_sharpe = self._calculate_rolling_sharpe(returns, window=126)
        fig.add_trace(go.Scatter(x=rolling_sharpe.index, y=rolling_sharpe, name='滾動夏普比率', line=dict(color='green')), row=1, col=2)
        drawdown = self._calculate_drawdown(returns)
        fig.add_trace(go.Scatter(x=drawdown.index, y=drawdown, name='回撤', fill='tozeroy', line=dict(color='red')), row=2, col=1)
        monthly = self._calculate_monthly_returns(returns)
        heat = monthly.pivot_table(index=monthly.index.year, columns=monthly.index.month, values='returns')
        fig.add_trace(go.Heatmap(z=heat.values, x=heat.columns, y=heat.index, colorscale='RdYlGn', showscale=False), row=2, col=2)
        fig.add_trace(go.Histogram(x=returns, name='收益分佈', nbinsx=50), row=3, col=1)
        for i, sc in enumerate(self._generate_stress_scenarios(returns)[:3]):
            fig.add_trace(go.Scatter(y=sc, name=f'情景 {i+1}', line=dict(width=1), showlegend=False), row=3, col=2)
        fig.update_layout(height=1200, title_text='策略性能綜合儀表板')
        return fig

    def plot_rolling_metrics(self, returns: pd.Series, windows: List[int] = [63, 126, 252]) -> go.Figure:
        fig = make_subplots(rows=2, cols=2, subplot_titles=('滾動夏普比率', '滾動最大回撤', '滾動年化收益', '滾動波動率'))
        for i, w in enumerate(windows):
            rs = self._calculate_rolling_sharpe(returns, w)
            fig.add_trace(go.Scatter(x=rs.index, y=rs, name=f'{w}天', showlegend=(i == 0)), row=1, col=1)
            rd = self._calculate_rolling_drawdown(returns, w)
            fig.add_trace(go.Scatter(x=rd.index, y=rd, name=f'{w}天', showlegend=False), row=1, col=2)
            rret = returns.rolling(w).mean() * 252
            fig.add_trace(go.Scatter(x=rret.index, y=rret, name=f'{w}天', showlegend=False), row=2, col=1)
            rvol = returns.rolling(w).std() * np.sqrt(252)
            fig.add_trace(go.Scatter(x=rvol.index, y=rvol, name=f'{w}天', showlegend=False), row=2, col=2)
        fig.update_layout(height=800, title_text='滾動性能指標')
        return fig

    def plot_stress_scenario_distribution(self, stress_results: Dict[str, Any]) -> go.Figure:
        fig = go.Figure()
        for scenario, color in zip(['black_swan', 'flash_crash', 'liquidity_crisis'], ['red', 'orange', 'purple']):
            if scenario in stress_results:
                perf = stress_results[scenario]['performance_distribution']
                fig.add_trace(go.Violin(y=perf, name=scenario.replace('_', ' ').title(), box_visible=True, meanline_visible=True, fillcolor=color, line_color=color, opacity=0.6))
        fig.update_layout(title='壓力測試性能分佈', yaxis_title='夏普比率', xaxis_title='情景類型', showlegend=True)
        return fig

    def _calculate_rolling_sharpe(self, returns: pd.Series, window: int = 126) -> pd.Series:
        rm = returns.rolling(window).mean()
        rs = returns.rolling(window).std()
        return (rm * np.sqrt(252)) / (rs + 1e-8)

    def _calculate_rolling_drawdown(self, returns: pd.Series, window: int = 126) -> pd.Series:
        cumulative = (1 + returns).cumprod()
        roll_max = cumulative.rolling(window, min_periods=1).max()
        dd = (cumulative - roll_max) / roll_max
        return dd.rolling(window).min()

    def _calculate_cumulative_returns(self, returns: pd.Series) -> pd.Series:
        return (1 + returns).cumprod()

    def _calculate_drawdown(self, returns: pd.Series) -> pd.Series:
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        return (cumulative - running_max) / running_max

    def _calculate_monthly_returns(self, returns: pd.Series) -> pd.DataFrame:
        monthly = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
        return monthly.to_frame('returns')

    def _generate_stress_scenarios(self, returns: pd.Series, n_scenarios: int = 5) -> List[List[float]]:
        scenarios = []
        base = returns.mean()
        for i in range(n_scenarios):
            lvl = np.random.uniform(0.5, 2.0)
            sc = [base * lvl * (1 + 0.1 * j) for j in range(10)]
            scenarios.append(sc)
        return scenarios
