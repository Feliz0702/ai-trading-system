from dataclasses import dataclass
from typing import List


@dataclass
class BacktestConfig:
    """回測配置類"""
    initial_capital: float = 1_000_000
    transaction_cost: float = 0.001
    slippage: float = 0.0005
    risk_free_rate: float = 0.02
    benchmark: str = "SPY"


@dataclass
class WalkForwardConfig:
    """前向分析配置"""
    window_size: int = 252
    step_size: int = 63
    min_periods: int = 63
    method: str = "rolling"  # rolling, expanding, anchored


@dataclass
class GANConfig:
    """GAN配置"""
    sequence_length: int = 60
    hidden_dim: int = 128
    num_layers: int = 3
    learning_rate: float = 0.001
    epochs: int = 1000


@dataclass
class AnalysisConfig:
    """分析配置"""
    confidence_level: float = 0.95
    monte_carlo_runs: int = 1000
    bootstrap_samples: int = 10000
    risk_metrics: List[str] = None

    def __post_init__(self):
        if self.risk_metrics is None:
            self.risk_metrics = ["var", "cvar", "max_drawdown"]
