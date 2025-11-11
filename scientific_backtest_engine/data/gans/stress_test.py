import numpy as np
import pandas as pd
from typing import List
from .financial_gan import FinancialGAN


class StressTestGenerator:
    """壓力測試情景生成器"""

    def __init__(self, base_data: pd.DataFrame):
        self.base_data = base_data
        self.gan_model: FinancialGAN | None = None

    def fit_gan(self, sequence_length: int = 60, epochs: int = 5):
        features = self._prepare_features()
        sequences = self._create_sequences(features, sequence_length)
        if sequences.size == 0:
            self.gan_model = FinancialGAN(sequence_length=sequence_length, feature_dim=features.shape[1] if features.ndim == 2 else 5)
            return
        self.gan_model = FinancialGAN(sequence_length=sequence_length, feature_dim=sequences.shape[2])
        self.gan_model.fit(sequences, epochs=max(epochs, 1))

    def generate_black_swan(self, n_scenarios: int = 20) -> List[pd.DataFrame]:
        if self.gan_model is None:
            self.fit_gan()
        synthetic = self.gan_model.generate_samples(n_scenarios)
        scenarios: List[pd.DataFrame] = []
        for i in range(n_scenarios):
            scenario = self._sequence_to_dataframe(synthetic[i])
            scenario = self._apply_extreme_shock(scenario)
            scenarios.append(scenario)
        return scenarios

    def generate_flash_crash(self, n_scenarios: int = 20, recovery_days: int = 5) -> List[pd.DataFrame]:
        scenarios: List[pd.DataFrame] = []
        n = len(self.base_data)
        for _ in range(n_scenarios):
            scenario = self.base_data.copy()
            if n < recovery_days + 20:
                scenarios.append(self._recalculate_returns(scenario))
                continue
            start = np.random.randint(10, n - recovery_days - 10)
            crash = float(np.random.uniform(0.15, 0.35))
            scenario.iloc[start, scenario.columns.get_loc('close')] *= (1 - crash)
            for i in range(1, recovery_days + 1):
                rec = float(np.random.uniform(0.02, 0.08))
                scenario.iloc[start + i, scenario.columns.get_loc('close')] = scenario.iloc[start + i - 1, scenario.columns.get_loc('close')] * (1 + rec)
            scenarios.append(self._recalculate_returns(scenario))
        return scenarios

    def generate_liquidity_crisis(self, n_scenarios: int = 20) -> List[pd.DataFrame]:
        scenarios: List[pd.DataFrame] = []
        scenario = self.base_data.copy()
        scenario['volatility'] = scenario['close'].pct_change().rolling(20).std().fillna(0)
        for _ in range(n_scenarios):
            sc = scenario.copy()
            vol_mult = float(np.random.uniform(2.0, 4.0))
            sc['volatility'] *= vol_mult
            spread_shock = float(np.random.uniform(3.0, 8.0))
            noise = np.random.normal(0, 0.01 * spread_shock, len(sc))
            sc['returns'] = sc['close'].pct_change().fillna(0) + noise
            sc['close'] = (1 + sc['returns']).cumprod() * sc['close'].iloc[0]
            scenarios.append(sc)
        return scenarios

    def _prepare_features(self) -> np.ndarray:
        cols = [c for c in ['open', 'high', 'low', 'close', 'volume'] if c in self.base_data.columns]
        arrs = [self.base_data[c].values for c in cols]
        if 'returns' in self.base_data.columns:
            arrs.append(self.base_data['returns'].values)
        return np.column_stack(arrs) if arrs else np.empty((0, 0))

    def _create_sequences(self, data: np.ndarray, length: int) -> np.ndarray:
        if data.size == 0 or len(data) <= length:
            return np.empty((0, length, data.shape[1] if data.ndim == 2 and data.size > 0 else 5))
        seqs = [data[i:i + length] for i in range(len(data) - length)]
        return np.array(seqs)

    def _sequence_to_dataframe(self, sequence: np.ndarray) -> pd.DataFrame:
        # 根據序列寬度自動匹配欄位（含 returns 作為第6欄的情況）
        cols = ['open', 'high', 'low', 'close', 'volume', 'returns']
        out_cols = cols[: sequence.shape[1]]
        # 調整長度匹配 base_data
        seq = self._resize_sequence(sequence, len(self.base_data)) if len(sequence) != len(self.base_data) else sequence
        idx = self.base_data.index[: len(seq)]
        df = pd.DataFrame(seq, columns=out_cols, index=idx)
        return self._recalculate_returns(df)

    def _resize_sequence(self, sequence: np.ndarray, target_length: int) -> np.ndarray:
        """調整序列長度到 target_length，使用線性插值或截斷。"""
        if len(sequence) == target_length:
            return sequence
        if len(sequence) > target_length:
            return sequence[:target_length]
        # 插值到 target_length
        try:
            from scipy.interpolate import interp1d
            x_old = np.linspace(0, 1, len(sequence))
            x_new = np.linspace(0, 1, target_length)
            resized = np.zeros((target_length, sequence.shape[1]), dtype=float)
            for i in range(sequence.shape[1]):
                f = interp1d(x_old, sequence[:, i], kind='linear', fill_value="extrapolate", bounds_error=False)
                resized[:, i] = f(x_new)
            return resized
        except Exception:
            # 後備：重複填充
            reps = int(np.ceil(target_length / len(sequence)))
            tiled = np.tile(sequence, (reps, 1))
            return tiled[:target_length]

    def _apply_extreme_shock(self, data: pd.DataFrame) -> pd.DataFrame:
        shock = float(np.random.uniform(0.2, 0.5))
        direction = np.random.choice([-1, 1])
        if len(data) < 20 or 'close' not in data.columns:
            return data
        p = np.random.randint(10, len(data) - 10)
        data.loc[data.index[p]:, 'close'] *= (1 + direction * shock)
        return self._recalculate_returns(data)

    def _recalculate_returns(self, data: pd.DataFrame) -> pd.DataFrame:
        if 'close' in data.columns:
            data['returns'] = data['close'].pct_change().fillna(0)
        return data
