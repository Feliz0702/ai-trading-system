import pandas as pd
import numpy as np
from typing import Callable, List
import warnings
warnings.filterwarnings('ignore')


class DataProcessor:
    """數據處理管道"""

    def __init__(self):
        self.cleaners: List[Callable] = []
        self.transformers: List[Callable] = []
        self.validators: List[Callable] = []

    def add_cleaner(self, cleaner_func: Callable):
        self.cleaners.append(cleaner_func)
        return self

    def add_transformer(self, transformer_func: Callable):
        self.transformers.append(transformer_func)
        return self

    def add_validator(self, validator_func: Callable):
        self.validators.append(validator_func)
        return self

    def process(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        data = raw_data.copy()
        for cleaner in self.cleaners:
            data = cleaner(data)
        for transformer in self.transformers:
            data = transformer(data)
        for validator in self.validators:
            if not validator(data):
                raise ValueError("數據驗證失敗")
        return data

    @staticmethod
    def remove_duplicates(data: pd.DataFrame) -> pd.DataFrame:
        return data[~data.index.duplicated(keep='first')]

    @staticmethod
    def fill_missing_values(data: pd.DataFrame, method: str = 'ffill') -> pd.DataFrame:
        if method == 'ffill':
            return data.ffill()
        elif method == 'bfill':
            return data.bfill()
        else:
            return data.fillna(method=method)

    @staticmethod
    def add_returns(data: pd.DataFrame, price_col: str = 'close') -> pd.DataFrame:
        data['returns'] = data[price_col].pct_change(fill_method=None).fillna(0)
        data['log_returns'] = np.log(data[price_col] / data[price_col].shift(1)).fillna(0)
        return data

    @staticmethod
    def add_technical_indicators(data: pd.DataFrame) -> pd.DataFrame:
        data['sma_20'] = data['close'].rolling(20).mean()
        data['sma_50'] = data['close'].rolling(50).mean()
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        data['rsi'] = 100 - (100 / (1 + rs))
        data['bb_middle'] = data['close'].rolling(20).mean()
        bb_std = data['close'].rolling(20).std()
        data['bb_upper'] = data['bb_middle'] + 2 * bb_std
        data['bb_lower'] = data['bb_middle'] - 2 * bb_std
        return data

    @staticmethod
    def validate_data(data: pd.DataFrame) -> bool:
        if data.empty:
            return False
        if data.isnull().sum().sum() / max(data.size, 1) > 0.1:
            return False
        if 'close' in data.columns and (data['close'] <= 0).any():
            return False
        return True
