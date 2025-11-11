from itertools import product
from typing import Dict, List, Any, Callable
import pandas as pd


def grid_search(strategy: Callable, data: pd.DataFrame, parameter_space: Dict[str, List[Any]], score_fn: Callable) -> Dict[str, Any]:
    """簡單網格搜索，返回分數最佳的參數。"""
    best_params: Dict[str, Any] | None = None
    best_score = float('-inf')
    for params in product(*parameter_space.values()):
        param_dict = dict(zip(parameter_space.keys(), params))
        try:
            returns = strategy(data.copy(), param_dict)
            score = float(score_fn(returns))
            if score > best_score:
                best_score = score
                best_params = param_dict
        except Exception:
            continue
    return best_params or {k: v[0] for k, v in parameter_space.items()}
