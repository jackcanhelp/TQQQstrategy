"""
Strategy Ensemble System
=========================
Combines multiple strategies into ensemble portfolios.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from strategy_base import BaseStrategy


class EnsembleStrategy(BaseStrategy):
    """
    Combines multiple strategies using various methods.
    """

    def __init__(
        self,
        strategies: List[BaseStrategy],
        weights: List[float] = None,
        method: str = 'weighted_average'
    ):
        """
        Args:
            strategies: List of strategy instances
            weights: Optional weights for each strategy (must sum to 1)
            method: 'weighted_average', 'vote', 'min', 'max', 'dynamic'
        """
        super().__init__()
        self.strategies = strategies
        self.method = method

        if weights is None:
            self.weights = [1.0 / len(strategies)] * len(strategies)
        else:
            assert len(weights) == len(strategies)
            assert abs(sum(weights) - 1.0) < 0.001
            self.weights = weights

        self.name = f"Ensemble_{method}_{len(strategies)}strats"

    def init(self, data: pd.DataFrame) -> None:
        self.data = data
        for s in self.strategies:
            s.init(data)

    def generate_signals(self) -> pd.Series:
        all_signals = []

        for s in self.strategies:
            signals = s.generate_signals()
            signals = s.validate_signals(signals)
            all_signals.append(signals)

        signals_df = pd.concat(all_signals, axis=1)
        signals_df.columns = [f'strat_{i}' for i in range(len(self.strategies))]

        if self.method == 'weighted_average':
            result = sum(signals_df.iloc[:, i] * self.weights[i]
                        for i in range(len(self.strategies)))

        elif self.method == 'vote':
            # Majority vote: if >50% strategies say invest, invest
            votes = (signals_df > 0.5).sum(axis=1) / len(self.strategies)
            result = (votes > 0.5).astype(float)

        elif self.method == 'min':
            # Conservative: take minimum signal
            result = signals_df.min(axis=1)

        elif self.method == 'max':
            # Aggressive: take maximum signal
            result = signals_df.max(axis=1)

        elif self.method == 'dynamic':
            # Weight by recent performance (risk parity-like)
            result = self._dynamic_weighting(signals_df)

        else:
            result = signals_df.mean(axis=1)

        return result

    def _dynamic_weighting(self, signals_df: pd.DataFrame, lookback: int = 60) -> pd.Series:
        """
        Dynamically weight strategies by inverse volatility of their signals.
        """
        # Calculate rolling volatility of each strategy's signals
        vols = signals_df.rolling(lookback).std()
        inv_vols = 1 / (vols + 0.01)  # Add small number to avoid div by zero
        weights = inv_vols.div(inv_vols.sum(axis=1), axis=0)

        # Weighted average
        result = (signals_df * weights).sum(axis=1)
        return result.fillna(0)

    def get_description(self) -> str:
        strat_names = [s.name for s in self.strategies]
        return f"Ensemble ({self.method}): {', '.join(strat_names)}"


class StrategyRanker:
    """
    Ranks and selects best strategies for ensemble.
    """

    @staticmethod
    def rank_by_sharpe(results: List[Dict]) -> List[Dict]:
        """Sort strategies by Sharpe ratio."""
        return sorted(results, key=lambda x: x['sharpe'], reverse=True)

    @staticmethod
    def rank_by_risk_adjusted(results: List[Dict]) -> List[Dict]:
        """
        Rank by custom risk-adjusted score:
        Score = Sharpe * (1 + Sortino) / (1 + |MaxDD|)
        """
        for r in results:
            r['risk_score'] = (
                r['sharpe'] * (1 + r.get('sortino', 0)) /
                (1 + abs(r['max_dd']))
            )
        return sorted(results, key=lambda x: x['risk_score'], reverse=True)

    @staticmethod
    def select_diverse_strategies(
        results: List[Dict],
        signals_dict: Dict[str, pd.Series],
        n_select: int = 3,
        min_correlation: float = 0.7
    ) -> List[str]:
        """
        Select top strategies that are not too correlated.
        """
        # Sort by Sharpe
        ranked = StrategyRanker.rank_by_sharpe(results)
        selected = []

        for r in ranked:
            name = r['name']
            if name not in signals_dict:
                continue

            # Check correlation with already selected
            is_diverse = True
            for sel_name in selected:
                corr = signals_dict[name].corr(signals_dict[sel_name])
                if abs(corr) > min_correlation:
                    is_diverse = False
                    break

            if is_diverse:
                selected.append(name)

            if len(selected) >= n_select:
                break

        return selected


def create_optimized_ensemble(
    strategies: Dict[str, BaseStrategy],
    results: Dict[str, Dict],
    data: pd.DataFrame,
    top_n: int = 3,
    method: str = 'weighted_average'
) -> Tuple[EnsembleStrategy, Dict]:
    """
    Create an ensemble from the best strategies.

    Args:
        strategies: Dict of {name: strategy_instance}
        results: Dict of {name: backtest_result_dict}
        data: Historical data
        top_n: Number of strategies to include
        method: Ensemble method

    Returns:
        Tuple of (ensemble_strategy, ensemble_info)
    """
    # Rank strategies
    results_list = [{'name': k, **v} for k, v in results.items()]
    ranked = StrategyRanker.rank_by_risk_adjusted(results_list)

    # Select top N
    selected_names = [r['name'] for r in ranked[:top_n]]
    selected_strategies = [strategies[n] for n in selected_names]

    # Weight by Sharpe ratio
    sharpes = [results[n]['sharpe'] for n in selected_names]
    total_sharpe = sum(max(s, 0.01) for s in sharpes)  # Avoid negative weights
    weights = [max(s, 0.01) / total_sharpe for s in sharpes]

    ensemble = EnsembleStrategy(
        strategies=selected_strategies,
        weights=weights,
        method=method
    )

    info = {
        'selected_strategies': selected_names,
        'weights': weights,
        'method': method
    }

    return ensemble, info
