"""
Backtest Engine
================
Runs strategies on historical data and computes performance metrics.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class BacktestResult:
    """Container for backtest results and analysis."""
    strategy_name: str
    start_date: str
    end_date: str

    # Core metrics
    total_return: float
    cagr: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    max_drawdown_duration_days: int

    # Additional metrics
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    total_trades: int

    # Time-in-market
    time_in_market: float

    # Crisis analysis
    crisis_periods: List[Dict]

    # Equity curve
    equity_curve: pd.Series
    drawdown_series: pd.Series

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'strategy_name': self.strategy_name,
            'start_date': self.start_date,
            'end_date': self.end_date,
            'total_return': round(self.total_return, 4),
            'cagr': round(self.cagr, 4),
            'sharpe_ratio': round(self.sharpe_ratio, 4),
            'sortino_ratio': round(self.sortino_ratio, 4),
            'max_drawdown': round(self.max_drawdown, 4),
            'max_drawdown_duration_days': self.max_drawdown_duration_days,
            'win_rate': round(self.win_rate, 4),
            'profit_factor': round(self.profit_factor, 4),
            'total_trades': self.total_trades,
            'time_in_market': round(self.time_in_market, 4),
            'crisis_periods': self.crisis_periods
        }

    def get_failure_analysis(self) -> str:
        """Generate human-readable failure analysis for AI context."""
        analysis = []

        if self.max_drawdown < -0.5:
            analysis.append(f"CRITICAL: Max drawdown of {self.max_drawdown:.1%} is catastrophic")
        elif self.max_drawdown < -0.3:
            analysis.append(f"SEVERE: Max drawdown of {self.max_drawdown:.1%} is too deep")
        elif self.max_drawdown < -0.2:
            analysis.append(f"WARNING: Max drawdown of {self.max_drawdown:.1%} needs improvement")

        if self.sharpe_ratio < 0.5:
            analysis.append(f"LOW SHARPE: {self.sharpe_ratio:.2f} indicates poor risk-adjusted returns")

        if self.cagr < 0:
            analysis.append(f"NEGATIVE RETURNS: CAGR of {self.cagr:.1%}")

        if self.time_in_market < 0.3:
            analysis.append(f"LOW EXPOSURE: Only {self.time_in_market:.1%} time in market")

        for crisis in self.crisis_periods:
            if crisis['drawdown'] < -0.25:
                analysis.append(
                    f"CRISIS FAILURE: {crisis['period']} saw {crisis['drawdown']:.1%} drawdown"
                )

        if not analysis:
            analysis.append("Strategy performed reasonably well across all metrics")

        return " | ".join(analysis)


class BacktestEngine:
    """
    Runs backtests on TQQQ strategies.
    """

    # Known crisis periods for TQQQ
    CRISIS_PERIODS = {
        '2020_COVID': ('2020-02-19', '2020-03-23'),
        '2022_BEAR': ('2022-01-03', '2022-10-12'),
        '2018_Q4': ('2018-10-01', '2018-12-24'),
    }

    def __init__(self, data: pd.DataFrame, risk_free_rate: float = 0.02):
        """
        Initialize backtest engine.

        Args:
            data: OHLCV DataFrame with DatetimeIndex
            risk_free_rate: Annual risk-free rate for Sharpe calculation
        """
        self.data = data.copy()
        self.risk_free_rate = risk_free_rate
        self.daily_rf = (1 + risk_free_rate) ** (1/252) - 1

        # Pre-calculate returns
        self.data['Return'] = self.data['Close'].pct_change()

    def run(self, strategy) -> BacktestResult:
        """
        Run backtest for a given strategy.

        Args:
            strategy: Instance of BaseStrategy subclass

        Returns:
            BacktestResult with all metrics
        """
        # Initialize strategy
        strategy.init(self.data)

        # Generate signals
        raw_signals = strategy.generate_signals()
        signals = strategy.validate_signals(raw_signals)

        # Align signals with returns (shift by 1 to avoid lookahead)
        # Signal on day T determines position on day T+1
        position = signals.shift(1).fillna(0)

        # Calculate strategy returns
        strategy_returns = position * self.data['Return']
        strategy_returns = strategy_returns.fillna(0)

        # Calculate equity curve
        equity_curve = (1 + strategy_returns).cumprod()

        # Calculate metrics
        result = self._calculate_metrics(
            strategy_name=strategy.name,
            returns=strategy_returns,
            equity_curve=equity_curve,
            position=position
        )

        return result

    def _calculate_metrics(
        self,
        strategy_name: str,
        returns: pd.Series,
        equity_curve: pd.Series,
        position: pd.Series
    ) -> BacktestResult:
        """Calculate all performance metrics."""

        # Basic info
        start_date = returns.index[0].strftime('%Y-%m-%d')
        end_date = returns.index[-1].strftime('%Y-%m-%d')

        # Total return
        total_return = equity_curve.iloc[-1] - 1

        # CAGR
        years = (returns.index[-1] - returns.index[0]).days / 365.25
        cagr = (equity_curve.iloc[-1] ** (1 / years)) - 1 if years > 0 else 0

        # Sharpe Ratio (annualized)
        excess_returns = returns - self.daily_rf
        sharpe = np.sqrt(252) * excess_returns.mean() / excess_returns.std() if excess_returns.std() > 0 else 0

        # Sortino Ratio
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std()
        sortino = np.sqrt(252) * excess_returns.mean() / downside_std if downside_std > 0 else 0

        # Drawdown analysis
        rolling_max = equity_curve.cummax()
        drawdown = (equity_curve - rolling_max) / rolling_max
        max_drawdown = drawdown.min()

        # Max drawdown duration
        dd_duration = self._calculate_dd_duration(equity_curve)

        # Win rate and profit factor
        winning_days = returns[returns > 0]
        losing_days = returns[returns < 0]
        win_rate = len(winning_days) / len(returns[returns != 0]) if len(returns[returns != 0]) > 0 else 0
        avg_win = winning_days.mean() if len(winning_days) > 0 else 0
        avg_loss = losing_days.mean() if len(losing_days) > 0 else 0
        profit_factor = abs(winning_days.sum() / losing_days.sum()) if losing_days.sum() != 0 else float('inf')

        # Trade count (position changes)
        position_changes = position.diff().abs()
        total_trades = int((position_changes > 0).sum())

        # Time in market
        time_in_market = (position > 0).mean()

        # Crisis period analysis
        crisis_periods = self._analyze_crisis_periods(equity_curve, drawdown)

        return BacktestResult(
            strategy_name=strategy_name,
            start_date=start_date,
            end_date=end_date,
            total_return=total_return,
            cagr=cagr,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown=max_drawdown,
            max_drawdown_duration_days=dd_duration,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            total_trades=total_trades,
            time_in_market=time_in_market,
            crisis_periods=crisis_periods,
            equity_curve=equity_curve,
            drawdown_series=drawdown
        )

    def _calculate_dd_duration(self, equity_curve: pd.Series) -> int:
        """Calculate maximum drawdown duration in days."""
        rolling_max = equity_curve.cummax()
        in_drawdown = equity_curve < rolling_max

        # Find consecutive drawdown periods
        dd_groups = (~in_drawdown).cumsum()
        dd_lengths = in_drawdown.groupby(dd_groups).sum()

        return int(dd_lengths.max()) if len(dd_lengths) > 0 else 0

    def _analyze_crisis_periods(
        self,
        equity_curve: pd.Series,
        drawdown: pd.Series
    ) -> List[Dict]:
        """Analyze strategy performance during known crisis periods."""
        crisis_analysis = []

        for name, (start, end) in self.CRISIS_PERIODS.items():
            try:
                start_dt = pd.to_datetime(start)
                end_dt = pd.to_datetime(end)

                # Check if period exists in data
                mask = (equity_curve.index >= start_dt) & (equity_curve.index <= end_dt)
                if mask.sum() == 0:
                    continue

                period_equity = equity_curve[mask]
                period_dd = drawdown[mask]

                crisis_analysis.append({
                    'period': name,
                    'start': start,
                    'end': end,
                    'return': float(period_equity.iloc[-1] / period_equity.iloc[0] - 1),
                    'drawdown': float(period_dd.min())
                })
            except Exception:
                continue

        return crisis_analysis


def compare_strategies(results: List[BacktestResult]) -> pd.DataFrame:
    """Create comparison table of multiple strategy results."""
    data = []
    for r in results:
        data.append({
            'Strategy': r.strategy_name,
            'CAGR': f"{r.cagr:.1%}",
            'Sharpe': f"{r.sharpe_ratio:.2f}",
            'Sortino': f"{r.sortino_ratio:.2f}",
            'Max DD': f"{r.max_drawdown:.1%}",
            'Win Rate': f"{r.win_rate:.1%}",
            'Trades': r.total_trades
        })
    return pd.DataFrame(data)
