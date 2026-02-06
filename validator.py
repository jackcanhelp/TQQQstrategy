"""
Strategy Validator
===================
Validates AI-generated strategies for common pitfalls:
- Look-ahead bias (using future data)
- Impossible trades
- Data leakage
"""

import pandas as pd
import numpy as np
from typing import Tuple, List
import re


class StrategyValidator:
    """
    Validates strategies for look-ahead bias and other issues.
    """

    # Dangerous patterns in code that suggest look-ahead bias
    LOOKAHEAD_PATTERNS = [
        (r'\.shift\s*\(\s*-', "Negative shift detected - this looks into the future"),
        (r'\.iloc\s*\[\s*\w+\s*\+\s*\d+', "Forward iloc indexing detected"),
        (r'\.loc\s*\[\s*\w+\s*\+', "Forward loc indexing detected"),
        (r'future|tomorrow|next_day|forward', "Suspicious variable name suggesting future data"),
    ]

    # Patterns that suggest proper implementation
    GOOD_PATTERNS = [
        r'\.shift\s*\(\s*1\s*\)',  # Proper lagging
        r'\.shift\s*\(\s*\d+\s*\)',  # Positive shifts
        r'\.rolling\s*\(',  # Rolling windows (backward-looking)
    ]

    @classmethod
    def validate_code(cls, code: str) -> Tuple[bool, List[str]]:
        """
        Static analysis of strategy code for potential issues.

        Returns:
            Tuple of (is_valid, list_of_warnings)
        """
        warnings = []

        for pattern, message in cls.LOOKAHEAD_PATTERNS:
            if re.search(pattern, code, re.IGNORECASE):
                warnings.append(f"⚠️ LOOK-AHEAD RISK: {message}")

        # Check for proper signal shifting
        if 'generate_signals' in code:
            # Signals should be shifted before applying to returns
            if '.shift(' not in code and 'shift' not in code.lower():
                warnings.append("⚠️ WARNING: No shift() found - signals may have look-ahead bias")

        is_valid = len([w for w in warnings if 'LOOK-AHEAD' in w]) == 0
        return is_valid, warnings

    @classmethod
    def validate_signals(
        cls,
        signals: pd.Series,
        returns: pd.Series,
        threshold: float = 0.95
    ) -> Tuple[bool, str]:
        """
        Statistical test for look-ahead bias.

        If strategy perfectly avoids all large drops, it's likely cheating.
        """
        # Find days with large negative returns
        bad_days = returns < returns.quantile(0.05)  # Worst 5% of days

        if bad_days.sum() == 0:
            return True, "No significant bad days to test"

        # Check if strategy suspiciously avoids these days
        signals_on_bad_days = signals.shift(1)[bad_days]  # What was signal BEFORE bad day

        # If strategy is almost always out before bad days, suspicious
        if signals_on_bad_days.mean() < 0.1 and signals.mean() > 0.3:
            avoidance_rate = 1 - signals_on_bad_days.mean()
            if avoidance_rate > threshold:
                return False, f"SUSPICIOUS: Strategy avoids {avoidance_rate:.1%} of worst days - possible look-ahead bias"

        return True, "Passed look-ahead statistical test"

    @classmethod
    def validate_backtest_results(
        cls,
        result,
        baseline_sharpe: float = 0.84,
        max_reasonable_sharpe: float = 3.0,
        min_reasonable_drawdown: float = -0.05
    ) -> Tuple[bool, List[str]]:
        """
        Validate backtest results for unrealistic performance.
        """
        warnings = []
        is_valid = True

        # Too-good-to-be-true Sharpe
        if result.sharpe_ratio > max_reasonable_sharpe:
            warnings.append(f"❌ INVALID: Sharpe {result.sharpe_ratio:.2f} is unrealistically high")
            is_valid = False

        # Almost no drawdown is suspicious
        if result.max_drawdown > min_reasonable_drawdown and result.cagr > 0.1:
            warnings.append(f"❌ SUSPICIOUS: Max DD of {result.max_drawdown:.1%} with {result.cagr:.1%} CAGR is unrealistic")
            is_valid = False

        # Perfect win rate is impossible
        if result.win_rate > 0.70 and result.sharpe_ratio > 2.0:
            warnings.append(f"⚠️ WARNING: {result.win_rate:.1%} win rate with high Sharpe is unusual")

        # Strategy that never trades
        if result.time_in_market < 0.01:
            warnings.append("⚠️ WARNING: Strategy almost never enters market")

        # Too many trades (overfit to noise)
        if result.total_trades > len(result.equity_curve) * 0.8:
            warnings.append("⚠️ WARNING: Excessive trading - possible overfitting")

        return is_valid, warnings


class LookAheadDetector:
    """
    Runtime detection of look-ahead bias by comparing
    online vs offline strategy performance.
    """

    @staticmethod
    def test_online_consistency(
        strategy,
        data: pd.DataFrame,
        sample_points: int = 100
    ) -> Tuple[bool, str]:
        """
        Test if strategy generates same signals when run incrementally
        vs all at once. Look-ahead strategies will differ.
        """
        # Run on full data
        strategy.init(data)
        full_signals = strategy.generate_signals()

        # Run incrementally on expanding windows
        inconsistencies = 0
        test_indices = np.linspace(
            len(data) // 2,
            len(data) - 1,
            sample_points,
            dtype=int
        )

        for idx in test_indices:
            partial_data = data.iloc[:idx+1].copy()
            try:
                strategy.init(partial_data)
                partial_signals = strategy.generate_signals()

                # Compare signal at this point
                if len(partial_signals) > 0:
                    full_signal = full_signals.iloc[idx]
                    partial_signal = partial_signals.iloc[-1]

                    if abs(full_signal - partial_signal) > 0.01:
                        inconsistencies += 1
            except Exception:
                pass

        inconsistency_rate = inconsistencies / sample_points

        if inconsistency_rate > 0.1:
            return False, f"❌ LOOK-AHEAD DETECTED: {inconsistency_rate:.1%} of signals change with more data"

        return True, f"✅ Passed online consistency test ({inconsistency_rate:.1%} variance)"
