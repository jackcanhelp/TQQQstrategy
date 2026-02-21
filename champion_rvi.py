"""
Champion RVI Strategy — Proven Seed Strategy
=============================================
Translated from a successful Pine Script strategy.
Uses Relative Volatility Index (RVI) with state machine logic
and conditional short selling with ATR-based exits.

This serves as the "Champion DNA" that the AI evolution system
should study, mutate, and improve upon.
"""

import pandas as pd
import numpy as np
from strategy_base import BaseStrategy


def calculate_rvi(src: pd.Series, stdev_length: int = 34,
                  smooth_length: int = 20) -> pd.Series:
    """
    Calculate Relative Volatility Index.

    RVI measures volatility direction: rising volatility in uptrends vs downtrends.
    Unlike RSI which measures price direction, RVI measures if volatility
    accompanies upward or downward moves.

    Args:
        src: Price series (typically Close)
        stdev_length: Lookback for standard deviation (default 34)
        smooth_length: EMA smoothing length (default 20)

    Returns:
        pd.Series: RVI values 0-100
    """
    stdev = src.rolling(stdev_length).std()
    change = src.diff()

    up_vol = stdev.where(change >= 0, 0.0)
    down_vol = stdev.where(change < 0, 0.0)

    up_sum = up_vol.ewm(span=smooth_length, adjust=False).mean()
    down_sum = down_vol.ewm(span=smooth_length, adjust=False).mean()

    rvi = 100.0 * up_sum / (up_sum + down_sum)
    return rvi


def calculate_rvi_refined(high: pd.Series, low: pd.Series,
                          stdev_length: int = 34,
                          smooth_length: int = 20) -> pd.Series:
    """
    Refined RVI: average of RVI(High) and RVI(Low).
    Smoother and more robust than single-source RVI.
    """
    rvi_high = calculate_rvi(high, stdev_length, smooth_length)
    rvi_low = calculate_rvi(low, stdev_length, smooth_length)
    return (rvi_high + rvi_low) / 2.0


class ChampionRVI(BaseStrategy):
    """
    Champion RVI Strategy — State Machine with Conditional Short Selling.

    Core Logic:
    1. RVI (Relative Volatility Index) divides market into 3 states:
       - Green: RVI > buy_trigger (59) — bullish volatility
       - Orange: sell_low (42) <= RVI <= buy_trigger (59) — neutral
       - Red: RVI < sell_low (42) — bearish volatility

    2. Entry: State transition from Orange/Red → Green (momentum building)
    3. Exit Long: RVI > sell_high (76) or RVI < sell_low (42)
    4. Entry Short: State transition from Orange → Red
    5. Exit Short: ATR × factor for take-profit and stop-loss

    This strategy captures the TRANSITION moments, not just levels.
    """

    def __init__(
        self,
        stdev_length: int = 34,
        smooth_length: int = 20,
        buy_trigger: int = 59,
        sell_high: int = 76,
        sell_low: int = 42,
        use_refined: bool = True,
        enable_short: bool = True,
        volatility_factor: float = 1.8,
        atr_period: int = 14,
    ):
        super().__init__()
        self.stdev_length = stdev_length
        self.smooth_length = smooth_length
        self.buy_trigger = buy_trigger
        self.sell_high = sell_high
        self.sell_low = sell_low
        self.use_refined = use_refined
        self.enable_short = enable_short
        self.volatility_factor = volatility_factor
        self.atr_period = atr_period

    def get_params(self) -> dict:
        """Return current parameters for sweep/optimization."""
        return {
            'stdev_length': self.stdev_length,
            'smooth_length': self.smooth_length,
            'buy_trigger': self.buy_trigger,
            'sell_high': self.sell_high,
            'sell_low': self.sell_low,
            'volatility_factor': self.volatility_factor,
            'atr_period': self.atr_period,
        }

    def init(self, data: pd.DataFrame) -> None:
        self.data = data.copy()

        # Calculate RVI
        if self.use_refined:
            self.rvi = calculate_rvi_refined(
                data['High'], data['Low'],
                self.stdev_length, self.smooth_length
            )
        else:
            self.rvi = calculate_rvi(
                data['Close'],
                self.stdev_length, self.smooth_length
            )

        # Calculate ATR for short selling exits
        high = data['High']
        low = data['Low']
        close = data['Close']
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs()
        ], axis=1).max(axis=1)
        self.atr = tr.rolling(self.atr_period).mean()

    def _get_state(self) -> pd.Series:
        """
        Determine market state based on RVI levels.

        Returns:
            pd.Series with values: 'green', 'orange', 'red'
        """
        state = pd.Series('orange', index=self.data.index)
        state = state.where(~(self.rvi > self.buy_trigger), 'green')
        state = state.where(~(self.rvi < self.sell_low), 'red')
        return state

    def _get_transitions(self, state: pd.Series) -> pd.Series:
        """
        Detect state transitions.

        Returns:
            pd.Series with transition labels like 'orange_to_green', etc.
        """
        prev_state = state.shift(1)
        transitions = prev_state.astype(str) + '_to_' + state.astype(str)
        return transitions

    def generate_signals(self) -> pd.Series:
        """
        Generate trading signals based on state machine transitions.

        Returns:
            pd.Series: -1.0 (short), 0.0 (cash), 1.0 (long)
        """
        state = self._get_state()
        transitions = self._get_transitions(state)

        signals = pd.Series(0.0, index=self.data.index)

        # Track position state
        position = 0  # 0=flat, 1=long, -1=short
        short_entry_price = None
        short_entry_atr = None

        for i in range(len(self.data)):
            rvi_val = self.rvi.iloc[i]
            transition = transitions.iloc[i]
            atr_val = self.atr.iloc[i]
            current_open = self.data['Open'].iloc[i]
            current_high = self.data['High'].iloc[i]
            current_low = self.data['Low'].iloc[i]

            if pd.isna(rvi_val) or pd.isna(atr_val):
                signals.iloc[i] = 0.0
                continue

            # === LONG LOGIC ===
            # Buy: transition from orange/red to green
            if transition in ('orange_to_green', 'red_to_green'):
                position = 1
                # Close any short
                if short_entry_price is not None:
                    short_entry_price = None
                    short_entry_atr = None

            # Sell long: RVI too high (overbought) or too low (breakdown)
            if position == 1:
                if rvi_val > self.sell_high or rvi_val < self.sell_low:
                    position = 0

            # === SHORT LOGIC ===
            if self.enable_short:
                # Short entry: transition from orange to red
                if transition == 'orange_to_red' and position != 1:
                    position = -1
                    short_entry_price = current_open
                    short_entry_atr = atr_val

                # Short exit: ATR-based take profit / stop loss
                if position == -1 and short_entry_price is not None:
                    tp_price = short_entry_price - (short_entry_atr * self.volatility_factor)
                    sl_price = short_entry_price + (short_entry_atr * self.volatility_factor)

                    if current_low <= tp_price:
                        position = 0
                        short_entry_price = None
                        short_entry_atr = None
                    elif current_high >= sl_price:
                        position = 0
                        short_entry_price = None
                        short_entry_atr = None

                # Stop short if buy signal fires
                if transition in ('orange_to_green', 'red_to_green'):
                    if position == -1:
                        short_entry_price = None
                        short_entry_atr = None

            signals.iloc[i] = float(position)

        return signals

    def get_description(self) -> str:
        return (
            f"Champion RVI Strategy: State machine with RVI "
            f"(stdev={self.stdev_length}, smooth={self.smooth_length}). "
            f"States: Green(>{self.buy_trigger}), Orange, Red(<{self.sell_low}). "
            f"Buy on orange/red→green transition, sell at RVI>{self.sell_high} or <{self.sell_low}. "
            f"Short on orange→red with ATR×{self.volatility_factor} TP/SL."
        )


# ═══════════════════════════════════════════════════════════════
# Quick test
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    from indicator_pool import get_enriched_data
    from backtest import BacktestEngine

    data = get_enriched_data()
    engine = BacktestEngine(data)

    # Test long-only first
    champ = ChampionRVI(enable_short=False)
    champ.init(data)
    signals = champ.generate_signals()
    print(f"Signal distribution: long={( signals > 0).sum()}, "
          f"cash={(signals == 0).sum()}, short={(signals < 0).sum()}")
    print(f"Time in market: {(signals != 0).mean():.1%}")

    # Run backtest (long-only compatible with current engine)
    result = engine.run(champ)
    print(f"\nChampion RVI (long-only):")
    print(f"  Sharpe: {result.sharpe_ratio:.2f}")
    print(f"  CAGR: {result.cagr:.1%}")
    print(f"  Max DD: {result.max_drawdown:.1%}")
    print(f"  Calmar: {result.calmar_ratio:.2f}")
