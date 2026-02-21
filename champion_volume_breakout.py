"""
Champion Volume Breakout Strategy — Seed Strategy #2
=====================================================
Translated from a proven Pine Script "突破放量策略".
Uses price breakout above MA + volume surge confirmation,
with ATR-based or percentage-based take profit / stop loss.

This serves alongside ChampionRVI as "Champion DNA" that
the AI evolution system should study, mutate, and improve upon.
"""

import pandas as pd
import numpy as np
from strategy_base import BaseStrategy


class ChampionVolumeBreakout(BaseStrategy):
    """
    Volume Breakout Strategy — Price + Volume Surge with Adaptive Exits.

    Core Logic:
    1. ENTRY: Close > MA(period) AND Volume > AvgVolume * multiplier
       - Price above moving average confirms uptrend
       - Volume surge confirms conviction / institutional participation

    2. EXIT (two modes):
       a) Percentage mode: fixed % take profit and stop loss from entry price
       b) ATR mode: ATR-scaled take profit and stop loss (adaptive to volatility)

    Why this works on TQQQ:
    - TQQQ is volatile; volume surges often precede strong directional moves
    - Combining trend (MA) + volume filter reduces false breakouts
    - ATR-based exits adapt to current volatility regime
    """

    def __init__(
        self,
        ma_length: int = 20,
        ma_type: str = 'SMA',
        vol_length: int = 20,
        vol_multiplier: float = 2.0,
        exit_mode: str = 'atr',
        tp_pct: float = 0.10,
        sl_pct: float = 0.05,
        atr_length: int = 14,
        atr_tp_mult: float = 2.5,
        atr_sl_mult: float = 1.5,
    ):
        super().__init__()
        self.ma_length = ma_length
        self.ma_type = ma_type
        self.vol_length = vol_length
        self.vol_multiplier = vol_multiplier
        self.exit_mode = exit_mode
        self.tp_pct = tp_pct
        self.sl_pct = sl_pct
        self.atr_length = atr_length
        self.atr_tp_mult = atr_tp_mult
        self.atr_sl_mult = atr_sl_mult

    def get_params(self) -> dict:
        """Return current parameters for sweep/optimization."""
        return {
            'ma_length': self.ma_length,
            'ma_type': self.ma_type,
            'vol_length': self.vol_length,
            'vol_multiplier': self.vol_multiplier,
            'exit_mode': self.exit_mode,
            'tp_pct': self.tp_pct,
            'sl_pct': self.sl_pct,
            'atr_length': self.atr_length,
            'atr_tp_mult': self.atr_tp_mult,
            'atr_sl_mult': self.atr_sl_mult,
        }

    def init(self, data: pd.DataFrame) -> None:
        self.data = data.copy()
        close = data['Close']
        high = data['High']
        low = data['Low']
        volume = data['Volume']

        # Moving Average
        if self.ma_type == 'EMA':
            self.ma = close.ewm(span=self.ma_length, adjust=False).mean()
        else:
            self.ma = close.rolling(self.ma_length).mean()

        # Average Volume
        self.avg_volume = volume.rolling(self.vol_length).mean()

        # ATR
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs()
        ], axis=1).max(axis=1)
        self.atr = tr.rolling(self.atr_length).mean()

    def generate_signals(self) -> pd.Series:
        """
        Generate trading signals with state machine logic.

        Entry: price > MA AND volume > avg_volume * multiplier (on signal day)
               Execution on NEXT bar (T+1), simulated by using Open price tracking.
        Exit:  TP/SL based on percentage or ATR from entry price.
        """
        close = self.data['Close']
        high = self.data['High']
        low = self.data['Low']
        open_price = self.data['Open']
        volume = self.data['Volume']

        signals = pd.Series(0.0, index=self.data.index)

        position = 0  # 0=flat, 1=long
        entry_price = None
        entry_atr = None

        for i in range(len(self.data)):
            ma_val = self.ma.iloc[i]
            avg_vol = self.avg_volume.iloc[i]
            atr_val = self.atr.iloc[i]
            cur_close = close.iloc[i]
            cur_high = high.iloc[i]
            cur_low = low.iloc[i]
            cur_open = open_price.iloc[i]
            cur_vol = volume.iloc[i]

            if pd.isna(ma_val) or pd.isna(avg_vol) or pd.isna(atr_val):
                signals.iloc[i] = 0.0
                continue

            # === ENTRY LOGIC ===
            # Signal day: close > MA AND volume surge
            if position == 0:
                price_above_ma = cur_close > ma_val
                volume_breakout = cur_vol > avg_vol * self.vol_multiplier

                if price_above_ma and volume_breakout:
                    position = 1
                    # Use close as proxy for entry price
                    # (backtest engine shifts signals by 1 bar anyway)
                    entry_price = cur_close
                    # Use current ATR (known at signal time)
                    entry_atr = atr_val

            # === EXIT LOGIC ===
            elif position == 1 and entry_price is not None:
                if self.exit_mode == 'atr':
                    tp_price = entry_price + entry_atr * self.atr_tp_mult
                    sl_price = entry_price - entry_atr * self.atr_sl_mult
                else:
                    tp_price = entry_price * (1 + self.tp_pct)
                    sl_price = entry_price * (1 - self.sl_pct)

                # Check TP/SL hit
                if cur_high >= tp_price or cur_low <= sl_price:
                    position = 0
                    entry_price = None
                    entry_atr = None

            signals.iloc[i] = float(position)

        return signals

    def get_description(self) -> str:
        exit_desc = (
            f"ATR x{self.atr_tp_mult}/{self.atr_sl_mult} TP/SL"
            if self.exit_mode == 'atr'
            else f"{self.tp_pct:.0%}/{self.sl_pct:.0%} TP/SL"
        )
        return (
            f"Volume Breakout Strategy: {self.ma_type}({self.ma_length}) trend filter "
            f"+ {self.vol_multiplier}x volume surge entry. "
            f"Exit: {exit_desc}."
        )


# ═══════════════════════════════════════════════════════════════
# Quick test
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    from indicator_pool import get_enriched_data
    from backtest import BacktestEngine

    data = get_enriched_data()
    engine = BacktestEngine(data)

    # Test ATR mode
    strat_atr = ChampionVolumeBreakout(exit_mode='atr')
    strat_atr.init(data)
    signals = strat_atr.generate_signals()
    print(f"Signal distribution: long={(signals > 0).sum()}, "
          f"cash={(signals == 0).sum()}")
    print(f"Time in market: {(signals != 0).mean():.1%}")

    result = engine.run(strat_atr)
    print(f"\nVolume Breakout (ATR mode):")
    print(f"  Sharpe: {result.sharpe_ratio:.2f}")
    print(f"  CAGR: {result.cagr:.1%}")
    print(f"  Max DD: {result.max_drawdown:.1%}")
    print(f"  Calmar: {result.calmar_ratio:.2f}")
    print(f"  Trades: {result.total_trades}")

    # Test percentage mode
    strat_pct = ChampionVolumeBreakout(exit_mode='pct')
    strat_pct.init(data)
    signals_pct = strat_pct.generate_signals()

    result_pct = engine.run(strat_pct)
    print(f"\nVolume Breakout (Percentage mode):")
    print(f"  Sharpe: {result_pct.sharpe_ratio:.2f}")
    print(f"  CAGR: {result_pct.cagr:.1%}")
    print(f"  Max DD: {result_pct.max_drawdown:.1%}")
    print(f"  Calmar: {result_pct.calmar_ratio:.2f}")
    print(f"  Trades: {result_pct.total_trades}")
