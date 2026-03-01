"""
RVI Champion Strategy — Python translation of the proven Pine Script strategy.
Modified T30 RVI Strategy with Conditional Short Selling and Volatility Adjustment.

Parameters (match Pine Script defaults):
  stdev_length = 34, smooth_length = 20
  buy_trigger = 59, sell_high = 76, sell_low = 42
  atr_factor = 1.8, atr_period = 14

State Machine:
  Green : RVI > 59 (bull)
  Orange: 42 <= RVI <= 59 (neutral)
  Red   : RVI < 42 (bear)

Entry rules (transition-based):
  LONG  : prev_state in (Orange, Red)  AND curr_state == Green
  SHORT : prev_state == Orange          AND curr_state == Red

Exit rules:
  Long  : RVI > 76 (overbought) OR RVI < 42 (breakdown)
  Short : Low  <= entry_close - ATR*1.8 (take profit)
          High >= entry_close + ATR*1.8 (stop loss)
          OR buy signal fires
"""
from strategy_base import BaseStrategy
import pandas as pd
import numpy as np


class Strategy_RVI_Champion(BaseStrategy):
    # ── Parameters ─────────────────────────────────────────────
    STDEV_LEN   = 34
    SMOOTH_LEN  = 20
    BUY_TRIGGER = 59   # orange/red → green triggers long
    SELL_HIGH   = 76   # overbought exit
    SELL_LOW    = 42   # bear zone: exit long / enter/stay short
    ATR_PERIOD  = 14
    ATR_FACTOR  = 1.8  # short TP/SL multiplier

    def __init__(self):
        super().__init__()

    # ── RVI Calculation ────────────────────────────────────────
    def _rvi_single(self, src: pd.Series) -> pd.Series:
        """
        RVI for one price series (High or Low).
        Pine Script equivalent:
            upSum   = ta.ema(ta.change(src) >= 0 ? stdev : 0, smooth_length)
            downSum = ta.ema(ta.change(src) >= 0 ? 0 : stdev, smooth_length)
            100 * upSum / (upSum + downSum)
        """
        std    = src.rolling(self.STDEV_LEN).std(ddof=0)   # population std
        change = src.diff()
        up_std   = std.where(change >= 0, 0.0)
        down_std = std.where(change <  0, 0.0)
        up_ema   = up_std.ewm(span=self.SMOOTH_LEN, adjust=False).mean()
        down_ema = down_std.ewm(span=self.SMOOTH_LEN, adjust=False).mean()
        return 100.0 * up_ema / (up_ema + down_ema + 1e-9)

    def init(self, data: pd.DataFrame) -> None:
        self.data = data

        # Refined RVI = (RVI(High) + RVI(Low)) / 2
        self.rvi = (
            self._rvi_single(data['High']) +
            self._rvi_single(data['Low'])
        ) / 2.0

        # ATR (14)
        h, l, c = data['High'], data['Low'], data['Close']
        tr = pd.concat(
            [h - l,
             (h - c.shift(1)).abs(),
             (l - c.shift(1)).abs()],
            axis=1
        ).max(axis=1)
        self.atr = tr.rolling(self.ATR_PERIOD).mean()

    def generate_signals(self) -> pd.Series:
        """
        Processes day-by-day (no lookahead): signal on day i is based only
        on data up to day i. BacktestEngine shifts by 1 to avoid lookahead.
        """
        rvi   = self.rvi
        high  = self.data['High']
        low   = self.data['Low']
        close = self.data['Close']
        atr   = self.atr

        signals          = pd.Series(0.0, index=self.data.index)
        position         = 0.0     # 0=cash, 1=long, -1=short
        short_entry_px   = np.nan
        prev_state       = 'orange'  # conservative start

        for i in range(1, len(signals)):
            rv = rvi.iloc[i]
            if np.isnan(rv):
                signals.iloc[i] = position
                continue

            # ── Current state ───────────────────────────────
            if rv > self.BUY_TRIGGER:
                curr = 'green'
            elif rv < self.SELL_LOW:
                curr = 'red'
            else:
                curr = 'orange'

            h_i   = high.iloc[i]
            l_i   = low.iloc[i]
            atr_i = atr.iloc[i] if not np.isnan(atr.iloc[i]) else 0.0

            # ── Manage open long ────────────────────────────
            if position == 1.0:
                if rv > self.SELL_HIGH or rv < self.SELL_LOW:
                    position = 0.0

            # ── Manage open short ───────────────────────────
            elif position == -1.0:
                # ATR take-profit / stop-loss
                if not np.isnan(short_entry_px) and atr_i > 0:
                    tp = short_entry_px - atr_i * self.ATR_FACTOR
                    sl = short_entry_px + atr_i * self.ATR_FACTOR
                    if l_i <= tp or h_i >= sl:
                        position       = 0.0
                        short_entry_px = np.nan
                # Exit short when buy signal fires
                if position == -1.0 and prev_state in ('orange', 'red') and curr == 'green':
                    position       = 0.0
                    short_entry_px = np.nan

            # ── Open new positions (flat only) ──────────────
            if position == 0.0:
                if prev_state in ('orange', 'red') and curr == 'green':
                    position = 1.0                                # LONG
                elif prev_state == 'orange' and curr == 'red':
                    position       = -1.0                         # SHORT
                    short_entry_px = close.iloc[i]                # record entry price

            signals.iloc[i] = position
            prev_state = curr

        return signals

    def get_description(self) -> str:
        return (
            "RVI Champion: Refined RVI=(RVI_H+RVI_L)/2, std=34, smooth=20. "
            "State machine: Green(>59)/Orange/Red(<42). "
            "LONG on Orange/Red→Green; exit on RVI>76 or <42. "
            "SHORT on Orange→Red; exit on ATR×1.8 TP/SL or buy signal."
        )
