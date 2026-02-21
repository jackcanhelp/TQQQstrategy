"""
Strategy Crossover Engine
=========================
Takes top-performing strategies, extracts their modular components
(regime filter, entry signal, exit logic), and recombines them
into new hybrid strategies.

Unlike ensemble (which averages signals), crossover creates NEW
strategies by mixing modules from different parents.
"""

import json
import itertools
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

from strategy_base import BaseStrategy
from backtest import BacktestEngine, BacktestResult


class ModularStrategy(BaseStrategy):
    """
    A strategy assembled from independent modules (regime, entry, exit).
    Each module is a callable that takes the data DataFrame and returns a pd.Series.
    """

    def __init__(
        self,
        regime_fn,
        entry_fn,
        exit_fn,
        short_fn=None,
        name_parts: Tuple[str, str, str] = ("", "", ""),
    ):
        super().__init__()
        self._regime_fn = regime_fn
        self._entry_fn = entry_fn
        self._exit_fn = exit_fn
        self._short_fn = short_fn
        self.name = f"Cross_{name_parts[0]}_{name_parts[1]}_{name_parts[2]}"

    def init(self, data: pd.DataFrame) -> None:
        self.data = data

    def generate_signals(self) -> pd.Series:
        regime = self._regime_fn(self.data)   # 1=favorable, 0=unfavorable
        entry = self._entry_fn(self.data)     # 1=buy signal, 0=no
        exit_sig = self._exit_fn(self.data)   # 1=exit, 0=hold

        signals = pd.Series(0.0, index=self.data.index)
        position = 0

        for i in range(len(self.data)):
            r = regime.iloc[i] if not pd.isna(regime.iloc[i]) else 0
            e = entry.iloc[i] if not pd.isna(entry.iloc[i]) else 0
            x = exit_sig.iloc[i] if not pd.isna(exit_sig.iloc[i]) else 0

            # Long entry: regime favorable + entry signal
            if r > 0 and e > 0 and position <= 0:
                position = 1

            # Long exit: exit signal or regime turns unfavorable
            if position > 0 and (x > 0 or r <= 0):
                position = 0

            # Short logic (optional)
            if self._short_fn is not None:
                short_sig = self._short_fn(self.data).iloc[i]
                if not pd.isna(short_sig) and short_sig < 0 and position == 0:
                    position = -1
                elif position < 0 and (r > 0 or e > 0):
                    position = 0

            signals.iloc[i] = float(position)

        return signals

    def get_description(self) -> str:
        return f"Crossover strategy: {self.name}"


# ═══════════════════════════════════════════════════════════════
# Module Library — Reusable building blocks extracted from proven strategies
# ═══════════════════════════════════════════════════════════════

def _safe_get(data, col, default=0.0):
    """Safely get column from data, return default Series if missing."""
    if col in data.columns:
        return data[col].fillna(default)
    return pd.Series(default, index=data.index)


# ─── REGIME MODULES ───

def regime_rvi(data: pd.DataFrame) -> pd.Series:
    """RVI-based regime: favorable when RVI_Refined > 50."""
    rvi = _safe_get(data, 'RVI_Refined', 50)
    return (rvi > 50).astype(float)


def regime_rvi_state(data: pd.DataFrame) -> pd.Series:
    """RVI State regime: favorable when state >= 0 (not bearish)."""
    state = _safe_get(data, 'RVI_State', 0)
    return (state >= 0).astype(float)


def regime_sma_trend(data: pd.DataFrame) -> pd.Series:
    """SMA trend regime: favorable when Close > SMA_200 and SMA_50 > SMA_200."""
    close = data['Close']
    sma50 = _safe_get(data, 'SMA_50')
    sma200 = _safe_get(data, 'SMA_200')
    return ((close > sma200) & (sma50 > sma200)).astype(float)


def regime_adx_trend(data: pd.DataFrame) -> pd.Series:
    """ADX regime: favorable when ADX > 20 (trending) and Close > SMA_50."""
    adx = _safe_get(data, 'ADX', 0)
    close = data['Close']
    sma50 = _safe_get(data, 'SMA_50')
    return ((adx > 20) & (close > sma50)).astype(float)


def regime_volatility(data: pd.DataFrame) -> pd.Series:
    """Volatility regime: favorable when Sim_VIX < 40 (not extreme vol)."""
    vix = _safe_get(data, 'Sim_VIX', 20)
    sma200 = _safe_get(data, 'SMA_200')
    close = data['Close']
    return ((vix < 40) & (close > sma200)).astype(float)


def regime_supertrend(data: pd.DataFrame) -> pd.Series:
    """Supertrend regime: favorable when Supertrend = 1."""
    st = _safe_get(data, 'Supertrend', 0)
    return (st > 0).astype(float)


def regime_ichimoku(data: pd.DataFrame) -> pd.Series:
    """Ichimoku regime: favorable when Close > Ichimoku base line."""
    close = data['Close']
    base = _safe_get(data, 'Ichimoku_base')
    conv = _safe_get(data, 'Ichimoku_conv')
    return ((close > base) & (conv > base)).astype(float)


def regime_vol_trend(data: pd.DataFrame) -> pd.Series:
    """Volume-confirmed trend regime: Close > SMA_20 AND volume above average."""
    close = data['Close']
    sma20 = _safe_get(data, 'SMA_20')
    vol_ratio = _safe_get(data, 'Vol_Ratio', 1.0)
    return ((close > sma20) & (vol_ratio > 1.0)).astype(float)


def regime_aroon(data: pd.DataFrame) -> pd.Series:
    """Aroon regime: uptrend when Aroon Oscillator > 50."""
    aroon_osc = _safe_get(data, 'Aroon_Osc', 0)
    return (aroon_osc > 50).astype(float)


def regime_di_trend(data: pd.DataFrame) -> pd.Series:
    """DI regime: +DI > -DI AND ADX > 20 (strong directional trend)."""
    di_diff = _safe_get(data, 'DI_Diff', 0)
    adx = _safe_get(data, 'ADX', 0)
    return ((di_diff > 0) & (adx > 20)).astype(float)


def regime_low_vol(data: pd.DataFrame) -> pd.Series:
    """Low volatility regime: ATR% below median AND no BB squeeze."""
    atr_pct = _safe_get(data, 'ATR_Pct', 3.0)
    bb_squeeze = _safe_get(data, 'BB_Squeeze', 0)
    close = data['Close']
    sma50 = _safe_get(data, 'SMA_50')
    return ((atr_pct < atr_pct.rolling(50).median()) & (close > sma50)).astype(float)


def regime_drawdown(data: pd.DataFrame) -> pd.Series:
    """Drawdown regime: only trade when drawdown is shallow (> -20%)."""
    dd = _safe_get(data, 'Drawdown', 0)
    return (dd > -0.20).astype(float)


# ─── ENTRY MODULES ───

def entry_rvi_transition(data: pd.DataFrame) -> pd.Series:
    """RVI transition entry: buy when RVI_State transitions to bullish."""
    state = _safe_get(data, 'RVI_State', 0)
    prev_state = state.shift(1).fillna(0)
    # Transition: was 0 or -1, now 1
    return ((prev_state <= 0) & (state > 0)).astype(float)


def entry_rsi_bounce(data: pd.DataFrame) -> pd.Series:
    """RSI bounce entry: buy when RSI crosses above 40 from below."""
    rsi = _safe_get(data, 'RSI', 50)
    prev_rsi = rsi.shift(1).fillna(50)
    return ((prev_rsi < 40) & (rsi >= 40)).astype(float)


def entry_macd_crossover(data: pd.DataFrame) -> pd.Series:
    """MACD crossover entry: buy when MACD crosses above signal line."""
    macd = _safe_get(data, 'MACD', 0)
    signal = _safe_get(data, 'MACD_signal', 0)
    prev_macd = macd.shift(1).fillna(0)
    prev_signal = signal.shift(1).fillna(0)
    return ((prev_macd <= prev_signal) & (macd > signal)).astype(float)


def entry_bb_squeeze(data: pd.DataFrame) -> pd.Series:
    """Bollinger squeeze entry: buy when BB_width contracts then expands upward."""
    width = _safe_get(data, 'BB_width', 0.1)
    bb_pct = _safe_get(data, 'BB_pct', 0.5)
    prev_width = width.shift(1).fillna(0.1)
    # Width expanding (squeeze releasing) AND price in upper half of bands
    return ((width > prev_width) & (bb_pct > 0.5) & (width.shift(5) < width)).astype(float)


def entry_stoch_cross(data: pd.DataFrame) -> pd.Series:
    """Stochastic crossover entry: buy when %K crosses above %D from oversold."""
    k = _safe_get(data, 'Stoch_K', 50)
    d = _safe_get(data, 'Stoch_D', 50)
    prev_k = k.shift(1).fillna(50)
    return ((prev_k < d.shift(1).fillna(50)) & (k > d) & (k < 50)).astype(float)


def entry_obv_breakout(data: pd.DataFrame) -> pd.Series:
    """OBV breakout entry: buy when OBV crosses above its SMA (volume confirmation)."""
    obv = _safe_get(data, 'OBV', 0)
    obv_sma = _safe_get(data, 'OBV_SMA', 0)
    prev_obv = obv.shift(1).fillna(0)
    prev_sma = obv_sma.shift(1).fillna(0)
    return ((prev_obv <= prev_sma) & (obv > obv_sma)).astype(float)


def entry_supertrend_flip(data: pd.DataFrame) -> pd.Series:
    """Supertrend flip entry: buy when Supertrend flips from 0 to 1."""
    st = _safe_get(data, 'Supertrend', 0)
    prev_st = st.shift(1).fillna(0)
    return ((prev_st <= 0) & (st > 0)).astype(float)


def entry_vol_breakout(data: pd.DataFrame) -> pd.Series:
    """Volume breakout entry: buy when volume surges 2x+ AND price above SMA_20."""
    close = data['Close']
    sma20 = _safe_get(data, 'SMA_20')
    vol_ratio = _safe_get(data, 'Vol_Ratio', 1.0)
    prev_vol_ratio = vol_ratio.shift(1).fillna(1.0)
    return ((prev_vol_ratio < 2.0) & (vol_ratio >= 2.0) & (close > sma20)).astype(float)


def entry_aroon_cross(data: pd.DataFrame) -> pd.Series:
    """Aroon crossover entry: buy when Aroon Up crosses above Aroon Down."""
    aroon_up = _safe_get(data, 'Aroon_Up', 0)
    aroon_down = _safe_get(data, 'Aroon_Down', 0)
    prev_up = aroon_up.shift(1).fillna(0)
    prev_down = aroon_down.shift(1).fillna(0)
    return ((prev_up <= prev_down) & (aroon_up > aroon_down)).astype(float)


def entry_zscore_bounce(data: pd.DataFrame) -> pd.Series:
    """Z-Score mean reversion entry: buy when Z-Score crosses from <-1.5 to >-1.5."""
    zscore = _safe_get(data, 'ZScore', 0)
    prev_z = zscore.shift(1).fillna(0)
    return ((prev_z < -1.5) & (zscore >= -1.5)).astype(float)


def entry_tsi_cross(data: pd.DataFrame) -> pd.Series:
    """TSI crossover entry: buy when TSI crosses from negative to positive."""
    tsi = _safe_get(data, 'TSI', 0)
    prev_tsi = tsi.shift(1).fillna(0)
    return ((prev_tsi < 0) & (tsi >= 0)).astype(float)


def entry_squeeze_fire(data: pd.DataFrame) -> pd.Series:
    """BB Squeeze breakout entry: buy when squeeze releases AND price moves up."""
    bb_squeeze = _safe_get(data, 'BB_Squeeze', 0)
    prev_squeeze = bb_squeeze.shift(1).fillna(0)
    close = data['Close']
    sma20 = _safe_get(data, 'SMA_20')
    return ((prev_squeeze == 1) & (bb_squeeze == 0) & (close > sma20)).astype(float)


def entry_elder_bull(data: pd.DataFrame) -> pd.Series:
    """Elder Ray entry: buy when Bull Power crosses positive while Bear Power improves."""
    bull = _safe_get(data, 'Elder_Bull', 0)
    prev_bull = bull.shift(1).fillna(0)
    bear = _safe_get(data, 'Elder_Bear', 0)
    prev_bear = bear.shift(1).fillna(0)
    return ((prev_bull <= 0) & (bull > 0) & (bear > prev_bear)).astype(float)


def entry_ppo_cross(data: pd.DataFrame) -> pd.Series:
    """PPO crossover entry: buy when PPO crosses above signal line."""
    ppo = _safe_get(data, 'PPO', 0)
    ppo_sig = _safe_get(data, 'PPO_signal', 0)
    prev_ppo = ppo.shift(1).fillna(0)
    prev_sig = ppo_sig.shift(1).fillna(0)
    return ((prev_ppo <= prev_sig) & (ppo > ppo_sig)).astype(float)


# ─── EXIT MODULES ───

def exit_rvi_extreme(data: pd.DataFrame) -> pd.Series:
    """RVI extreme exit: sell when RVI > 76 (overbought) or < 42 (breakdown)."""
    rvi = _safe_get(data, 'RVI_Refined', 50)
    return ((rvi > 76) | (rvi < 42)).astype(float)


def exit_rsi_overbought(data: pd.DataFrame) -> pd.Series:
    """RSI overbought exit: sell when RSI > 75."""
    rsi = _safe_get(data, 'RSI', 50)
    return (rsi > 75).astype(float)


def exit_sma_death_cross(data: pd.DataFrame) -> pd.Series:
    """SMA death cross exit: sell when SMA_50 < SMA_200."""
    sma50 = _safe_get(data, 'SMA_50')
    sma200 = _safe_get(data, 'SMA_200')
    return (sma50 < sma200).astype(float)


def exit_atr_volatility_spike(data: pd.DataFrame) -> pd.Series:
    """ATR volatility spike exit: sell when ATR > 2x its 50-day average."""
    atr = _safe_get(data, 'ATR', 1)
    atr_avg = atr.rolling(50).mean().fillna(atr)
    return (atr > 2.0 * atr_avg).astype(float)


def exit_macd_bearish(data: pd.DataFrame) -> pd.Series:
    """MACD bearish exit: sell when MACD histogram turns negative."""
    hist = _safe_get(data, 'MACD_hist', 0)
    prev_hist = hist.shift(1).fillna(0)
    return ((prev_hist > 0) & (hist < 0)).astype(float)


def exit_bb_lower_break(data: pd.DataFrame) -> pd.Series:
    """BB lower break exit: sell when Close drops below BB lower band."""
    close = data['Close']
    bb_lower = _safe_get(data, 'BB_lower')
    return (close < bb_lower).astype(float)


def exit_supertrend_down(data: pd.DataFrame) -> pd.Series:
    """Supertrend down exit: sell when Supertrend flips to 0."""
    st = _safe_get(data, 'Supertrend', 1)
    prev_st = st.shift(1).fillna(1)
    return ((prev_st > 0) & (st <= 0)).astype(float)


def exit_zscore_extreme(data: pd.DataFrame) -> pd.Series:
    """Z-Score exit: sell when price is extremely overbought (Z > 2.0)."""
    zscore = _safe_get(data, 'ZScore', 0)
    return (zscore > 2.0).astype(float)


def exit_aroon_flip(data: pd.DataFrame) -> pd.Series:
    """Aroon exit: sell when Aroon Down crosses above Aroon Up."""
    aroon_up = _safe_get(data, 'Aroon_Up', 100)
    aroon_down = _safe_get(data, 'Aroon_Down', 0)
    prev_up = aroon_up.shift(1).fillna(100)
    prev_down = aroon_down.shift(1).fillna(0)
    return ((prev_up > prev_down) & (aroon_up <= aroon_down)).astype(float)


def exit_tsi_negative(data: pd.DataFrame) -> pd.Series:
    """TSI exit: sell when TSI crosses from positive to negative."""
    tsi = _safe_get(data, 'TSI', 0)
    prev_tsi = tsi.shift(1).fillna(0)
    return ((prev_tsi >= 0) & (tsi < 0)).astype(float)


def exit_drawdown_limit(data: pd.DataFrame) -> pd.Series:
    """Drawdown exit: sell when drawdown exceeds -15%."""
    dd = _safe_get(data, 'Drawdown', 0)
    return (dd < -0.15).astype(float)


def exit_consec_down(data: pd.DataFrame) -> pd.Series:
    """Consecutive down days exit: sell after 4+ consecutive down days."""
    days_down = _safe_get(data, 'Days_Down', 0)
    return (days_down >= 4).astype(float)


def exit_pct_tp_sl(data: pd.DataFrame) -> pd.Series:
    """Percentage TP/SL exit: sell when price moves 10% up or 5% down from recent entry."""
    close = data['Close']
    # Use rolling max of recent 20 bars as proxy for entry price area
    rolling_high = close.rolling(20).max()
    rolling_entry = close.rolling(20).apply(
        lambda x: x.iloc[0] if len(x) > 0 else x.iloc[-1], raw=False
    )
    pct_change = (close - rolling_entry) / rolling_entry
    return ((pct_change > 0.10) | (pct_change < -0.05)).astype(float)


# ═══════════════════════════════════════════════════════════════
# Module Registry
# ═══════════════════════════════════════════════════════════════

REGIME_MODULES = {
    "RVI": (regime_rvi, "RVI > 50 regime filter"),
    "RVI_State": (regime_rvi_state, "RVI State not bearish"),
    "SMA_Trend": (regime_sma_trend, "Close > SMA_200 + golden cross"),
    "ADX_Trend": (regime_adx_trend, "ADX > 20 trending + above SMA_50"),
    "Volatility": (regime_volatility, "Low volatility + above SMA_200"),
    "Supertrend": (regime_supertrend, "Supertrend bullish"),
    "Ichimoku": (regime_ichimoku, "Above Ichimoku cloud"),
    "Vol_Trend": (regime_vol_trend, "Close > SMA_20 + above-avg volume"),
    "Aroon": (regime_aroon, "Aroon Oscillator > 50 uptrend"),
    "DI_Trend": (regime_di_trend, "+DI > -DI with ADX > 20"),
    "Low_Vol": (regime_low_vol, "Below-median ATR% + above SMA_50"),
    "No_DD": (regime_drawdown, "Drawdown > -20% (shallow)"),
}

ENTRY_MODULES = {
    "RVI_Trans": (entry_rvi_transition, "RVI state transition to bullish"),
    "RSI_Bounce": (entry_rsi_bounce, "RSI crosses above 40"),
    "MACD_Cross": (entry_macd_crossover, "MACD crosses above signal"),
    "BB_Squeeze": (entry_bb_squeeze, "Bollinger squeeze release upward"),
    "Stoch_Cross": (entry_stoch_cross, "Stochastic %K crosses %D from oversold"),
    "OBV_Break": (entry_obv_breakout, "OBV crosses above SMA"),
    "ST_Flip": (entry_supertrend_flip, "Supertrend flips bullish"),
    "Vol_Surge": (entry_vol_breakout, "Volume surges 2x+ with price above SMA"),
    "Aroon_Cross": (entry_aroon_cross, "Aroon Up crosses above Aroon Down"),
    "ZScore_Bounce": (entry_zscore_bounce, "Z-Score bounces from oversold <-1.5"),
    "TSI_Cross": (entry_tsi_cross, "TSI crosses from negative to positive"),
    "Squeeze_Fire": (entry_squeeze_fire, "BB squeeze releases with price above SMA"),
    "Elder_Bull": (entry_elder_bull, "Elder Bull Power turns positive"),
    "PPO_Cross": (entry_ppo_cross, "PPO crosses above signal line"),
}

EXIT_MODULES = {
    "RVI_Extreme": (exit_rvi_extreme, "RVI > 76 or < 42"),
    "RSI_OB": (exit_rsi_overbought, "RSI > 75"),
    "SMA_Death": (exit_sma_death_cross, "SMA_50 < SMA_200 death cross"),
    "ATR_Spike": (exit_atr_volatility_spike, "ATR > 2x average"),
    "MACD_Bear": (exit_macd_bearish, "MACD histogram turns negative"),
    "BB_Break": (exit_bb_lower_break, "Close below BB lower band"),
    "ST_Down": (exit_supertrend_down, "Supertrend flips bearish"),
    "Pct_TPSL": (exit_pct_tp_sl, "10% TP / 5% SL from entry area"),
    "ZScore_OB": (exit_zscore_extreme, "Z-Score > 2.0 overbought exit"),
    "Aroon_Flip": (exit_aroon_flip, "Aroon Down crosses above Aroon Up"),
    "TSI_Neg": (exit_tsi_negative, "TSI crosses from positive to negative"),
    "DD_Limit": (exit_drawdown_limit, "Drawdown exceeds -15%"),
    "Consec_Down": (exit_consec_down, "4+ consecutive down days"),
}


# ═══════════════════════════════════════════════════════════════
# Crossover Engine
# ═══════════════════════════════════════════════════════════════

CROSSOVER_RESULTS_FILE = Path("crossover_results.json")


def run_crossover(
    data: pd.DataFrame,
    regime_keys: Optional[List[str]] = None,
    entry_keys: Optional[List[str]] = None,
    exit_keys: Optional[List[str]] = None,
    top_n: int = 10,
) -> List[Dict]:
    """
    Test all permutations of regime × entry × exit modules.

    Args:
        data: Enriched OHLCV DataFrame
        regime_keys: Which regime modules to test (None = all)
        entry_keys: Which entry modules to test (None = all)
        exit_keys: Which exit modules to test (None = all)
        top_n: Return top N results

    Returns:
        List of top results with module combination and metrics
    """
    if regime_keys is None:
        regime_keys = list(REGIME_MODULES.keys())
    if entry_keys is None:
        entry_keys = list(ENTRY_MODULES.keys())
    if exit_keys is None:
        exit_keys = list(EXIT_MODULES.keys())

    combos = list(itertools.product(regime_keys, entry_keys, exit_keys))
    total = len(combos)

    print(f"═══════════════════════════════════════════")
    print(f"🧬 Strategy Crossover: {total} combinations")
    print(f"   Regime modules: {regime_keys}")
    print(f"   Entry modules: {entry_keys}")
    print(f"   Exit modules: {exit_keys}")
    print(f"═══════════════════════════════════════════")

    engine = BacktestEngine(data)
    results = []

    for idx, (r_key, e_key, x_key) in enumerate(combos):
        try:
            r_fn, _ = REGIME_MODULES[r_key]
            e_fn, _ = ENTRY_MODULES[e_key]
            x_fn, _ = EXIT_MODULES[x_key]

            strategy = ModularStrategy(
                regime_fn=r_fn,
                entry_fn=e_fn,
                exit_fn=x_fn,
                name_parts=(r_key, e_key, x_key),
            )

            strategy.init(data)
            signals = strategy.generate_signals()

            # Skip if all cash
            if (signals == 0).all():
                continue

            bt = engine.run(strategy)

            results.append({
                "regime": r_key,
                "entry": e_key,
                "exit": x_key,
                "name": strategy.name,
                "sharpe": bt.sharpe_ratio,
                "cagr": bt.cagr,
                "max_dd": bt.max_drawdown,
                "calmar": bt.calmar_ratio,
                "sortino": bt.sortino_ratio,
                "trades": bt.total_trades,
                "time_in_market": bt.time_in_market,
            })

        except Exception as e:
            continue

        if (idx + 1) % 50 == 0:
            if results:
                best = max(results, key=lambda r: r["sharpe"])
                print(f"   [{idx+1}/{total}] Best: {best['name']} "
                      f"Sharpe={best['sharpe']:.2f}")

    if not results:
        print("   ❌ No valid crossover results")
        return []

    results.sort(key=lambda r: r["sharpe"], reverse=True)
    top = results[:top_n]

    print(f"\n🏆 Top {min(top_n, len(top))} Crossover Strategies:")
    for i, r in enumerate(top[:10], 1):
        print(f"   #{i} [{r['regime']}×{r['entry']}×{r['exit']}] "
              f"Sharpe={r['sharpe']:.2f}, CAGR={r['cagr']:.1%}, "
              f"MaxDD={r['max_dd']:.1%}, Calmar={r['calmar']:.2f}")

    # Save results
    _save_crossover_results(top)

    return top


def _save_crossover_results(results: List[Dict]):
    """Save crossover results to JSON."""
    existing = []
    if CROSSOVER_RESULTS_FILE.exists():
        with open(CROSSOVER_RESULTS_FILE) as f:
            existing = json.load(f)

    existing.append({
        "timestamp": datetime.now().isoformat(),
        "total_tested": len(results),
        "results": results,
    })

    with open(CROSSOVER_RESULTS_FILE, 'w') as f:
        json.dump(existing, f, indent=2, default=str)


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    from indicator_pool import get_enriched_data

    data = get_enriched_data()
    results = run_crossover(data)
