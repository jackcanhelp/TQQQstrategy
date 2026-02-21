"""
Indicator Pool — Pre-calculated Technical Indicator Data Layer
================================================================
Downloads TQQQ data and pre-calculates 30+ technical indicators.
Provides enriched DataFrame and indicator metadata for LLM strategy generation.
"""

import os
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf
import ta


CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)


# ═══════════════════════════════════════════════════════════════
# Indicator Registry — metadata for LLM prompts
# ═══════════════════════════════════════════════════════════════
INDICATOR_REGISTRY = {
    # --- Trend ---
    "SMA_20":    {"category": "Trend",      "column": "SMA_20",    "desc": "20-day Simple Moving Average"},
    "SMA_50":    {"category": "Trend",      "column": "SMA_50",    "desc": "50-day Simple Moving Average"},
    "SMA_200":   {"category": "Trend",      "column": "SMA_200",   "desc": "200-day Simple Moving Average"},
    "EMA_20":    {"category": "Trend",      "column": "EMA_20",    "desc": "20-day Exponential Moving Average"},
    "EMA_50":    {"category": "Trend",      "column": "EMA_50",    "desc": "50-day Exponential Moving Average"},
    "EMA_200":   {"category": "Trend",      "column": "EMA_200",   "desc": "200-day Exponential Moving Average"},
    "MACD":      {"category": "Trend",      "column": "MACD",      "desc": "MACD line (12,26)"},
    "MACD_signal": {"category": "Trend",    "column": "MACD_signal", "desc": "MACD signal line (9)"},
    "MACD_hist": {"category": "Trend",      "column": "MACD_hist", "desc": "MACD histogram"},
    "ADX":       {"category": "Trend",      "column": "ADX",       "desc": "Average Directional Index (14). ADX<20=no trend"},
    "Supertrend":{"category": "Trend",      "column": "Supertrend","desc": "Supertrend direction: 1=up, 0=down"},
    "Ichimoku_conv": {"category": "Trend",  "column": "Ichimoku_conv", "desc": "Ichimoku Conversion Line (Tenkan-sen)"},
    "Ichimoku_base": {"category": "Trend",  "column": "Ichimoku_base", "desc": "Ichimoku Base Line (Kijun-sen)"},

    # --- Momentum ---
    "RSI":       {"category": "Momentum",   "column": "RSI",       "desc": "RSI (14). >70=overbought, <30=oversold"},
    "Stoch_K":   {"category": "Momentum",   "column": "Stoch_K",   "desc": "Stochastic %K (14,3)"},
    "Stoch_D":   {"category": "Momentum",   "column": "Stoch_D",   "desc": "Stochastic %D (3)"},
    "Williams_R":{"category": "Momentum",   "column": "Williams_R","desc": "Williams %R (14). -80=oversold, -20=overbought"},
    "CCI":       {"category": "Momentum",   "column": "CCI",       "desc": "Commodity Channel Index (20)"},
    "ROC":       {"category": "Momentum",   "column": "ROC",       "desc": "Rate of Change (10)"},
    "MFI":       {"category": "Momentum",   "column": "MFI",       "desc": "Money Flow Index (14). RSI with Volume"},

    # --- Volatility ---
    "ATR":       {"category": "Volatility", "column": "ATR",       "desc": "Average True Range (14)"},
    "BB_upper":  {"category": "Volatility", "column": "BB_upper",  "desc": "Bollinger Band upper (20,2)"},
    "BB_middle": {"category": "Volatility", "column": "BB_middle", "desc": "Bollinger Band middle (SMA 20)"},
    "BB_lower":  {"category": "Volatility", "column": "BB_lower",  "desc": "Bollinger Band lower (20,2)"},
    "BB_width":  {"category": "Volatility", "column": "BB_width",  "desc": "Bollinger Band Width (squeeze detector)"},
    "BB_pct":    {"category": "Volatility", "column": "BB_pct",    "desc": "Bollinger %B position within bands (0-1)"},
    "KC_upper":  {"category": "Volatility", "column": "KC_upper",  "desc": "Keltner Channel upper (20)"},
    "KC_lower":  {"category": "Volatility", "column": "KC_lower",  "desc": "Keltner Channel lower (20)"},
    "Donchian_upper": {"category": "Volatility", "column": "Donchian_upper", "desc": "Donchian Channel upper (20)"},
    "Donchian_lower": {"category": "Volatility", "column": "Donchian_lower", "desc": "Donchian Channel lower (20)"},
    "Sim_VIX":   {"category": "Volatility", "column": "Sim_VIX",   "desc": "Simulated VIX (20-day std of returns * sqrt(252))"},

    # --- RVI (Relative Volatility Index) ---
    "RVI":       {"category": "Momentum",   "column": "RVI",       "desc": "Relative Volatility Index (34,20). Measures volatility direction: >59=bullish, <42=bearish"},
    "RVI_Refined":{"category": "Momentum",  "column": "RVI_Refined","desc": "Refined RVI: avg of RVI(High) and RVI(Low). Smoother than single-source RVI"},
    "RVI_State":  {"category": "Momentum",  "column": "RVI_State",  "desc": "RVI State: 1=green(bull), 0=orange(neutral), -1=red(bear). Use state transitions for signals"},

    # --- Volume ---
    "OBV":       {"category": "Volume",     "column": "OBV",       "desc": "On-Balance Volume"},
    "OBV_SMA":   {"category": "Volume",     "column": "OBV_SMA",   "desc": "OBV 20-day SMA (for divergence)"},
    "CMF":       {"category": "Volume",     "column": "CMF",       "desc": "Chaikin Money Flow (20)"},
    "Force_Index":{"category": "Volume",    "column": "Force_Index","desc": "Force Index EMA (13)"},
    "Vol_Ratio": {"category": "Volume",     "column": "Vol_Ratio", "desc": "Volume / 20-day avg volume ratio"},
    "VWAP_Ratio":{"category": "Volume",     "column": "VWAP_Ratio","desc": "Close / rolling VWAP ratio. >1=above VWAP, <1=below"},

    # --- Trend Quality ---
    "DI_Plus":   {"category": "Trend",      "column": "DI_Plus",   "desc": "Positive Directional Indicator (+DI). Rising=strong uptrend"},
    "DI_Minus":  {"category": "Trend",      "column": "DI_Minus",  "desc": "Negative Directional Indicator (-DI). Rising=strong downtrend"},
    "DI_Diff":   {"category": "Trend",      "column": "DI_Diff",   "desc": "+DI minus -DI. >0=bullish dominance, <0=bearish"},
    "Aroon_Up":  {"category": "Trend",      "column": "Aroon_Up",  "desc": "Aroon Up (25). 100=new high within window, 0=no high"},
    "Aroon_Down":{"category": "Trend",      "column": "Aroon_Down","desc": "Aroon Down (25). 100=new low within window"},
    "Aroon_Osc": {"category": "Trend",      "column": "Aroon_Osc", "desc": "Aroon Oscillator (Up-Down). >0=uptrend, <0=downtrend"},
    "TRIX":      {"category": "Trend",      "column": "TRIX",      "desc": "Triple EMA oscillator (15). Smooth trend with momentum"},
    "PPO":       {"category": "Momentum",   "column": "PPO",       "desc": "Percentage Price Oscillator (12,26). Normalized MACD"},
    "PPO_signal":{"category": "Momentum",   "column": "PPO_signal","desc": "PPO signal line (9)"},
    "SMA_10":    {"category": "Trend",      "column": "SMA_10",    "desc": "10-day SMA (short-term trend)"},

    # --- Mean Reversion ---
    "ZScore":    {"category": "MeanReversion","column": "ZScore",   "desc": "Price Z-Score (20). Distance from mean in std devs. >2=overbought, <-2=oversold"},
    "SMA50_Dist":{"category": "MeanReversion","column": "SMA50_Dist","desc": "% distance from SMA_50. >0.1=extended up, <-0.1=extended down"},
    "SMA200_Dist":{"category": "MeanReversion","column": "SMA200_Dist","desc": "% distance from SMA_200. Measures long-term extension"},
    "RSI_7":     {"category": "Momentum",   "column": "RSI_7",     "desc": "Fast RSI (7). More responsive than RSI(14)"},

    # --- Volatility Regime ---
    "ATR_Pct":   {"category": "Volatility", "column": "ATR_Pct",   "desc": "ATR / Close as %. Normalized volatility across price levels"},
    "HV_10":     {"category": "Volatility", "column": "HV_10",     "desc": "10-day Historical Volatility (annualized). Short-term vol"},
    "HV_30":     {"category": "Volatility", "column": "HV_30",     "desc": "30-day Historical Volatility (annualized). Medium-term vol"},
    "BB_Squeeze":{"category": "Volatility", "column": "BB_Squeeze","desc": "BB inside KC squeeze. 1=squeeze(low vol, expect breakout), 0=no squeeze"},
    "VoV":       {"category": "Volatility", "column": "VoV",       "desc": "Volatility of Volatility: std of ATR_Pct(10). High=unstable regime"},

    # --- Market Structure ---
    "Drawdown":  {"category": "Structure",  "column": "Drawdown",  "desc": "Current drawdown from rolling peak. 0=at high, -0.5=50% down"},
    "Days_Up":   {"category": "Structure",  "column": "Days_Up",   "desc": "Consecutive up-close days. >5=strong momentum, reset on down day"},
    "Days_Down": {"category": "Structure",  "column": "Days_Down", "desc": "Consecutive down-close days. >5=strong selling, reset on up day"},
    "Gap_Pct":   {"category": "Structure",  "column": "Gap_Pct",   "desc": "Overnight gap: (Open - prev Close) / prev Close. Large gaps signal events"},
    "ROC_5":     {"category": "Momentum",   "column": "ROC_5",     "desc": "5-day Rate of Change. Short-term momentum"},
    "ROC_20":    {"category": "Momentum",   "column": "ROC_20",    "desc": "20-day Rate of Change. Medium-term momentum"},

    # --- Advanced Momentum ---
    "TSI":       {"category": "Momentum",   "column": "TSI",       "desc": "True Strength Index (25,13). Double-smoothed momentum. >0=bullish"},
    "Elder_Bull":{"category": "Momentum",   "column": "Elder_Bull","desc": "Elder Ray Bull Power: High - EMA(13). >0=bulls dominate"},
    "Elder_Bear":{"category": "Momentum",   "column": "Elder_Bear","desc": "Elder Ray Bear Power: Low - EMA(13). <0=bears dominate"},
    "AO":        {"category": "Momentum",   "column": "AO",        "desc": "Awesome Oscillator (5,34). Momentum histogram. Cross zero=trend change"},
    "UO":        {"category": "Momentum",   "column": "UO",        "desc": "Ultimate Oscillator (7,14,28). Multi-timeframe momentum. >70=OB, <30=OS"},
}


def download_tqqq(force_refresh: bool = False) -> pd.DataFrame:
    """Download or load cached TQQQ OHLCV data."""
    cache_file = CACHE_DIR / "TQQQ_raw.pkl"

    if cache_file.exists() and not force_refresh:
        cache_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
        if cache_age.days < 1:
            data = pd.read_pickle(cache_file)
            print(f"📊 Loaded cached TQQQ: {len(data)} rows")
            return data

    print("📥 Downloading TQQQ data...")
    ticker = yf.Ticker("TQQQ")
    data = ticker.history(period="max", auto_adjust=True)
    data = data[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
    data.to_pickle(cache_file)
    print(f"✅ Downloaded {len(data)} rows ({data.index[0].date()} → {data.index[-1].date()})")
    return data


def calculate_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate 30+ technical indicators on OHLCV data.
    Returns enriched DataFrame with all indicator columns.
    """
    df = data.copy()
    close = df['Close']
    high = df['High']
    low = df['Low']
    volume = df['Volume']

    print("🔧 Calculating indicators...")

    # ── Trend ──
    df['SMA_20']  = ta.trend.sma_indicator(close, window=20)
    df['SMA_50']  = ta.trend.sma_indicator(close, window=50)
    df['SMA_200'] = ta.trend.sma_indicator(close, window=200)
    df['EMA_20']  = ta.trend.ema_indicator(close, window=20)
    df['EMA_50']  = ta.trend.ema_indicator(close, window=50)
    df['EMA_200'] = ta.trend.ema_indicator(close, window=200)

    macd = ta.trend.MACD(close, window_slow=26, window_fast=12, window_sign=9)
    df['MACD']        = macd.macd()
    df['MACD_signal'] = macd.macd_signal()
    df['MACD_hist']   = macd.macd_diff()

    df['ADX'] = ta.trend.adx(high, low, close, window=14)

    # Supertrend (simplified: use STC direction as proxy)
    atr_val = ta.volatility.average_true_range(high, low, close, window=10)
    hl2 = (high + low) / 2
    upper = hl2 + 2 * atr_val
    lower = hl2 - 2 * atr_val
    df['Supertrend'] = (close > lower.shift(1)).astype(int)

    ichimoku = ta.trend.IchimokuIndicator(high, low, window1=9, window2=26, window3=52)
    df['Ichimoku_conv'] = ichimoku.ichimoku_conversion_line()
    df['Ichimoku_base'] = ichimoku.ichimoku_base_line()

    # ── Momentum ──
    df['RSI']       = ta.momentum.rsi(close, window=14)
    stoch = ta.momentum.StochasticOscillator(high, low, close, window=14, smooth_window=3)
    df['Stoch_K']   = stoch.stoch()
    df['Stoch_D']   = stoch.stoch_signal()
    df['Williams_R'] = ta.momentum.williams_r(high, low, close, lbp=14)
    df['CCI']       = ta.trend.cci(high, low, close, window=20)
    df['ROC']       = ta.momentum.roc(close, window=10)
    df['MFI']       = ta.volume.money_flow_index(high, low, close, volume, window=14)

    # ── Volatility ──
    df['ATR'] = ta.volatility.average_true_range(high, low, close, window=14)

    bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
    df['BB_upper']  = bb.bollinger_hband()
    df['BB_middle'] = bb.bollinger_mavg()
    df['BB_lower']  = bb.bollinger_lband()
    df['BB_width']  = (df['BB_upper'] - df['BB_lower']) / df['BB_middle']
    df['BB_pct']    = bb.bollinger_pband()

    kc = ta.volatility.KeltnerChannel(high, low, close, window=20)
    df['KC_upper'] = kc.keltner_channel_hband()
    df['KC_lower'] = kc.keltner_channel_lband()

    dc = ta.volatility.DonchianChannel(high, low, close, window=20)
    df['Donchian_upper'] = dc.donchian_channel_hband()
    df['Donchian_lower'] = dc.donchian_channel_lband()

    # Simulated VIX
    df['Sim_VIX'] = close.pct_change().rolling(20).std() * np.sqrt(252) * 100

    # ── RVI (Relative Volatility Index) ──
    def _calc_rvi(src, stdev_len=34, smooth_len=20):
        stdev = src.rolling(stdev_len).std()
        change = src.diff()
        up_vol = stdev.where(change >= 0, 0.0)
        down_vol = stdev.where(change < 0, 0.0)
        up_sum = up_vol.ewm(span=smooth_len, adjust=False).mean()
        down_sum = down_vol.ewm(span=smooth_len, adjust=False).mean()
        return 100.0 * up_sum / (up_sum + down_sum)

    df['RVI'] = _calc_rvi(close)
    df['RVI_Refined'] = (_calc_rvi(high) + _calc_rvi(low)) / 2.0
    # RVI State: 1=green(bullish), 0=orange(neutral), -1=red(bearish)
    df['RVI_State'] = 0
    df.loc[df['RVI_Refined'] > 59, 'RVI_State'] = 1
    df.loc[df['RVI_Refined'] < 42, 'RVI_State'] = -1

    # ── Volume ──
    df['OBV'] = ta.volume.on_balance_volume(close, volume)
    df['OBV_SMA'] = df['OBV'].rolling(20).mean()
    df['CMF'] = ta.volume.chaikin_money_flow(high, low, close, volume, window=20)
    df['Force_Index'] = ta.volume.force_index(close, volume, window=13)
    df['Vol_Ratio'] = volume / volume.rolling(20).mean()

    # ── Trend Quality ──
    df['DI_Plus'] = ta.trend.adx_pos(high, low, close, window=14)
    df['DI_Minus'] = ta.trend.adx_neg(high, low, close, window=14)
    df['DI_Diff'] = df['DI_Plus'] - df['DI_Minus']

    aroon = ta.trend.AroonIndicator(high=high, low=low, window=25)
    df['Aroon_Up'] = aroon.aroon_up()
    df['Aroon_Down'] = aroon.aroon_down()
    df['Aroon_Osc'] = df['Aroon_Up'] - df['Aroon_Down']

    df['TRIX'] = ta.trend.trix(close, window=15)

    df['SMA_10'] = ta.trend.sma_indicator(close, window=10)

    # PPO (Percentage Price Oscillator = normalized MACD)
    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    df['PPO'] = ((ema_12 - ema_26) / ema_26) * 100
    df['PPO_signal'] = df['PPO'].ewm(span=9, adjust=False).mean()

    # ── Mean Reversion ──
    rolling_mean = close.rolling(20).mean()
    rolling_std = close.rolling(20).std()
    df['ZScore'] = (close - rolling_mean) / rolling_std
    df['SMA50_Dist'] = (close - df['SMA_50']) / df['SMA_50']
    df['SMA200_Dist'] = (close - df['SMA_200']) / df['SMA_200']
    df['RSI_7'] = ta.momentum.rsi(close, window=7)

    # ── Volatility Regime ──
    df['ATR_Pct'] = df['ATR'] / close * 100
    df['HV_10'] = close.pct_change().rolling(10).std() * np.sqrt(252) * 100
    df['HV_30'] = close.pct_change().rolling(30).std() * np.sqrt(252) * 100
    # BB Squeeze: Bollinger Bands inside Keltner Channel = low volatility
    df['BB_Squeeze'] = ((df['BB_lower'] > df['KC_lower']) & (df['BB_upper'] < df['KC_upper'])).astype(int)
    df['VoV'] = df['ATR_Pct'].rolling(10).std()

    # ── Market Structure ──
    df['Drawdown'] = close / close.cummax() - 1.0
    # Consecutive up/down days
    up_close = (close > close.shift(1)).astype(int)
    down_close = (close < close.shift(1)).astype(int)
    days_up = pd.Series(0, index=close.index)
    days_down = pd.Series(0, index=close.index)
    for i in range(1, len(close)):
        if up_close.iloc[i]:
            days_up.iloc[i] = days_up.iloc[i-1] + 1
        if down_close.iloc[i]:
            days_down.iloc[i] = days_down.iloc[i-1] + 1
    df['Days_Up'] = days_up
    df['Days_Down'] = days_down
    df['Gap_Pct'] = (df['Open'] - close.shift(1)) / close.shift(1)
    df['ROC_5'] = ta.momentum.roc(close, window=5)
    df['ROC_20'] = ta.momentum.roc(close, window=20)

    # ── Advanced Momentum ──
    df['TSI'] = ta.momentum.tsi(close, window_slow=25, window_fast=13)
    ema_13 = close.ewm(span=13, adjust=False).mean()
    df['Elder_Bull'] = high - ema_13
    df['Elder_Bear'] = low - ema_13
    df['AO'] = ta.momentum.awesome_oscillator(high, low, window1=5, window2=34)
    df['UO'] = ta.momentum.ultimate_oscillator(high, low, close, window1=7, window2=14, window3=28)

    # ── Volume: VWAP ratio ──
    typical_price = (high + low + close) / 3
    vwap_cum = (typical_price * volume).rolling(20).sum() / volume.rolling(20).sum()
    df['VWAP_Ratio'] = close / vwap_cum

    # ── Derived ──
    df['Returns'] = close.pct_change()
    df['SMA_50_slope'] = (df['SMA_50'] - df['SMA_50'].shift(5)) / df['SMA_50'].shift(5)
    df['SMA_200_slope'] = (df['SMA_200'] - df['SMA_200'].shift(5)) / df['SMA_200'].shift(5)

    indicator_count = len([c for c in df.columns if c not in data.columns and c != 'Returns'])
    print(f"   ✅ {indicator_count} indicators calculated")

    return df


def get_enriched_data(force_refresh: bool = False) -> pd.DataFrame:
    """Get TQQQ data with all indicators pre-calculated. Cached."""
    cache_file = CACHE_DIR / "TQQQ_enriched.pkl"

    if cache_file.exists() and not force_refresh:
        cache_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
        if cache_age.days < 1:
            df = pd.read_pickle(cache_file)
            print(f"📊 Loaded cached enriched data: {len(df)} rows, {len(df.columns)} columns")
            return df

    raw = download_tqqq(force_refresh)
    enriched = calculate_indicators(raw)
    enriched.to_pickle(cache_file)
    return enriched


def get_indicator_menu() -> str:
    """
    Generate a formatted indicator menu string for LLM prompts.
    Groups indicators by category with descriptions.
    """
    categories = {}
    for name, info in INDICATOR_REGISTRY.items():
        cat = info["category"]
        if cat not in categories:
            categories[cat] = []
        categories[cat].append((info["column"], info["desc"]))

    lines = [
        "═══════════════════════════════════════════",
        "📊 AVAILABLE INDICATORS (pre-calculated)",
        "═══════════════════════════════════════════",
        "All columns are available in self.data DataFrame.",
        "Use them directly: self.data['RSI'], self.data['ATR'], etc.",
        ""
    ]

    for cat, indicators in categories.items():
        lines.append(f"── {cat} ──")
        for col, desc in indicators:
            lines.append(f"  • {col}: {desc}")
        lines.append("")

    lines.append("── Price & Volume (always available) ──")
    lines.append("  • Open, High, Low, Close, Volume, Returns")
    lines.append("")
    lines.append("IMPORTANT: All indicators are pre-calculated.")
    lines.append("Do NOT recalculate them. Just reference the column names above.")

    return "\n".join(lines)


def get_indicator_summary(data: pd.DataFrame) -> str:
    """Generate a summary of current indicator values (latest row) for LLM context."""
    latest = data.iloc[-1]
    lines = ["Current market snapshot:"]
    for name, info in INDICATOR_REGISTRY.items():
        col = info["column"]
        if col in data.columns:
            val = latest[col]
            if pd.notna(val):
                lines.append(f"  {col}: {val:.2f}")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
# CLI test
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    df = get_enriched_data(force_refresh=True)
    print(f"\nDataFrame shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nLast 3 rows sample:")
    print(df[['Close', 'RSI', 'ATR', 'MACD', 'ADX', 'Sim_VIX']].tail(3))
    print(f"\n{get_indicator_menu()}")
