"""
AI Researcher Engine
=====================
Uses Gemini to generate, evolve, and improve trading strategies.
"""

import os
import re
import json
import importlib.util
import traceback
from pathlib import Path
from typing import Optional, Tuple, Dict, List
from datetime import datetime
import random

import google.generativeai as genai
from dotenv import load_dotenv
from api_manager import get_api_manager

load_dotenv()

# GitHub Models 作為主力引擎（避免 Gemini rate limit 浪費時間）
_github_client = None

def _get_github_client():
    """取得 GitHub Models MultiModelClient（主力引擎）。"""
    global _github_client
    if _github_client is not None:
        return _github_client
    try:
        from multi_model_client import MultiModelClient
        _github_client = MultiModelClient()
        return _github_client
    except Exception as e:
        print(f"   ⚠️ GitHub Models 初始化失敗: {e}")
        return None


# ═══════════════════════════════════════════════════════════════
# 🗂️ INDICATOR LIBRARY - 擴展 TQQQ 指標宇宙
# ═══════════════════════════════════════════════════════════════
INDICATOR_LIBRARY = {
    "A_TREND": {
        "name": "Trend & Direction (捕捉大波段)",
        "indicators": [
            ("HMA", "Hull Moving Average - 比 SMA/EMA 更快響應"),
            ("Supertrend", "適合強趨勢的追蹤止損"),
            ("Parabolic_SAR", "嚴格的反轉點識別"),
            ("Ichimoku", "Kumo Breakout 趨勢確認"),
            ("TEMA", "Triple EMA - 更平滑的趨勢線"),
        ]
    },
    "B_VOLATILITY": {
        "name": "Volatility & Regime (TQQQ 救命符)",
        "indicators": [
            ("ATR", "Average True Range - 標準化止損"),
            ("BB_Width", "Bollinger Band Width - 偵測 Squeeze"),
            ("Keltner", "Keltner Channels - 突破確認"),
            ("Donchian", "Donchian Channels - 海龜交易法"),
            ("Simulated_VIX", "N日標準差模擬VIX - 高波動時持現金"),
        ]
    },
    "C_MOMENTUM": {
        "name": "Momentum & Oscillators (進出場時機)",
        "indicators": [
            ("Williams_R", "Williams %R - 比 RSI 更敏感"),
            ("Stochastic_RSI", "Stochastic RSI - 震盪市場快速信號"),
            ("CCI", "Commodity Channel Index - 週期轉折"),
            ("MFI", "Money Flow Index - 帶成交量的 RSI"),
            ("ROC", "Rate of Change - 動量變化率"),
        ]
    },
    "D_VOLUME": {
        "name": "Volume & Strength (確認訊號真偽)",
        "indicators": [
            ("OBV", "On-Balance Volume - 價量背離"),
            ("VWMA", "Volume Weighted MA - 成交量加權均線"),
            ("ADX", "Average Directional Index - ADX<20不交易"),
            ("CMF", "Chaikin Money Flow - 資金流向"),
            ("Force_Index", "Force Index - 力量指標"),
        ]
    }
}


class StrategyGenerator:
    """
    AI-powered strategy code generator using Gemini.
    """

    GENERATED_DIR = Path("generated_strategies")
    HISTORY_FILE = Path("history_of_thoughts.json")

    def __init__(self, model_name: str = "gemini-2.5-flash-lite"):
        """Initialize the Gemini model with API Key Manager."""
        self.model_name = model_name
        self.api_manager = get_api_manager()

        # Groq as primary engine (5-key pool allocation)
        self._groq = None
        try:
            from groq_client import GroqClient
            self._groq = GroqClient()
            if not self._groq.keys:
                self._groq = None
        except Exception as e:
            print(f"   ⚠️ Groq init failed in researcher: {e}")

        # Ensure directories exist
        self.GENERATED_DIR.mkdir(exist_ok=True)

        # Load or initialize history
        self.history = self._load_history()

    def _load_history(self) -> Dict:
        """Load the history of thoughts from JSON."""
        if self.HISTORY_FILE.exists():
            with open(self.HISTORY_FILE, 'r') as f:
                return json.load(f)
        return {
            "total_iterations": 0,
            "best_sharpe": 0.0,
            "best_strategy": None,
            "strategies": []
        }

    def _save_history(self) -> None:
        """Save the history of thoughts to JSON."""
        with open(self.HISTORY_FILE, 'w') as f:
            json.dump(self.history, f, indent=2, default=str)

    def get_next_strategy_id(self) -> int:
        """Get the next strategy ID number."""
        return self.history["total_iterations"] + 1

    def _get_used_indicators(self) -> set:
        """取得已使用過的指標。"""
        used = set()
        for s in self.history.get("strategies", [])[-10:]:  # 看最近 10 個
            idea = s.get("idea", "").upper()
            for cat in INDICATOR_LIBRARY.values():
                for ind, _ in cat["indicators"]:
                    if ind.upper() in idea:
                        used.add(ind)
        return used

    def _select_exploration_indicators(self) -> str:
        """
        從指標庫中選擇指標組合。
        規則：必須從至少 2 個不同類別選擇。
        """
        used = self._get_used_indicators()
        categories = list(INDICATOR_LIBRARY.keys())

        # 隨機選 2-3 個類別
        selected_cats = random.sample(categories, min(3, len(categories)))

        selected = []
        for cat_key in selected_cats:
            cat = INDICATOR_LIBRARY[cat_key]
            # 優先選未使用過的指標
            available = [(ind, desc) for ind, desc in cat["indicators"] if ind not in used]
            if not available:
                available = cat["indicators"]

            # 從這個類別選 1 個
            ind, desc = random.choice(available)
            selected.append((cat_key, cat["name"], ind, desc))

        # 構建指標選擇說明
        lines = ["═══════════════════════════════════════════════════════════════",
                 "🎲 MANDATORY INDICATORS FOR THIS GENERATION",
                 "═══════════════════════════════════════════════════════════════",
                 "You MUST use these indicators (from different categories):"]

        for cat_key, cat_name, ind, desc in selected:
            lines.append(f"  • [{cat_key}] {ind}: {desc}")

        lines.append("")
        lines.append("Combine them creatively! Example logic:")

        # 給一個組合範例
        if len(selected) >= 2:
            ind1 = selected[0][2]
            ind2 = selected[1][2]
            lines.append(f"  → Use {ind1} for trend/entry, filter with {ind2} for confirmation")

        return "\n".join(lines)

    def generate_strategy_idea(self) -> str:
        """
        Ask Gemini to propose a new strategy idea based on past results.
        使用模組化思考 + 痛苦回饋機制 + 指標探索。
        """
        # Build context from history
        context = self._build_context()

        # 根據迭代次數選擇演化模式
        iteration = self.history["total_iterations"]
        evolution_mode = self._get_evolution_mode(iteration)

        # 從指標庫選擇必用指標
        indicator_selection = self._select_exploration_indicators()

        prompt = f"""You are a Quantitative Research Director at a hedge fund specializing in leveraged ETFs.

CONTEXT:
{context}

{evolution_mode}

{indicator_selection}

═══════════════════════════════════════════════════════════════
🧬 CHAMPION DNA — PROVEN STRATEGY (Sharpe=0.95, CAGR=43%, MaxDD=-49%)
═══════════════════════════════════════════════════════════════
Our best strategy uses RVI (Relative Volatility Index) with STATE MACHINE:
- 3 States: Green (RVI>59=bull), Orange (42-59=neutral), Red (RVI<42=bear)
- BUY on state TRANSITION: Orange/Red → Green (momentum building)
- SELL: RVI > 76 (overbought) OR RVI < 42 (breakdown)
- SHORT: Orange → Red transition only, ATR×1.8 take-profit/stop-loss
- RVI formula: std=34 (population, ddof=0), smooth EMA=20, refined=(H+L)/2

WHY IT WORKS: TRANSITIONS capture momentum SHIFTS, not static levels.
YOUR TASK: MUTATE one module while keeping the winning pattern.
Example mutations:
  • Change buy_trigger (try 55, 62), sell_low (try 38, 45), atr_factor (1.5-2.2)
  • Add a volume confirmation filter before the transition entry
  • Add a 200-SMA regime gate: only trade longs when Close > SMA(200)
  • Replace ATR TP/SL with Donchian channel or trailing stop
  • Combine RVI states with RSI divergence for higher-conviction entries

═══════════════════════════════════════════════════════════════
🎯 OBJECTIVE: Sharpe ≥ 0.5, CAGR ≥ 5%, MaxDD ≥ -70%
═══════════════════════════════════════════════════════════════

REQUIRED MODULES:
1. STATE MACHINE — define 2-3 market states using indicators
2. TRANSITION-BASED ENTRY — buy on state changes, not thresholds
3. ADAPTIVE EXIT — ATR-based or volatility-adjusted stops
4. DRAWDOWN PROTECTION (MANDATORY) — one of:
   a. ATR volatility filter: don't enter when ATR > N-day 90th percentile
   b. Hard trailing stop: exit when price drops X*ATR from rolling high
   c. Regime gate: only long when Close > SMA(200)
5. OPTIONAL SHORT — state transition to bearish with ATR TP/SL

RULES:
❌ FORBIDDEN: shift(-1), future prices, forward indexing
✅ ALLOWED: SMA, EMA, RSI, ATR, RVI, Bollinger, MACD
- Use INTEGER parameters only
- Signals: -1.0 (short) to 1.0 (long), 0.0 = cash

RESPOND WITH:
1. Strategy Name
2. State Machine Logic (what states, what indicators define them)
3. Entry: Which transitions trigger buy/short
4. Exit & Risk: Adaptive exit + drawdown protection mechanism
5. Key Parameters (integers only)

Keep response concise and actionable."""

        # 主力：Groq (2 keys × multiple models, highest quota)
        result = None
        if self._groq:
            result = self._groq.generate(prompt, task="idea")
        if result is None:
            # 次要：GitHub Models (50 RPD)
            gh = _get_github_client()
            if gh:
                print("   🔄 Groq 不可用，切換到 GitHub Models...")
                result = gh.generate(prompt)
        if result is None:
            # 最終備援：Gemini
            print("   🔄 切換到 Gemini...")
            result = self.api_manager.generate_with_retry(prompt, self.model_name)
        if result is None:
            raise Exception("API 呼叫失敗，Groq、GitHub Models 和 Gemini 都不可用")
        return result

    def generate_strategy_code(self, idea: str, strategy_id: int) -> Tuple[str, str]:
        """
        Ask Gemini to write Python code for the strategy.
        強制模組化設計 + 整數參數。

        Returns:
            Tuple of (code_string, file_path)
        """
        class_name = f"Strategy_Gen{strategy_id}"

        prompt = f"""You are an expert Python developer writing trading strategy code.

STRATEGY IDEA:
{idea}

TASK:
Write a complete Python class implementing this strategy with MODULAR DESIGN.

═══════════════════════════════════════════════════════════════
🧩 REQUIRED MODULAR STRUCTURE
═══════════════════════════════════════════════════════════════
Your code MUST have these 3 separate methods:

1. `_get_regime(self) -> pd.Series`
   Returns: 1 = bullish, 0 = neutral/bearish
   Use: 200-day SMA slope, volatility percentile, etc.

2. `_get_entry_signal(self) -> pd.Series`
   Returns: 1 = buy signal, 0 = no signal
   Use: RSI, MACD, Bollinger, etc.

3. `_get_exit_signal(self) -> pd.Series`
   Returns: 1 = exit signal, 0 = hold
   Use: trailing stop, volatility spike, trend break

Then combine in generate_signals():
   signal = regime * entry * (1 - exit)

═══════════════════════════════════════════════════════════════
🚫 NO LOOK-AHEAD BIAS — STRATEGY WILL BE REJECTED IF VIOLATED
═══════════════════════════════════════════════════════════════
❌ FORBIDDEN (automatic rejection):
  • df.shift(-1) or df.shift(periods=-N)   ← looks into future bar
  • df.pct_change(-1)                       ← tomorrow's return
  • df.diff(-1)                             ← future difference
  • df['Close'].max()  / df['Close'].min()  ← global: uses ALL future data
  • df['Close'].mean() / df['Close'].std()  ← global: uses ALL future data
  • df['Close'].quantile(0.9)              ← global: uses ALL future data
  • rolling(center=True)                   ← symmetric window = future bars
  • Any variable named: tomorrow, next_bar, future_price, look_ahead

✅ ALLOWED (backward-looking only):
  • df.rolling(N).mean() / .std() / .max() ← uses only past N bars
  • df.ewm(span=N).mean()                  ← exponential moving average
  • df.shift(1) or df.shift(N) where N>0   ← delays signal by N bars
  • df.diff(1)                             ← today minus yesterday
  • df.pct_change(1)                       ← today's return (positive period)
  • df.rolling(N).quantile(q)             ← rolling quantile = OK

═══════════════════════════════════════════════════════════════
📐 ANTI-OVERFITTING RULES
═══════════════════════════════════════════════════════════════
- Use INTEGER parameters ONLY: 10, 20, 50, 100, 200
- NO magic numbers like 13.42 or 0.0237
- Maximum 4 conditions per signal

═══════════════════════════════════════════════════════════════
🏆 COPY-PASTE READY: PROVEN RVI FORMULA (Sharpe=0.95, CAGR=43%)
═══════════════════════════════════════════════════════════════
This is the CHAMPION formula — tested and working. Use or mutate it:

def _rvi_single(self, src: pd.Series) -> pd.Series:
    std    = src.rolling(34).std(ddof=0)         # population std, length=34
    change = src.diff()
    up_ema   = std.where(change >= 0, 0.0).ewm(span=20, adjust=False).mean()
    down_ema = std.where(change <  0, 0.0).ewm(span=20, adjust=False).mean()
    return 100.0 * up_ema / (up_ema + down_ema + 1e-9)

def init(self, data):
    self.data = data
    # Refined RVI = average of RVI(High) and RVI(Low)
    self.rvi = (self._rvi_single(data['High']) + self._rvi_single(data['Low'])) / 2
    # ATR
    h, l, c = data['High'], data['Low'], data['Close']
    tr = pd.concat([h-l, (h-c.shift(1)).abs(), (l-c.shift(1)).abs()], axis=1).max(axis=1)
    self.atr = tr.rolling(14).mean()

def generate_signals(self):
    # State machine — TRANSITION-based entry (key insight!)
    rvi = self.rvi
    signals, position, prev = pd.Series(0.0, index=self.data.index), 0.0, 'orange'
    short_px = np.nan
    for i in range(1, len(signals)):
        rv = rvi.iloc[i]
        if np.isnan(rv): signals.iloc[i] = position; continue
        curr = 'green' if rv > 59 else ('red' if rv < 42 else 'orange')
        if position == 1.0 and (rv > 76 or rv < 42): position = 0.0       # exit long
        elif position == -1.0:
            atr_i = self.atr.iloc[i]
            if not np.isnan(short_px) and atr_i > 0:
                if self.data['Low'].iloc[i] <= short_px - atr_i*1.8: position = 0.0; short_px = np.nan  # TP
                if self.data['High'].iloc[i] >= short_px + atr_i*1.8: position = 0.0; short_px = np.nan  # SL
            if position == -1.0 and prev in ('orange','red') and curr=='green': position = 0.0; short_px = np.nan
        if position == 0.0:
            if prev in ('orange','red') and curr=='green': position = 1.0  # BUY on →Green
            elif prev=='orange' and curr=='red':                            # SHORT on Orange→Red
                position = -1.0; short_px = self.data['Close'].iloc[i]
        signals.iloc[i] = position; prev = curr
    return signals

═══════════════════════════════════════════════════════════════
📊 OTHER INDICATOR FORMULAS (copy-paste ready)
═══════════════════════════════════════════════════════════════
# ATR (Average True Range)
tr = pd.concat([high-low, abs(high-close.shift(1)), abs(low-close.shift(1))], axis=1).max(axis=1)
atr = tr.rolling(14).mean()

# HMA (Hull Moving Average)
def hma(series, period):
    half_wma = series.rolling(period//2).mean()
    full_wma = series.rolling(period).mean()
    return (2 * half_wma - full_wma).rolling(int(np.sqrt(period))).mean()

# Bollinger Band Width (squeeze detection)
bb_width = (upper_band - lower_band) / middle_band

# Williams %R
williams_r = (highest_high - close) / (highest_high - lowest_low) * -100

# OBV (On-Balance Volume)
obv = (np.sign(close.diff()) * volume).cumsum()

# ADX simplified proxy (use ATR slope)
adx_proxy = atr.diff(5)  # positive = trending, negative = ranging

═══════════════════════════════════════════════════════════════
📋 CLASS REQUIREMENTS
═══════════════════════════════════════════════════════════════
1. Class name: `{class_name}`
2. Inherit from: `BaseStrategy`
3. Required methods:
   - `init(self, data: pd.DataFrame)` - calculate all indicators
   - `generate_signals(self) -> pd.Series` - return 0.0 to 1.0
   - `get_description(self) -> str` - explain strategy

4. Import EXACTLY: `from strategy_base import BaseStrategy` (NOT `from BaseStrategy import ...`)
5. `__init__` must take NO arguments: `def __init__(self): super().__init__()`
6. Data columns available in self.data: ['Open', 'High', 'Low', 'Close', 'Volume']
   ❌ FORBIDDEN imports: talib, ta, pandas_ta, finta — NOT installed. Use pandas/numpy ONLY.
7. Signals: 0.0 = cash, 1.0 = fully invested, 0-1 for partial
8. Handle NaN: Use .fillna(0) or .bfill() (never forward-fill from future!)
9. INDEX ALIGNMENT (CRITICAL): When wrapping numpy arrays in pd.Series, ALWAYS add index:
   ✅ CORRECT: pd.Series(np.where(...), index=self.data.index)
   ❌ WRONG:   pd.Series(np.where(...))  ← integer index vs datetime index = DOUBLED signal length!
   This applies to: np.where(), np.array(), and any variable created from numpy operations.

EXAMPLE STRUCTURE:
from strategy_base import BaseStrategy
import pandas as pd
import numpy as np

class {class_name}(BaseStrategy):
    def __init__(self):
        super().__init__()

    def init(self, data: pd.DataFrame) -> None:
        self.data = data
        # calculate indicators here

    def _get_regime(self) -> pd.Series: ...
    def _get_entry_signal(self) -> pd.Series: ...
    def _get_exit_signal(self) -> pd.Series: ...

    def generate_signals(self) -> pd.Series:
        regime = self._get_regime()
        entry = self._get_entry_signal()
        exit_signal = self._get_exit_signal()
        return (regime * entry * (1 - exit_signal)).clip(0, 1)

    def get_description(self) -> str:
        return "{class_name}: <brief description>"

⚠️ SELF-CHECK BEFORE SUBMITTING:
- Can your conditions ACTUALLY trigger on real data? (e.g. RSI crosses 50 → YES; RSI == 50.000 exactly → NEVER fires)
- If generate_signals() returns all 0.0, Sharpe=0 and the strategy is USELESS.
- Internal helper methods (e.g. _get_regime) must take ONLY self as argument.
  ❌ WRONG: def _get_regime(self, data)  ← will crash when called as self._get_regime()
  ✅ RIGHT:  def _get_regime(self)        ← accesses self.data set in init()

OUTPUT ONLY PYTHON CODE. NO MARKDOWN, NO EXPLANATIONS, NO ```python TAGS."""

        # 主力：Groq (code task — strong logic models)
        result = None
        if self._groq:
            result = self._groq.generate(prompt, task="code")
        if result is None:
            # 次要：GitHub Models
            gh = _get_github_client()
            if gh:
                print("   🔄 Groq 不可用，切換到 GitHub Models 生成代碼...")
                result = gh.generate(prompt)
        if result is None:
            # 最終備援：Gemini
            print("   🔄 切換到 Gemini 生成代碼...")
            result = self.api_manager.generate_with_retry(prompt, self.model_name)
        if result is None:
            raise Exception("API 呼叫失敗，Groq、GitHub Models 和 Gemini 都不可用")
        code = self._clean_code(result)
        code = self._fix_imports(code)
        code = self._fix_code_structure(code, class_name)

        # Pre-validate syntax before saving — catch obvious LLM errors early
        import ast
        try:
            ast.parse(code)
        except SyntaxError as e:
            print(f"   ⚠️ 代碼語法錯誤，立即嘗試修復: {e}")
            code, _ = self.fix_strategy_code(code, f"SyntaxError: {e}", strategy_id)

        # Save to file
        file_path = self.GENERATED_DIR / f"strategy_gen_{strategy_id}.py"
        with open(file_path, 'w') as f:
            f.write(code)

        return code, str(file_path)

    def fix_strategy_code(self, code: str, error: str, strategy_id: int) -> Tuple[str, str]:
        """
        Ask Gemini to fix broken strategy code.
        """
        class_name = f"Strategy_Gen{strategy_id}"

        # Truncate code to avoid 413 tokens_limit_reached on GitHub Models
        # GitHub Models context limit is ~8K tokens; code + prompt must fit
        MAX_CODE_CHARS = 4000
        code_for_prompt = code if len(code) <= MAX_CODE_CHARS else (
            code[:MAX_CODE_CHARS] + f"\n...(truncated {len(code)-MAX_CODE_CHARS} chars)..."
        )

        # Detect "never enters market" to add targeted fix guidance
        is_tim_error = 'time_in_market' in error or 'never enters' in error or 'Signal stats' in error
        tim_fix_section = ""
        if is_tim_error:
            tim_fix_section = """
7. ⚠️ CRITICAL — SIGNALS ARE ALL ZERO (never enters market):
   DIAGNOSE: Entry condition is too strict or logically always-False.
   FIX strategies (choose one or combine):
   a) RELAX thresholds: RSI>70 → RSI>50, RVI>80 → RVI>55, MA crossover window
   b) ADD rolling baseline: use close > close.rolling(20).mean() instead of fixed levels
   c) ADD multiple OR paths: signal = (condition_a | condition_b).astype(float)
   d) CHECK boolean logic: ensure comparison produces True values on real TQQQ daily data
   VERIFY mentally: on a trending asset, your entry condition should be True ≥10% of days."""

        prompt = f"""You are debugging Python code for a trading strategy.

BROKEN CODE:
```python
{code_for_prompt}
```

ERROR: {error[:400]}

FIX REQUIREMENTS:
1. Class name: `{class_name}`, inherits `BaseStrategy`
2. Methods: init(self, data), generate_signals(self) -> pd.Series [0.0–1.0], get_description(self)
3. INDEX: pd.Series(np_array, index=self.data.index) — NOT pd.Series(np_array)
4. NO external libs: talib, ta, pandas_ta — pandas/numpy only
5. Internal helpers take ONLY self: def _helper(self), NOT def _helper(self, data)
6. NO LOOK-AHEAD BIAS:
   ❌ .shift(-1), .pct_change(-1), .diff(-1) — looks into future
   ❌ data['Close'].max()/.min()/.mean()/.quantile() — global = uses all future data
   ✅ .rolling(N).mean()/.std()/.max()/.quantile(q) — use rolling version instead{tim_fix_section}

OUTPUT ONLY THE FIXED PYTHON CODE. NO MARKDOWN."""

        # 主力：Groq (fix task — fast models)
        result = None
        if self._groq:
            result = self._groq.generate(prompt, task="fix")
        if result is None:
            # 次要：GitHub Models
            gh = _get_github_client()
            if gh:
                print("   🔄 Groq 不可用，切換到 GitHub Models 修復代碼...")
                result = gh.generate(prompt)
        if result is None:
            # 最終備援：Gemini
            print("   🔄 切換到 Gemini 修復代碼...")
            result = self.api_manager.generate_with_retry(prompt, self.model_name)
        if result is None:
            raise Exception("API 呼叫失敗，Groq、GitHub Models 和 Gemini 都不可用")
        code = self._clean_code(result)
        code = self._fix_imports(code)
        code = self._fix_code_structure(code, class_name)

        # Validate syntax of fixed code
        import ast as _ast
        try:
            _ast.parse(code)
        except SyntaxError as e:
            print(f"   ⚠️ 修復後仍有語法錯誤: {e}（會在 sandbox 階段被捕獲）")

        # Save to file
        file_path = self.GENERATED_DIR / f"strategy_gen_{strategy_id}.py"
        with open(file_path, 'w') as f:
            f.write(code)

        return code, str(file_path)

    def _clean_code(self, code: str) -> str:
        """Clean up AI-generated code."""
        # If model returned JSON wrapper like {"code": "..."}, extract the code
        import json as _json
        stripped = code.strip()
        if stripped.startswith('{') and '"code"' in stripped:
            try:
                parsed = _json.loads(stripped)
                if isinstance(parsed, dict):
                    for key in ('code', 'python_code', 'strategy_code', 'content'):
                        if key in parsed and isinstance(parsed[key], str):
                            code = parsed[key]
                            break
            except Exception:
                pass  # not valid JSON, treat as raw code

        # Remove markdown code blocks
        code = re.sub(r'^```python\s*', '', code, flags=re.MULTILINE)
        code = re.sub(r'^```\s*$', '', code, flags=re.MULTILINE)
        code = re.sub(r'```$', '', code)

        # Replace Unicode smart quotes and punctuation that break Python parsing
        unicode_replacements = {
            '\u201c': '"', '\u201d': '"',  # curly double quotes
            '\u2018': "'", '\u2019': "'",  # curly single quotes
            '\u2003': ' ', '\u00a0': ' ',  # em space, non-breaking space
            '\u2013': '-', '\u2014': '-',  # en/em dash
        }
        for bad, good in unicode_replacements.items():
            code = code.replace(bad, good)

        return code.strip()

    def _fix_imports(self, code: str) -> str:
        """
        Aggressively ensure correct imports.
        LLM tends to generate broken variants like:
          - from strategy_base \\nimport pandas  (split across lines)
          - from BaseStrategy import BaseStrategy
          - from strategy_base.BaseStrategy import ...
        Strategy: nuke ALL strategy_base/BaseStrategy import lines, then re-add the correct one.
        """
        # Remove ALL lines that start with 'from strategy_base' (any variant, including split lines)
        code = re.sub(r'from\s+strategy_base\b[^\n]*', '', code, flags=re.MULTILINE)
        # Remove wrong BaseStrategy imports
        code = re.sub(r'from\s+BaseStrategy\b[^\n]*', '', code, flags=re.MULTILINE)
        code = re.sub(r'import\s+BaseStrategy\b[^\n]*', '', code, flags=re.MULTILINE)
        # Remove any orphan 'import' lines that appear to be the continuation of a split import
        code = re.sub(r'^\s*import\s+BaseStrategy\b[^\n]*', '', code, flags=re.MULTILINE)

        # P-021: Remove unavailable TA libraries (talib, ta, pandas_ta not installed)
        # LLM sometimes imports these — replace with nothing so pure pandas/numpy is used
        FORBIDDEN_TA_LIBS = ['talib', 'ta', 'pandas_ta', 'finta', 'stockstats']
        for lib in FORBIDDEN_TA_LIBS:
            code = re.sub(rf'^import\s+{lib}\b[^\n]*', '', code, flags=re.MULTILINE)
            code = re.sub(rf'^from\s+{lib}\b[^\n]*', '', code, flags=re.MULTILINE)
            # Also replace usage like talib.RSI(...) with a comment that prompts fix
            code = re.sub(rf'\b{lib}\.', f'# REMOVED_{lib.upper()}_CALL.', code)

        # Collapse multiple blank lines
        code = re.sub(r'\n{3,}', '\n\n', code).strip()

        # Build canonical header
        header_lines = ["from strategy_base import BaseStrategy"]
        if "import pandas as pd" not in code:
            header_lines.append("import pandas as pd")
        if "import numpy as np" not in code:
            header_lines.append("import numpy as np")

        return "\n".join(header_lines) + "\n\n" + code

    def _fix_code_structure(self, code: str, class_name: str) -> str:
        """
        Fix common structural mistakes in LLM-generated strategy code.
        Called after _fix_imports().
        """
        # Fix: def init(self): → def init(self, data: pd.DataFrame) -> None:
        # LLM sometimes omits the data parameter
        code = re.sub(
            r'def init\s*\(\s*self\s*\)\s*(?:->.*?)?:',
            'def init(self, data: pd.DataFrame) -> None:',
            code
        )

        # Fix: def init(self, data): → def init(self, data: pd.DataFrame) -> None:
        code = re.sub(
            r'def init\s*\(\s*self\s*,\s*data\s*\)\s*(?:->.*?)?:',
            'def init(self, data: pd.DataFrame) -> None:',
            code
        )

        # Fix TypeError: bad operand type for unary ~: 'float'
        # LLM writes `~exit_signal` (bitwise NOT) but signal is float 0.0/1.0.
        # Replace `~(expr)` → `(1 - (expr))` and `~var` → `(1 - var)`
        code = re.sub(r'~\s*\(([^)]+)\)', r'(1 - (\1))', code)
        code = re.sub(r'~\s*([a-zA-Z_]\w*)\b', r'(1 - \1)', code)

        # Fix bitwise OR/AND type errors on floats: `float | series` or `series | float`
        # LLM sometimes writes `position | series` where position is a float — invalid in Python
        # Safe fix: replace `0.0 | ` / `0 | ` (common in signal combination) with logical equivalent
        # Note: this is a heuristic — only catch the most common pattern
        code = re.sub(r'(?<!\w)0\.0\s*\|(?!\|)\s*', '', code)  # `0.0 | x` → `x`
        code = re.sub(r'(?<!\w)0\s*\|(?!\|)\s*(?=[a-zA-Z_(])', '', code)  # `0 | x` → `x`

        # Fix P-019: pd.Series(np_array) without index → datetime index misalignment
        # When a numpy array (from np.where etc.) is wrapped in pd.Series without index,
        # combining it with datetime-indexed data creates union index = doubled length.
        # Fix: pd.Series(var) → pd.Series(var, index=self.data.index)
        # Only for simple variable name arguments (not lists/dicts/complex expressions)
        def _fix_series_index(m):
            args = m.group(1).strip()
            # Skip if already has index=, or is a list literal, or is a complex expression
            if 'index=' in args:
                return m.group(0)
            if args.startswith('[') or args.startswith('{'):
                return m.group(0)  # literal - intentional
            # Only fix simple identifiers (numpy array variables)
            if re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', args):
                return f'pd.Series({args}, index=self.data.index)'
            return m.group(0)

        code = re.sub(r'pd\.Series\(([^)]+)\)', _fix_series_index, code)

        # Detect missing abstract method implementations and append stubs
        has_init = bool(re.search(r'def init\s*\(', code))
        has_generate = bool(re.search(r'def generate_signals\s*\(', code))

        if not has_init:
            stub = (
                f"\n    def init(self, data: pd.DataFrame) -> None:\n"
                f"        self.data = data\n"
            )
            # Insert before class body end (before last line with content)
            code = code.rstrip() + stub

        if not has_generate:
            stub = (
                f"\n    def generate_signals(self) -> pd.Series:\n"
                f"        return pd.Series(0.0, index=self.data.index)\n"
            )
            code = code.rstrip() + stub

        return code

    def _get_evolution_mode(self, iteration: int) -> str:
        """根據迭代次數決定演化模式。"""
        if iteration < 20:
            # 探索期：嘗試各種方向
            return """
═══════════════════════════════════════════════════════════════
🔬 EVOLUTION MODE: EXPLORATION (Iteration {})
═══════════════════════════════════════════════════════════════
Try a DIFFERENT approach from previous strategies.
Explore new indicator combinations and logic patterns.
""".format(iteration + 1)

        elif iteration < 50:
            # 優化期：基於最佳策略改進
            best = self.history.get("best_strategy", "N/A")
            best_sharpe = self.history.get("best_sharpe", 0)
            return f"""
═══════════════════════════════════════════════════════════════
🎯 EVOLUTION MODE: OPTIMIZATION (Iteration {iteration + 1})
═══════════════════════════════════════════════════════════════
Current best: {best} (Sharpe: {best_sharpe:.2f})
Your task: IMPROVE upon the best strategy.
- Keep what works, fix what doesn't
- Focus on reducing MaxDrawdown while maintaining returns
"""
        else:
            # 精煉期：參數微調
            return f"""
═══════════════════════════════════════════════════════════════
🔧 EVOLUTION MODE: REFINEMENT (Iteration {iteration + 1})
═══════════════════════════════════════════════════════════════
Fine-tune the best strategies:
- Adjust parameters (window sizes, thresholds)
- Add small improvements to risk management
- Test slight variations
"""

    def _build_context(self) -> str:
        """Build context string from history for the AI."""
        if self.history["total_iterations"] == 0:
            return """This is the FIRST iteration. No previous strategies have been tested.
Start with a robust baseline strategy that focuses on trend-following and volatility management.
TQQQ is extremely volatile - the 2022 bear market saw >75% drawdown.

CONCEPT INJECTION: Try incorporating Volume Analysis (OBV) or Volatility Targeting
(adjust position size based on current ATR)."""

        # Get last 5 strategies for context
        recent = self.history["strategies"][-5:]

        # 找出品質通過的策略（quality_pass=True）
        # 舊記錄向下相容：用 sharpe>0 and cagr>5% 作為 fallback
        successful = [s for s in self.history["strategies"] if s.get("success")]
        rankable = [
            s for s in successful
            if s.get("quality_pass", (s.get("sharpe", 0) > 0 and s.get("cagr", 0) > 0.05))
        ]
        best_strategies = sorted(rankable, key=lambda x: x.get("composite", x.get("calmar", 0)), reverse=True)[:3]

        best_composite = self.history.get('best_composite', self.history.get('best_calmar', self.history.get('best_sharpe', 0)))
        context_lines = [
            f"Total iterations: {self.history['total_iterations']}",
            f"Best Composite: {best_composite:.4f}",
            f"Best strategy: {self.history['best_strategy']}",
            "",
            "🏆 TOP 3 STRATEGIES (learn from these):"
        ]

        for s in best_strategies[:3]:
            cs = s.get('composite', 0)
            context_lines.append(
                f"  - {s['name']}: Composite={cs:.4f}, Sharpe={s['sharpe']:.2f}, "
                f"CAGR={s.get('cagr', 0):.1%}, MaxDD={s['max_dd']:.1%}"
            )

        context_lines.append("")
        context_lines.append("📉 RECENT ATTEMPTS:")

        for s in recent:
            status = "✅" if s.get("success") else "❌"
            context_lines.append(
                f"  {status} {s['name']}: Sharpe={s['sharpe']:.2f}, MaxDD={s['max_dd']:.1%}"
            )
            if s.get("failure_analysis"):
                context_lines.append(f"      → {s['failure_analysis'][:80]}")

        # 痛苦回饋：找出最常失敗的時期
        context_lines.append("")
        context_lines.append("⚠️ PAIN POINTS (strategies died here):")
        context_lines.append("  - 2022-04: Fed rate hikes caused false breakouts")
        context_lines.append("  - 2020-03: COVID crash - need regime detection")
        context_lines.append("  - 2018-12: Q4 selloff - volatility spike ignored")

        # 概念注入（隨機選一個）
        import random
        concepts = [
            "Try Volume Analysis (OBV, Volume-Weighted MACD) to confirm trends.",
            "Explore Volatility Targeting: adjust position size inversely to ATR.",
            "Consider Dual Momentum: compare TQQQ vs QQQ vs Cash momentum.",
            "Add Mean Reversion filter: avoid buying when RSI > 70.",
            "Use Regime Detection: 200-day SMA slope + VIX level combination.",
        ]
        context_lines.append("")
        context_lines.append(f"💡 CONCEPT TO EXPLORE: {random.choice(concepts)}")

        return "\n".join(context_lines)

    def record_result(
        self,
        strategy_id: int,
        strategy_name: str,
        idea: str,
        sharpe: float,
        cagr: float,
        max_dd: float,
        failure_analysis: str,
        success: bool,
        calmar: float = 0.0,     # 主要優化指標
        quality_pass: bool = False,  # 是否通過品質門檻
        quality_reason: str = ""     # 未通過的原因
    ) -> None:
        """Record strategy result in history."""
        self.history["total_iterations"] += 1

        result = {
            "id": strategy_id,
            "name": strategy_name,
            "idea": idea[:500],  # Truncate long ideas
            "sharpe": sharpe,
            "calmar": calmar,
            "cagr": cagr,
            "max_dd": max_dd,
            "failure_analysis": failure_analysis,
            "success": success,
            "quality_pass": quality_pass,
            "quality_reason": quality_reason,
            "timestamp": datetime.now().isoformat()
        }

        self.history["strategies"].append(result)

        # Update best — only quality-passing strategies qualify
        # Primary metric: Calmar; secondary guard: quality_pass
        if (quality_pass
                and calmar > self.history.get("best_calmar", self.history.get("best_sharpe", 0))):
            self.history["best_sharpe"] = calmar  # 向下相容
            self.history["best_calmar"] = calmar
            self.history["best_strategy"] = strategy_name

        self._save_history()


class StrategySandbox:
    """
    Safely load and execute AI-generated strategy code.
    """

    @staticmethod
    def load_strategy(file_path: str, class_name: str):
        """
        Dynamically load a strategy class from a file.

        Args:
            file_path: Path to the Python file
            class_name: Name of the class to load

        Returns:
            Strategy class instance

        Raises:
            Exception if loading fails
        """
        try:
            spec = importlib.util.spec_from_file_location("dynamic_strategy", file_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            strategy_class = getattr(module, class_name)
            return strategy_class()

        except SyntaxError as e:
            raise Exception(f"Syntax error in generated code: {e}")
        except AttributeError as e:
            raise Exception(f"Class {class_name} not found in generated code: {e}")
        except Exception as e:
            raise Exception(f"Failed to load strategy: {e}")

    @staticmethod
    def test_strategy(strategy, data) -> Tuple[bool, str]:
        """
        Test if a strategy can run without errors.

        Returns:
            Tuple of (success: bool, error_message: str)
        """
        try:
            # Test init
            strategy.init(data)

            # Test signal generation
            signals = strategy.generate_signals()

            # Validate output
            if not isinstance(signals, pd.Series):
                return False, "generate_signals() must return pd.Series"

            if len(signals) != len(data):
                return False, f"Signal length ({len(signals)}) != data length ({len(data)})"

            # Check for NaN
            if signals.isna().all():
                return False, "All signals are NaN"

            # Check for all-zero (strategy never enters market = useless)
            time_in_market = (signals.abs() > 0.01).mean()
            if time_in_market < 0.01:
                sig_min  = float(signals.min())
                sig_max  = float(signals.max())
                sig_mean = float(signals.mean())
                return False, (
                    f"Strategy never enters the market (time_in_market={time_in_market:.3%}). "
                    f"Signal stats: min={sig_min:.4f}, max={sig_max:.4f}, mean={sig_mean:.6f}. "
                    "Entry conditions are too strict or logically impossible. "
                    "FIX: Relax thresholds (e.g., RSI>70→RSI>55, RVI>80→RVI>60) or add "
                    "OR conditions so entries actually trigger on real TQQQ data."
                )

            return True, ""

        except Exception as e:
            return False, f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"


# Import pandas for type hints
import pandas as pd
