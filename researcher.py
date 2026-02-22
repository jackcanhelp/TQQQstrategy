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
🧬 CHAMPION DNA — PROVEN STRATEGY (Sharpe=1.28)
═══════════════════════════════════════════════════════════════
Our best strategy uses RVI (Relative Volatility Index) with STATE MACHINE:
- 3 States: Green (RVI>59=bull), Orange (neutral), Red (RVI<42=bear)
- BUY on state TRANSITION: Orange/Red → Green (momentum building)
- SELL: RVI > 76 (overbought) or RVI < 42 (breakdown)
- SHORT: Orange → Red transition, ATR×1.8 take-profit/stop-loss

WHY IT WORKS: Transitions capture MOMENTUM SHIFTS, not static levels.
YOUR TASK: MUTATE one module while keeping the winning pattern.

═══════════════════════════════════════════════════════════════
🎯 OBJECTIVE: Beat Sharpe=1.28 AND MaxDD > -30%
═══════════════════════════════════════════════════════════════

REQUIRED MODULES:
1. STATE MACHINE — define 2-3 market states using indicators
2. TRANSITION-BASED ENTRY — buy on state changes, not thresholds
3. ADAPTIVE EXIT — ATR-based or volatility-adjusted stops
4. OPTIONAL SHORT — state transition to bearish with TP/SL

RULES:
❌ FORBIDDEN: shift(-1), future prices, forward indexing
✅ ALLOWED: SMA, EMA, RSI, ATR, RVI, Bollinger, MACD
- Use INTEGER parameters only
- Signals: -1.0 (short) to 1.0 (long), 0.0 = cash

RESPOND WITH:
1. Strategy Name
2. State Machine Logic (what states, what indicators define them)
3. Entry: Which transitions trigger buy/short
4. Exit & Risk: Adaptive exit conditions
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
                result = gh._call_model_chain(prompt)
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
⚠️ NO LOOK-AHEAD BIAS (CRITICAL)
═══════════════════════════════════════════════════════════════
❌ FORBIDDEN: df.shift(-1), df.iloc[i+1], future data
✅ ALLOWED: df.rolling(20).mean(), df.shift(1), backward-looking only

═══════════════════════════════════════════════════════════════
📐 ANTI-OVERFITTING RULES
═══════════════════════════════════════════════════════════════
- Use INTEGER parameters ONLY: 10, 20, 50, 100, 200
- NO magic numbers like 13.42 or 0.0237
- Maximum 4 conditions per signal

═══════════════════════════════════════════════════════════════
📊 INDICATOR IMPLEMENTATION GUIDE
═══════════════════════════════════════════════════════════════
Common indicator formulas (copy-paste ready):

# HMA (Hull Moving Average)
def hma(series, period):
    half_wma = series.rolling(period//2).mean()
    full_wma = series.rolling(period).mean()
    return (2 * half_wma - full_wma).rolling(int(np.sqrt(period))).mean()

# ATR (Average True Range)
tr = pd.concat([high-low, abs(high-close.shift(1)), abs(low-close.shift(1))], axis=1).max(axis=1)
atr = tr.rolling(14).mean()

# ADX (for regime filter: ADX < 20 = no trend = cash)
# Simplified: Use ATR slope as proxy

# Bollinger Band Width (squeeze detection)
bb_width = (upper_band - lower_band) / middle_band

# Williams %R
williams_r = (highest_high - close) / (highest_high - lowest_low) * -100

# OBV (On-Balance Volume)
obv = (np.sign(close.diff()) * volume).cumsum()

# Supertrend (simplified)
upper = (high + low) / 2 + 2 * atr
lower = (high + low) / 2 - 2 * atr

═══════════════════════════════════════════════════════════════
📋 CLASS REQUIREMENTS
═══════════════════════════════════════════════════════════════
1. Class name: `{class_name}`
2. Inherit from: `BaseStrategy`
3. Required methods:
   - `init(self, data: pd.DataFrame)` - calculate all indicators
   - `generate_signals(self) -> pd.Series` - return 0.0 to 1.0
   - `get_description(self) -> str` - explain strategy

4. Available: pandas as pd, numpy as np, BaseStrategy
5. Data columns: ['Open', 'High', 'Low', 'Close', 'Volume']
6. Signals: 0.0 = cash, 1.0 = fully invested, 0-1 for partial

7. Handle NaN: Use .fillna(0) or .bfill() (never forward-fill from future!)

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
                result = gh._call_model_chain(prompt)
        if result is None:
            # 最終備援：Gemini
            print("   🔄 切換到 Gemini 生成代碼...")
            result = self.api_manager.generate_with_retry(prompt, self.model_name)
        if result is None:
            raise Exception("API 呼叫失敗，Groq、GitHub Models 和 Gemini 都不可用")
        code = self._clean_code(result)

        # Add imports if missing
        if "from strategy_base import BaseStrategy" not in code:
            code = "from strategy_base import BaseStrategy\nimport pandas as pd\nimport numpy as np\n\n" + code

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

        prompt = f"""You are debugging Python code for a trading strategy.

BROKEN CODE:
```python
{code}
```

ERROR MESSAGE:
{error}

TASK:
Fix the code so it runs without errors.

REQUIREMENTS:
1. The class must be named exactly: `{class_name}`
2. It must inherit from `BaseStrategy`
3. Must implement init(), generate_signals(), get_description()
4. generate_signals() must return pd.Series with values 0.0 to 1.0

OUTPUT ONLY THE FIXED PYTHON CODE. NO MARKDOWN, NO EXPLANATIONS."""

        # 主力：Groq (fix task — fast models)
        result = None
        if self._groq:
            result = self._groq.generate(prompt, task="fix")
        if result is None:
            # 次要：GitHub Models
            gh = _get_github_client()
            if gh:
                print("   🔄 Groq 不可用，切換到 GitHub Models 修復代碼...")
                result = gh._call_model_chain(prompt)
        if result is None:
            # 最終備援：Gemini
            print("   🔄 切換到 Gemini 修復代碼...")
            result = self.api_manager.generate_with_retry(prompt, self.model_name)
        if result is None:
            raise Exception("API 呼叫失敗，Groq、GitHub Models 和 Gemini 都不可用")
        code = self._clean_code(result)

        # Add imports if missing
        if "from strategy_base import BaseStrategy" not in code:
            code = "from strategy_base import BaseStrategy\nimport pandas as pd\nimport numpy as np\n\n" + code

        # Save to file
        file_path = self.GENERATED_DIR / f"strategy_gen_{strategy_id}.py"
        with open(file_path, 'w') as f:
            f.write(code)

        return code, str(file_path)

    def _clean_code(self, code: str) -> str:
        """Clean up AI-generated code."""
        # Remove markdown code blocks
        code = re.sub(r'^```python\s*', '', code, flags=re.MULTILINE)
        code = re.sub(r'^```\s*$', '', code, flags=re.MULTILINE)
        code = re.sub(r'```$', '', code)

        # Remove any leading/trailing whitespace
        code = code.strip()

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

        # 找出成功的策略
        successful = [s for s in self.history["strategies"] if s.get("success")]
        # Filter out "do nothing" strategies (Sharpe <= 0 or CAGR <= 5%)
        rankable = [s for s in successful if s.get("sharpe", 0) > 0 and s.get("cagr", 0) > 0.05]
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
        calmar: float = 0.0  # 主要優化指標
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
            "timestamp": datetime.now().isoformat()
        }

        self.history["strategies"].append(result)

        # Update best — use Calmar as primary ranking metric
        # Filter: must have Sharpe > 0 and CAGR > 5% to qualify (no "do nothing" strategies)
        if (calmar > self.history.get("best_calmar", self.history.get("best_sharpe", 0))
                and sharpe > 0.0 and cagr > 0.05):
            self.history["best_sharpe"] = calmar  # 向下相容：欄位名保留但存 Calmar
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

            return True, ""

        except Exception as e:
            return False, f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"


# Import pandas for type hints
import pandas as pd
