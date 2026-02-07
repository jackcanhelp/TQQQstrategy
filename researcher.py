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

import google.generativeai as genai
from dotenv import load_dotenv
from api_manager import get_api_manager

load_dotenv()


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

    def generate_strategy_idea(self) -> str:
        """
        Ask Gemini to propose a new strategy idea based on past results.
        使用模組化思考 + 痛苦回饋機制。
        """
        # Build context from history
        context = self._build_context()

        # 根據迭代次數選擇演化模式
        iteration = self.history["total_iterations"]
        evolution_mode = self._get_evolution_mode(iteration)

        prompt = f"""You are a Quantitative Research Director at a hedge fund specializing in leveraged ETFs.

CONTEXT:
{context}

{evolution_mode}

═══════════════════════════════════════════════════════════════
🎯 PRIMARY OBJECTIVE: SURVIVAL > PROFIT
═══════════════════════════════════════════════════════════════
Your goal is to MAXIMIZE Calmar Ratio (CAGR / |MaxDrawdown|).
Target: Calmar > 1.0 (acceptable returns with controlled drawdowns)

TQQQ can drop 80%+ in bear markets. A strategy that avoids catastrophic
losses is MORE VALUABLE than one with higher returns but deeper drawdowns.

═══════════════════════════════════════════════════════════════
🧩 MODULAR STRATEGY DESIGN (REQUIRED)
═══════════════════════════════════════════════════════════════
Your strategy MUST have these 3 independent modules:

1. 🚦 REGIME FILTER (Market State Detection)
   - Detect: Uptrend / Downtrend / High Volatility / Sideways
   - When bearish or high-vol: MUST go to Cash (0% exposure)
   - Indicators: 200-day SMA slope, VIX level, ATR percentile

2. 🏹 ENTRY SIGNAL (When to Buy)
   - Only trigger when Regime is favorable
   - Focus on: buying dips in uptrends, NOT catching falling knives
   - Indicators: RSI divergence, MACD histogram, Bollinger squeeze

3. 🛡️ EXIT & RISK MANAGEMENT (How to Protect)
   - MUST have trailing stop or volatility-based exit
   - Indicators: Chandelier Exit, ATR trailing stop, % stop-loss

═══════════════════════════════════════════════════════════════
⚠️ CRITICAL RULES - NO LOOK-AHEAD BIAS
═══════════════════════════════════════════════════════════════
❌ FORBIDDEN: shift(-1), future prices, forward indexing
✅ ALLOWED: SMA, EMA, RSI, ATR, Bollinger, MACD (all backward-looking)

═══════════════════════════════════════════════════════════════
📐 AVOID OVERFITTING
═══════════════════════════════════════════════════════════════
- Use INTEGER parameters only (10, 20, 50, 200 - not 13.42)
- Logic must be explainable in 3 sentences
- Maximum 4 conditions for entry/exit

═══════════════════════════════════════════════════════════════
💡 KEY INSIGHT: CASH IS A POSITION
═══════════════════════════════════════════════════════════════
For TQQQ, holding Cash (0 exposure) is POWERFUL due to volatility decay.
Good strategies sit out during sideways/choppy markets.

RESPOND WITH:
1. Strategy Name
2. Regime Filter Logic (when to be in cash)
3. Entry Signal Logic (when to buy)
4. Exit & Risk Logic (when to sell/protect)
5. Key Parameters (integers only)

Keep response concise and actionable."""

        result = self.api_manager.generate_with_retry(prompt, self.model_name)
        if result is None:
            raise Exception("API 呼叫失敗，所有 Key 都不可用")
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

        result = self.api_manager.generate_with_retry(prompt, self.model_name)
        if result is None:
            raise Exception("API 呼叫失敗，所有 Key 都不可用")
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

        result = self.api_manager.generate_with_retry(prompt, self.model_name)
        if result is None:
            raise Exception("API 呼叫失敗，所有 Key 都不可用")
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
        best_strategies = sorted(successful, key=lambda x: x.get("sharpe", 0), reverse=True)[:3]

        context_lines = [
            f"Total iterations: {self.history['total_iterations']}",
            f"Best Calmar/Sharpe: {self.history['best_sharpe']:.2f}",
            f"Best strategy: {self.history['best_strategy']}",
            "",
            "🏆 TOP 3 STRATEGIES (learn from these):"
        ]

        for s in best_strategies[:3]:
            context_lines.append(
                f"  - {s['name']}: Sharpe={s['sharpe']:.2f}, CAGR={s.get('cagr', 0):.1%}, MaxDD={s['max_dd']:.1%}"
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

        # Update best if applicable (use Calmar as primary metric)
        # 如果 calmar 有值就用 calmar，否則用 sharpe 向後兼容
        metric = calmar if calmar > 0 else sharpe
        if metric > self.history["best_sharpe"]:
            self.history["best_sharpe"] = metric
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
