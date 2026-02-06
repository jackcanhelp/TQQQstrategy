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
        """
        # Build context from history
        context = self._build_context()

        prompt = f"""You are a Quantitative Research Director at a hedge fund.

CONTEXT:
{context}

YOUR TASK:
Propose ONE new trading strategy idea for TQQQ (3x leveraged NASDAQ-100 ETF).

CRITICAL RULES - MUST FOLLOW:
⚠️ NO LOOK-AHEAD BIAS: You can ONLY use data available UP TO the current day.
   - ❌ FORBIDDEN: Using future prices, future volatility, future returns
   - ❌ FORBIDDEN: Knowing which days will be bad before they happen
   - ❌ FORBIDDEN: Any indicator that uses data from day T+1 or later
   - ✅ ALLOWED: Moving averages, RSI, ATR, historical volatility (all backward-looking)

REQUIREMENTS:
- Strategy must be DIFFERENT from previously tried strategies
- Focus on RISK MANAGEMENT (TQQQ can drop 80%+ in bear markets)
- Use ONLY backward-looking indicators: SMA, EMA, RSI, ATR, Bollinger Bands, MACD, historical volatility
- All decisions on day T must use ONLY data from day T and earlier

RESPOND WITH:
1. Strategy Name (short, descriptive)
2. Core Logic (2-3 sentences explaining the CAUSAL mechanism - why would this work?)
3. Entry Rules (using only past data)
4. Exit Rules (using only past data)
5. Key Parameters

Keep your response concise and actionable."""

        result = self.api_manager.generate_with_retry(prompt, self.model_name)
        if result is None:
            raise Exception("API 呼叫失敗，所有 Key 都不可用")
        return result

    def generate_strategy_code(self, idea: str, strategy_id: int) -> Tuple[str, str]:
        """
        Ask Gemini to write Python code for the strategy.

        Returns:
            Tuple of (code_string, file_path)
        """
        class_name = f"Strategy_Gen{strategy_id}"

        prompt = f"""You are an expert Python developer writing trading strategy code.

STRATEGY IDEA:
{idea}

TASK:
Write a complete Python class that implements this strategy.

⚠️⚠️⚠️ CRITICAL - NO LOOK-AHEAD BIAS ⚠️⚠️⚠️
The signal on day T determines the position on day T+1 (the backtester handles this shift).
Your generate_signals() must ONLY use data available up to each day.

FORBIDDEN PATTERNS (will cause rejection):
❌ df.shift(-1) or any negative shift (looks into future)
❌ df.iloc[i+1] or forward indexing
❌ Any calculation using future prices/returns
❌ Rolling windows that somehow peek ahead

CORRECT PATTERNS:
✅ df['Close'].rolling(20).mean() - backward-looking moving average
✅ df.shift(1) - yesterday's value (positive shift = backward)
✅ All indicators calculated from historical data only

STRICT REQUIREMENTS:
1. The class MUST be named exactly: `{class_name}`
2. It MUST inherit from `BaseStrategy` (imported from strategy_base)
3. It MUST implement:
   - `init(self, data: pd.DataFrame)` - store data and calculate indicators
   - `generate_signals(self) -> pd.Series` - return position weights 0.0 to 1.0
   - `get_description(self) -> str` - return strategy description

4. Available imports (already in scope):
   - pandas as pd
   - numpy as np
   - from strategy_base import BaseStrategy

5. The data DataFrame has columns: ['Open', 'High', 'Low', 'Close', 'Volume']
   with a DatetimeIndex

6. Signals should be:
   - 0.0 = fully in cash (no position)
   - 1.0 = fully invested in TQQQ
   - Values between 0-1 for partial positions

7. Handle edge cases:
   - Use .fillna(0) or .bfill()/.ffill() for NaN (never forward-fill from future!)
   - Ensure signals align with data index

OUTPUT ONLY THE PYTHON CODE. NO MARKDOWN, NO EXPLANATIONS, NO ```python TAGS.
Start directly with 'import' or 'from' or 'class'."""

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

    def _build_context(self) -> str:
        """Build context string from history for the AI."""
        if self.history["total_iterations"] == 0:
            return """This is the FIRST iteration. No previous strategies have been tested.
Start with a robust baseline strategy that focuses on trend-following and volatility management.
TQQQ is extremely volatile - the 2022 bear market saw >75% drawdown."""

        # Get last 5 strategies for context
        recent = self.history["strategies"][-5:]

        context_lines = [
            f"Total iterations so far: {self.history['total_iterations']}",
            f"Best Sharpe Ratio achieved: {self.history['best_sharpe']:.2f}",
            f"Best strategy: {self.history['best_strategy']}",
            "",
            "RECENT STRATEGIES TRIED:"
        ]

        for s in recent:
            context_lines.append(
                f"- {s['name']}: Sharpe={s['sharpe']:.2f}, MaxDD={s['max_dd']:.1%}, "
                f"Failure: {s.get('failure_analysis', 'N/A')}"
            )

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
        success: bool
    ) -> None:
        """Record strategy result in history."""
        self.history["total_iterations"] += 1

        result = {
            "id": strategy_id,
            "name": strategy_name,
            "idea": idea[:500],  # Truncate long ideas
            "sharpe": sharpe,
            "cagr": cagr,
            "max_dd": max_dd,
            "failure_analysis": failure_analysis,
            "success": success,
            "timestamp": datetime.now().isoformat()
        }

        self.history["strategies"].append(result)

        # Update best if applicable
        if sharpe > self.history["best_sharpe"]:
            self.history["best_sharpe"] = sharpe
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
