#!/usr/bin/env python3
"""
Automated Strategy Discovery Engine — Main Loop
==================================================
LLM-driven iteration loop that autonomously invents, codes,
backtests, and evolves TQQQ trading strategies.

Architecture:
  1. Indicator Pool → pre-calculated 38 indicators
  2. LLM Generator  → gpt-4.1 / DeepSeek / Llama / Gemini
  3. Backtest Arena  → existing BacktestEngine
  4. Evolution Loop  → mutation feedback + hall_of_fame

Usage:
  python main_loop.py                       # Run 50 iterations
  python main_loop.py --iterations 100      # Run 100 iterations
  python main_loop.py --notify telegram     # Telegram report at end
"""

import os
import sys
import json
import time
import random
import argparse
import traceback
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, Dict, List

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from indicator_pool import get_enriched_data, get_indicator_menu, INDICATOR_REGISTRY
from strategy_base import BaseStrategy
from backtest import BacktestEngine
from validator import StrategyValidator, LookAheadDetector
from researcher import StrategySandbox


# ═══════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════
HALL_OF_FAME_FILE = Path("hall_of_fame.json")
HISTORY_FILE = Path("history_of_thoughts.json")
GENERATED_DIR = Path("generated_strategies")
GENERATED_DIR.mkdir(exist_ok=True)

# Hall of Fame thresholds
HOF_SHARPE_MIN = 1.2
HOF_MAX_DD_MIN = -0.30  # Max drawdown must be better (less negative) than -30%


# ═══════════════════════════════════════════════════════════════
# LLM Client — unified interface
# ═══════════════════════════════════════════════════════════════
class LLMClient:
    """Unified LLM client: Groq (5-key pool) → GitHub Models → Gemini failover."""

    def __init__(self):
        self._groq = None
        self._github = None
        self._gemini = None
        self.calls = 0
        self.groq_ok = 0
        self.github_ok = 0
        self.gemini_ok = 0

    def _get_groq(self):
        if self._groq is not None:
            return self._groq
        try:
            from groq_client import GroqClient
            self._groq = GroqClient()
            if not self._groq.keys:
                self._groq = None
                return None
            print(f"   🚀 Groq engine: {len(self._groq.keys)} keys, pool-based allocation")
            return self._groq
        except Exception as e:
            print(f"   ⚠️ Groq init failed: {e}")
            return None

    def _get_github(self):
        if self._github is not None:
            return self._github
        try:
            from multi_model_client import MultiModelClient
            self._github = MultiModelClient()
            return self._github
        except Exception as e:
            print(f"   ⚠️ GitHub Models init failed: {e}")
            return None

    def _get_gemini(self):
        if self._gemini is not None:
            return self._gemini
        try:
            from api_manager import get_api_manager
            self._gemini = get_api_manager()
            return self._gemini
        except Exception:
            return None

    def generate(self, prompt: str, task: str = "idea") -> Optional[str]:
        """Generate text via LLM with failover: Groq → GitHub → Gemini."""
        self.calls += 1

        # Primary: Groq (2 keys × multiple models, highest daily quota)
        groq = self._get_groq()
        if groq:
            result = groq.generate(prompt, task=task)
            if result:
                self.groq_ok += 1
                return result

        # Secondary: GitHub Models (50 RPD)
        gh = self._get_github()
        if gh:
            result = gh._call_model_chain(prompt)
            if result:
                self.github_ok += 1
                return result

        # Tertiary: Gemini (10 keys, per-project quota)
        gm = self._get_gemini()
        if gm:
            result = gm.generate_with_failover(prompt)
            if result:
                self.gemini_ok += 1
                return result

        return None

    def get_stats(self) -> str:
        groq_detail = ""
        if self._groq:
            groq_detail = f" [{self._groq.get_stats()}]"
        return f"LLM calls: {self.calls} (Groq: {self.groq_ok}{groq_detail}, GitHub: {self.github_ok}, Gemini: {self.gemini_ok})"


# ═══════════════════════════════════════════════════════════════
# Hall of Fame
# ═══════════════════════════════════════════════════════════════
def load_hall_of_fame() -> List[Dict]:
    if HALL_OF_FAME_FILE.exists():
        with open(HALL_OF_FAME_FILE) as f:
            return json.load(f)
    return []


def save_hall_of_fame(hof: List[Dict]):
    with open(HALL_OF_FAME_FILE, 'w') as f:
        json.dump(hof, f, indent=2, default=str)


def check_hall_of_fame(name: str, sharpe: float, max_dd: float,
                       cagr: float, calmar: float, idea: str) -> bool:
    """Check if strategy qualifies for hall of fame. Returns True if added."""
    if sharpe >= HOF_SHARPE_MIN and max_dd > HOF_MAX_DD_MIN:
        hof = load_hall_of_fame()
        entry = {
            "name": name,
            "sharpe": sharpe,
            "max_dd": max_dd,
            "cagr": cagr,
            "calmar": calmar,
            "idea": idea[:500],
            "inducted": datetime.now().isoformat(),
        }
        hof.append(entry)
        hof.sort(key=lambda x: x.get("calmar", 0), reverse=True)
        save_hall_of_fame(hof)
        print(f"   🏆 HALL OF FAME! {name} inducted (Sharpe={sharpe:.2f}, MaxDD={max_dd:.1%})")
        return True
    return False


# ═══════════════════════════════════════════════════════════════
# History tracking
# ═══════════════════════════════════════════════════════════════
def load_history() -> Dict:
    if HISTORY_FILE.exists():
        with open(HISTORY_FILE) as f:
            return json.load(f)
    return {"total_iterations": 0, "best_sharpe": 0.0, "best_calmar": 0.0, "best_strategy": None, "strategies": []}


def save_history(history: Dict):
    with open(HISTORY_FILE, 'w') as f:
        json.dump(history, f, indent=2, default=str)


def record_result(history: Dict, strategy_id: int, name: str, idea: str,
                  sharpe: float, cagr: float, max_dd: float, calmar: float,
                  analysis: str, success: bool):
    history["total_iterations"] += 1
    history["strategies"].append({
        "id": strategy_id,
        "name": name,
        "idea": idea[:500],
        "sharpe": sharpe,
        "calmar": calmar,
        "cagr": cagr,
        "max_dd": max_dd,
        "failure_analysis": analysis,
        "success": success,
        "timestamp": datetime.now().isoformat(),
    })
    if calmar > history.get("best_calmar", history.get("best_sharpe", 0)):
        history["best_sharpe"] = calmar  # 向下相容
        history["best_calmar"] = calmar
        history["best_strategy"] = name
    save_history(history)


# ═══════════════════════════════════════════════════════════════
# Prompt builders
# ═══════════════════════════════════════════════════════════════
def build_idea_prompt(history: Dict, indicator_menu: str) -> str:
    """Build the strategy idea generation prompt with Champion DNA + mutation feedback."""

    # Context from history
    total = history["total_iterations"]
    strategies = history["strategies"]
    successful = [s for s in strategies if s.get("success")]
    top3 = sorted(successful, key=lambda x: x.get("calmar", 0), reverse=True)[:3]
    recent5 = strategies[-5:] if strategies else []

    # Hall of fame
    hof = load_hall_of_fame()

    best_calmar = history.get('best_calmar', history.get('best_sharpe', 0))
    context = f"Total iterations: {total}\n"
    context += f"Best: {history['best_strategy']} (Calmar: {best_calmar:.2f})\n"
    context += f"Hall of Fame entries: {len(hof)}\n\n"

    if top3:
        context += "🏆 TOP 3 STRATEGIES:\n"
        for s in top3:
            context += f"  {s['name']}: Sharpe={s['sharpe']:.2f}, CAGR={s['cagr']:.1%}, MaxDD={s['max_dd']:.1%}\n"
        context += "\n"

    if recent5:
        context += "📝 LAST 5 ATTEMPTS:\n"
        for s in recent5:
            st = "✅" if s.get("success") else "❌"
            context += f"  {st} {s['name']}: Sharpe={s['sharpe']:.2f} | {s.get('failure_analysis','')[:60]}\n"
        context += "\n"

    # Mutation instruction based on latest result
    mutation = ""
    if recent5 and recent5[-1].get("success"):
        last = recent5[-1]
        if last["sharpe"] > 0.5:
            mutation = f"""
MUTATION DIRECTIVE: The previous strategy ({last['name']}) achieved Sharpe={last['sharpe']:.2f},
MaxDD={last['max_dd']:.1%}. Keep the good parts (regime filter, trend logic),
MUTATE the weak parts (drawdown control), and try adding a NEW indicator
from a different category."""
        else:
            mutation = f"""
MUTATION DIRECTIVE: The previous strategy had poor results (Sharpe={last['sharpe']:.2f}).
Try a COMPLETELY DIFFERENT approach. Explore a new indicator combination."""

    # Select random indicators to force exploration
    categories = list(set(v["category"] for v in INDICATOR_REGISTRY.values()))
    chosen_cats = random.sample(categories, min(3, len(categories)))
    forced_indicators = []
    for cat in chosen_cats:
        cat_inds = [(k, v) for k, v in INDICATOR_REGISTRY.items() if v["category"] == cat]
        name, info = random.choice(cat_inds)
        forced_indicators.append(f"  • {info['column']} ({info['category']}): {info['desc']}")

    forced_str = "\n".join(forced_indicators)

    # Randomly choose a mutation mode for champion DNA
    mutation_modes = [
        "MUTATE the REGIME FILTER: keep the entry/exit logic similar to Champion, but use a DIFFERENT regime detector (e.g., ADX, Ichimoku cloud, Bollinger squeeze instead of RVI).",
        "MUTATE the ENTRY SIGNAL: keep RVI-based regime filter, but use a DIFFERENT entry trigger (e.g., RSI divergence, MACD crossover, Stochastic bounce instead of RVI state transition).",
        "MUTATE the EXIT LOGIC: keep RVI regime + transition entry, but use a DIFFERENT exit method (e.g., trailing ATR stop, Chandelier exit, Donchian breakout instead of fixed RVI levels).",
        "COMBINE: use RVI_State for regime, add a MOMENTUM CONFIRMATION from a different category (RSI, MFI, CCI), and use ATR-based trailing stop for exits.",
        "HYBRID: create a strategy that blends RVI transitions with SMA trend direction and volume confirmation (OBV, CMF). Use state transitions for entry timing.",
        "IMPROVE SHORT SELLING: keep long-side RVI logic, but design a BETTER short-selling module using ADX, Supertrend, or Bollinger Band breakdown with tighter ATR stops.",
        "VOLUME BREAKOUT HYBRID: use Volume Surge (Vol_Ratio > 2.0 + Close > SMA) for entry timing, combined with RVI regime filter. Use ATR-based TP/SL for exits.",
        "VOLUME + MOMENTUM COMBO: combine volume breakout signals (Vol_Ratio) with momentum indicators (RSI, MFI, CCI) for multi-confirmation entries. Exit with trailing ATR stop.",
        "VOLUME REGIME FILTER: use volume patterns as regime filter (Vol_Ratio > 1.0 for trending, < 0.5 for quiet), combined with traditional trend entry signals.",
        "MEAN REVERSION: use ZScore for entry (buy when ZScore < -2, sell when > 2), with Aroon or ADX as trend confirmation to avoid catching falling knives.",
        "DI DIRECTIONAL: use DI_Plus/DI_Minus crossover for entry, ADX > 25 for regime filter. This captures strong directional moves early.",
        "BB SQUEEZE BREAKOUT: detect BB_Squeeze periods (low volatility), enter on squeeze release with volume confirmation, exit on ATR spike or ZScore extreme.",
        "MARKET STRUCTURE: use Drawdown + Days_Down for crash avoidance, Days_Up for momentum entry, Gap_Pct for event detection.",
        "MULTI-TIMEFRAME MOMENTUM: combine fast (ROC_5, RSI_7) and slow (ROC_20, TSI) momentum for signal confirmation. Enter when both align, exit when they diverge.",
        "ELDER RAY POWER: use Elder_Bull/Elder_Bear power to measure bull/bear strength. Enter when Bull Power turns positive with trend, exit when Bear Power dominates.",
    ]
    champion_mutation = random.choice(mutation_modes)

    return f"""You are a Quantitative Research Director designing TQQQ (3x Leveraged Nasdaq) strategies.

{context}
{mutation}

═══════════════════════════════════════════════════════════════
🧬 CHAMPION DNA — PROVEN STRATEGY TO BUILD UPON
═══════════════════════════════════════════════════════════════
Our BEST PROVEN strategy is the Champion RVI (Sharpe=1.28, CAGR=52.5%):

HOW IT WORKS:
1. RVI (Relative Volatility Index) creates 3 STATES:
   - Green: RVI_Refined > 59 (bullish volatility)
   - Orange: 42 ≤ RVI_Refined ≤ 59 (neutral)
   - Red: RVI_Refined < 42 (bearish volatility)

2. ENTRY: State TRANSITION from Orange/Red → Green (not just level!)
   The TRANSITION is key — it captures momentum BUILDING, not static levels.

3. EXIT LONG: RVI > 76 (overbought) OR RVI < 42 (breakdown)

4. SHORT: State transition Orange → Red, with ATR×1.8 TP/SL

═══════════════════════════════════════════════════════════════
📊 CHAMPION #2 — VOLUME BREAKOUT STRATEGY
═══════════════════════════════════════════════════════════════
Our SECOND proven strategy is the Volume Breakout:

HOW IT WORKS:
1. ENTRY: Close > SMA(20) AND Volume > AvgVolume × 2.0 (爆量突破)
   - Price above moving average confirms uptrend
   - Volume surge confirms institutional conviction

2. EXIT: ATR-based adaptive TP/SL
   - Take Profit: Entry + ATR × 2.5
   - Stop Loss: Entry - ATR × 1.5
   - Adapts to current volatility regime

KEY INSIGHT: Volume surges often precede strong directional moves in TQQQ.
Combining trend filter (MA) + volume confirmation reduces false breakouts.
Can be combined with RVI regime filter for even better results.

═══════════════════════════════════════════════════════════════
🧪 NEW INDICATOR CATEGORIES TO EXPLORE
═══════════════════════════════════════════════════════════════
We now have 60+ indicators across 7 categories:

• TREND QUALITY: DI_Plus, DI_Minus, DI_Diff, Aroon_Up/Down/Osc, TRIX, PPO
  → DI crossover (+DI > -DI) captures trend direction shifts
  → Aroon detects new highs/lows within window

• MEAN REVERSION: ZScore, SMA50_Dist, SMA200_Dist, RSI_7
  → ZScore < -2 = extremely oversold, potential bounce
  → SMA distance measures extension from trend

• VOLATILITY REGIME: ATR_Pct, HV_10, HV_30, BB_Squeeze, VoV
  → BB_Squeeze (BB inside KC) = low vol, expect breakout
  → VoV (volatility of volatility) = unstable regime detection

• MARKET STRUCTURE: Drawdown, Days_Up, Days_Down, Gap_Pct
  → Drawdown from peak = crash detection
  → Consecutive up/down days = momentum strength

• ADVANCED MOMENTUM: TSI, Elder_Bull/Bear, AO, UO, PPO, ROC_5/20
  → TSI zero-cross = strong trend confirmation
  → Elder Ray = bull/bear power separation

WHY TRANSITIONS BEAT THRESHOLDS:
- Threshold: "buy when RSI > 50" = many false signals in choppy markets
- Transition: "buy when RSI crosses FROM below 50 TO above 50" = fewer, higher-quality signals
- State machine: "buy when market STATE changes from neutral to bullish" = captures regime shifts

YOUR TASK — {champion_mutation}

{indicator_menu}

🎲 MANDATORY INDICATORS (must use at least 2 of these):
{forced_str}

═══════════════════════════════════════════
🎯 OBJECTIVE: Beat Champion RVI (Sharpe=1.28, MaxDD=-40%)
═══════════════════════════════════════════
Target: Sharpe > 1.2 AND MaxDD > -30%

DESIGN PRINCIPLES:
1. Use STATE TRANSITIONS (not just thresholds) for entry signals
2. REGIME FILTER must exist — cash during bear/high-vol markets
3. EXIT must be adaptive — ATR-based or volatility-adjusted
4. Short selling OPTIONAL but can add 5-10% annual return

RULES:
- Use ONLY backward-looking indicators (NO shift(-1), NO future data)
- Integer parameters only (10, 20, 50, 200)
- Maximum 4 conditions per signal
- Cash (0 exposure) is a valid and powerful position
- Signals: -1.0 (short) to 1.0 (long), 0.0 = cash

RESPOND WITH:
1. Strategy Name
2. Which Champion module(s) you KEPT vs CHANGED
3. State Machine Logic (what states, what transitions trigger signals)
4. Entry Signal Logic
5. Exit & Risk Logic
6. Short Selling Logic (if any)
7. Key Parameters

Keep response concise and actionable."""


def build_code_prompt(idea: str, strategy_id: int, indicator_menu: str) -> str:
    """Build the strategy code generation prompt with state machine patterns."""
    class_name = f"Strategy_Gen{strategy_id}"

    return f"""Write a Python trading strategy class with STATE MACHINE logic.

STRATEGY IDEA:
{idea}

{indicator_menu}

═══════════════════════════════════════════════════════════════
🧬 CHAMPION CODE PATTERN — FOLLOW THIS STRUCTURE
═══════════════════════════════════════════════════════════════
```
from strategy_base import BaseStrategy
import pandas as pd
import numpy as np

class {class_name}(BaseStrategy):
    def init(self, data: pd.DataFrame):
        self.data = data
        # All indicators are PRE-CALCULATED in self.data
        # Available: self.data['RVI'], self.data['RVI_Refined'], self.data['RVI_State']
        # Plus: RSI, ATR, MACD, SMA_50, SMA_200, ADX, Supertrend, BB_width, etc.

    def _get_state(self) -> pd.Series:
        \"\"\"Define market states using indicators.\"\"\"
        # Example using RVI:
        state = pd.Series('neutral', index=self.data.index)
        state[self.data['RVI_Refined'] > 59] = 'bull'
        state[self.data['RVI_Refined'] < 42] = 'bear'
        return state

    def generate_signals(self) -> pd.Series:
        \"\"\"Use state TRANSITIONS for signals. Range: -1.0 to 1.0.\"\"\"
        state = self._get_state()
        prev_state = state.shift(1).fillna('unknown')
        signals = pd.Series(0.0, index=self.data.index)

        # IMPORTANT: Use a loop to track position state properly
        position = 0  # 0=cash, 1=long, -1=short
        for i in range(len(self.data)):
            curr = state.iloc[i]
            prev = prev_state.iloc[i]

            # Entry: state TRANSITION (not just level!)
            if prev in ('neutral', 'bear') and curr == 'bull':
                position = 1
            # Exit conditions
            elif position == 1 and <exit_condition>:
                position = 0
            # Short entry (optional)
            elif prev == 'neutral' and curr == 'bear':
                position = -1
            # Short exit
            elif position == -1 and <short_exit_condition>:
                position = 0

            signals.iloc[i] = float(position)

        return signals.clip(-1, 1)

    def get_description(self) -> str:
        return "Description of strategy"
```

═══════════════════════════════════════════════════════════════
⚡ KEY PATTERNS TO USE
═══════════════════════════════════════════════════════════════

STATE TRANSITION ENTRY (better than threshold):
  # BAD: signals[self.data['RSI'] > 50] = 1.0  (too many false signals)
  # GOOD: buy on state transition neutral→bull (captures momentum shift)

ATR-BASED ADAPTIVE EXITS:
  atr = self.data['ATR']
  entry_price = ...
  tp_price = entry_price + atr * 2.0  # take profit
  sl_price = entry_price - atr * 1.5  # stop loss

RVI STATE (pre-calculated):
  self.data['RVI_State']  # 1=bull, 0=neutral, -1=bear
  self.data['RVI_Refined']  # 0-100 continuous value

CRITICAL RULES:
- Class name MUST be: {class_name}
- Inherit from BaseStrategy
- Indicators are ALREADY in self.data — do NOT recalculate them
- NO look-ahead: no shift(-1), no iloc[i+1], no future data
- Handle NaN with .fillna(0) or .bfill()
- Return pd.Series of floats -1.0 to 1.0 (negative = short)
- Use a for-loop to track position state (stateful logic)

OUTPUT ONLY PYTHON CODE. No markdown, no explanations."""


def build_fix_prompt(code: str, error: str, strategy_id: int) -> str:
    class_name = f"Strategy_Gen{strategy_id}"
    return f"""Fix this broken Python trading strategy.

BROKEN CODE:
{code}

ERROR:
{error}

REQUIREMENTS:
1. Class name: {class_name}
2. Inherit from BaseStrategy
3. Implement init(self, data), generate_signals() -> pd.Series, get_description() -> str
4. Indicators are pre-calculated in self.data — just reference them
5. generate_signals() returns pd.Series with values 0.0 to 1.0

OUTPUT ONLY THE FIXED PYTHON CODE. No markdown."""


# ═══════════════════════════════════════════════════════════════
# Code cleaning
# ═══════════════════════════════════════════════════════════════
def clean_code(code: str) -> str:
    import re
    code = code.strip()
    # Handle JSON-wrapped code (GPT-4.1 sometimes wraps in {"code": "..."})
    if code.startswith('{') and '"code"' in code[:50]:
        try:
            parsed = json.loads(code)
            if isinstance(parsed, dict) and "code" in parsed:
                code = parsed["code"]
        except json.JSONDecodeError:
            pass
    # Remove markdown code blocks
    code = re.sub(r'^```python\s*', '', code, flags=re.MULTILINE)
    code = re.sub(r'^```\s*$', '', code, flags=re.MULTILINE)
    code = re.sub(r'```$', '', code)
    return code.strip()


# ═══════════════════════════════════════════════════════════════
# Single iteration
# ═══════════════════════════════════════════════════════════════
def run_iteration(
    llm: LLMClient,
    engine: BacktestEngine,
    data: pd.DataFrame,
    history: Dict,
    indicator_menu: str,
    strategy_id: int,
) -> Dict:
    """Run a single strategy discovery iteration."""

    result = {
        "id": strategy_id,
        "name": f"Strategy_Gen{strategy_id}",
        "success": False,
        "sharpe": 0.0,
        "cagr": 0.0,
        "max_dd": 0.0,
        "calmar": 0.0,
        "error": None,
    }

    # Step 1: Generate idea
    print("   💡 Generating idea...")
    idea_prompt = build_idea_prompt(history, indicator_menu)
    idea = llm.generate(idea_prompt, task="idea")
    if not idea:
        result["error"] = "LLM failed to generate idea"
        return result

    # Step 2: Generate code
    print("   💻 Generating code...")
    code_prompt = build_code_prompt(idea, strategy_id, indicator_menu)
    raw_code = llm.generate(code_prompt, task="code")
    if not raw_code:
        result["error"] = "LLM failed to generate code"
        record_result(history, strategy_id, result["name"], idea,
                      0, 0, 0, 0, result["error"], False)
        return result

    code = clean_code(raw_code)
    if "from strategy_base import BaseStrategy" not in code:
        code = "from strategy_base import BaseStrategy\nimport pandas as pd\nimport numpy as np\n\n" + code

    # Step 3: Validate for look-ahead bias
    is_valid, warnings = StrategyValidator.validate_code(code)
    if not is_valid:
        result["error"] = "Look-ahead bias in code"
        record_result(history, strategy_id, result["name"], idea,
                      0, 0, 0, 0, result["error"], False)
        return result

    # Step 4: Save, load, and test
    file_path = GENERATED_DIR / f"strategy_gen_{strategy_id}.py"
    with open(file_path, 'w') as f:
        f.write(code)

    class_name = f"Strategy_Gen{strategy_id}"
    sandbox = StrategySandbox()

    for attempt in range(2):
        try:
            strategy = sandbox.load_strategy(str(file_path), class_name)
            success, err = sandbox.test_strategy(strategy, data)

            if success:
                break

            if attempt == 0:
                print(f"   🔧 Fix attempt ({err[:50]})...")
                fix_prompt = build_fix_prompt(code, err, strategy_id)
                fixed = llm.generate(fix_prompt, task="fix")
                if fixed:
                    code = clean_code(fixed)
                    if "from strategy_base import BaseStrategy" not in code:
                        code = "from strategy_base import BaseStrategy\nimport pandas as pd\nimport numpy as np\n\n" + code
                    with open(file_path, 'w') as f:
                        f.write(code)

        except Exception as e:
            err = str(e)
            if attempt == 0:
                print(f"   🔧 Fix attempt ({err[:50]})...")
                fix_prompt = build_fix_prompt(code, err, strategy_id)
                fixed = llm.generate(fix_prompt, task="fix")
                if fixed:
                    code = clean_code(fixed)
                    if "from strategy_base import BaseStrategy" not in code:
                        code = "from strategy_base import BaseStrategy\nimport pandas as pd\nimport numpy as np\n\n" + code
                    with open(file_path, 'w') as f:
                        f.write(code)
                success = False
            else:
                success = False

    if not success:
        result["error"] = err[:100] if 'err' in dir() else "Load failed"
        record_result(history, strategy_id, result["name"], idea,
                      0, 0, 0, 0, result["error"], False)
        return result

    # Step 5: Backtest
    print("   📊 Backtesting...")
    try:
        bt = engine.run(strategy)

        # Validate results
        valid, _ = StrategyValidator.validate_backtest_results(bt)
        if not valid:
            result["error"] = "Unrealistic results (possible bug)"
            record_result(history, strategy_id, result["name"], idea,
                          0, 0, 0, 0, result["error"], False)
            return result

        # Runtime look-ahead detection for promising strategies
        if bt.sharpe_ratio > 0.5:
            print("   🔍 Running look-ahead detection...")
            try:
                # Statistical test: does strategy suspiciously dodge worst days?
                signals = strategy.generate_signals()
                returns = data['Close'].pct_change()
                stat_valid, stat_msg = StrategyValidator.validate_signals(
                    signals, returns
                )
                if not stat_valid:
                    print(f"   {stat_msg}")
                    result["error"] = f"Statistical look-ahead: {stat_msg}"
                    record_result(history, strategy_id, result["name"], idea,
                                  0, 0, 0, 0, result["error"], False)
                    return result

                # Online consistency test: do signals change with more data?
                la_valid, la_msg = LookAheadDetector.test_online_consistency(
                    strategy, data, sample_points=50
                )
                print(f"   {la_msg}")
                if not la_valid:
                    result["error"] = f"Look-ahead bias detected: {la_msg}"
                    record_result(history, strategy_id, result["name"], idea,
                                  0, 0, 0, 0, result["error"], False)
                    return result
            except Exception as e:
                print(f"   ⚠️ Look-ahead test error (continuing): {str(e)[:60]}")

        result["success"] = True
        result["sharpe"] = bt.sharpe_ratio
        result["cagr"] = bt.cagr
        result["max_dd"] = bt.max_drawdown
        result["calmar"] = bt.calmar_ratio
        result["idea"] = idea[:200]

        analysis = bt.get_failure_analysis()
        record_result(history, strategy_id, result["name"], idea,
                      bt.sharpe_ratio, bt.cagr, bt.max_drawdown, bt.calmar_ratio,
                      analysis, True)

        # Check hall of fame
        check_hall_of_fame(result["name"], bt.sharpe_ratio, bt.max_drawdown,
                          bt.cagr, bt.calmar_ratio, idea)

    except Exception as e:
        result["error"] = str(e)[:100]
        record_result(history, strategy_id, result["name"], idea,
                      0, 0, 0, 0, f"Backtest error: {result['error']}", False)

    return result


# ═══════════════════════════════════════════════════════════════
# Reporting
# ═══════════════════════════════════════════════════════════════
def generate_report(history: Dict, session_stats: Dict) -> str:
    total = history["total_iterations"]
    strategies = history["strategies"]
    successful = [s for s in strategies if s.get("success")]
    top5 = sorted(successful, key=lambda x: x.get("calmar", 0), reverse=True)[:5]
    recent10 = strategies[-10:]
    hof = load_hall_of_fame()

    report = f"""
{'='*50}
📊 TQQQ Strategy Discovery Report
   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*50}

📈 Overall
  Total iterations: {total}
  Successful: {len(successful)} ({len(successful)/total*100:.0f}%)
  Best: {history['best_strategy']} (Calmar: {history.get('best_calmar', history.get('best_sharpe', 0)):.2f})
  Hall of Fame: {len(hof)} strategies

  This session: {session_stats['iterations']} iters, {session_stats['successes']} ok
  Duration: {session_stats['duration']}
  {session_stats['llm_stats']}

🏆 Top 5 Strategies
{'─'*50}"""
    for i, s in enumerate(top5, 1):
        report += f"\n  #{i} {s['name']}: Calmar={s.get('calmar',0):.2f} Sharpe={s['sharpe']:.2f} CAGR={s['cagr']:.1%} MaxDD={s['max_dd']:.1%}"

    if hof:
        report += f"\n\n🥇 Hall of Fame ({len(hof)})\n{'─'*50}"
        for h in hof[:5]:
            report += f"\n  {h['name']}: Calmar={h.get('calmar',0):.2f} Sharpe={h['sharpe']:.2f} MaxDD={h['max_dd']:.1%}"

    report += f"\n\n📝 Last 10 Iterations\n{'─'*50}"
    for s in recent10:
        st = "✅" if s.get("success") else "❌"
        if s.get("success"):
            info = f"Calmar={s.get('calmar',0):.2f} Sharpe={s['sharpe']:.2f} CAGR={s['cagr']:.1%} MaxDD={s['max_dd']:.1%}"
        else:
            info = s.get("failure_analysis", "")[:40]
        report += f"\n  {st} {s['name']}: {info}"

    report += f"\n\n{'='*50}\n"
    return report


def send_telegram(report: str):
    try:
        import requests
        bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
        chat_id = os.getenv("TELEGRAM_CHAT_ID")
        if not bot_token or not chat_id:
            return
        text = report[:4000] if len(report) > 4000 else report
        resp = requests.post(
            f"https://api.telegram.org/bot{bot_token}/sendMessage",
            data={"chat_id": chat_id, "text": text}
        )
        if resp.status_code == 200:
            print("📱 Telegram report sent")
    except Exception as e:
        print(f"⚠️ Telegram failed: {e}")


# ═══════════════════════════════════════════════════════════════
# Main Loop
# ═══════════════════════════════════════════════════════════════
def run_champion_baseline(engine: BacktestEngine, data: pd.DataFrame, history: Dict):
    """Run Champion RVI and Volume Breakout as baselines."""
    # Check if champion already in history
    has_rvi = any(s.get("name") == "ChampionRVI" for s in history.get("strategies", []))

    if has_rvi:
        print("   📌 Champion RVI baseline already recorded")
    else:
        print("\n🧬 Running Champion RVI baseline...")
        try:
            from champion_rvi import ChampionRVI

            # Long only version
            champ = ChampionRVI(enable_short=False)
            bt = engine.run(champ)
            record_result(history, 0, "ChampionRVI", "Champion RVI state machine strategy",
                          bt.sharpe_ratio, bt.cagr, bt.max_drawdown, bt.calmar_ratio,
                          "BASELINE: Champion DNA from proven Pine Script strategy", True)
            check_hall_of_fame("ChampionRVI", bt.sharpe_ratio, bt.max_drawdown,
                              bt.cagr, bt.calmar_ratio, "Champion RVI baseline")
            print(f"   📌 Champion baseline: Sharpe={bt.sharpe_ratio:.2f}, "
                  f"CAGR={bt.cagr:.1%}, MaxDD={bt.max_drawdown:.1%}")

            # With short
            champ_short = ChampionRVI(enable_short=True)
            bt_s = engine.run(champ_short)
            record_result(history, 0, "ChampionRVI_Short", "Champion RVI with short selling",
                          bt_s.sharpe_ratio, bt_s.cagr, bt_s.max_drawdown, bt_s.calmar_ratio,
                          "BASELINE: Champion DNA with short selling", True)
            print(f"   📌 Champion+Short: Sharpe={bt_s.sharpe_ratio:.2f}, "
                  f"CAGR={bt_s.cagr:.1%}, MaxDD={bt_s.max_drawdown:.1%}")

        except Exception as e:
            print(f"   ⚠️ Champion baseline failed: {e}")

    # Volume Breakout baseline
    print("\n📊 Running Volume Breakout baseline...")
    try:
        from champion_volume_breakout import ChampionVolumeBreakout

        # Check if already recorded
        has_vb = any(s.get("name") == "VolumeBreakout_ATR" for s in history.get("strategies", []))
        if not has_vb:
            # ATR mode
            vb_atr = ChampionVolumeBreakout(exit_mode='atr')
            bt_vb = engine.run(vb_atr)
            record_result(history, 0, "VolumeBreakout_ATR",
                          "Volume Breakout with ATR-based TP/SL",
                          bt_vb.sharpe_ratio, bt_vb.cagr, bt_vb.max_drawdown,
                          bt_vb.calmar_ratio,
                          "BASELINE: Volume surge breakout with ATR exits", True)
            check_hall_of_fame("VolumeBreakout_ATR", bt_vb.sharpe_ratio,
                              bt_vb.max_drawdown, bt_vb.cagr, bt_vb.calmar_ratio,
                              "Volume Breakout ATR baseline")
            print(f"   📌 VolumeBreakout ATR: Sharpe={bt_vb.sharpe_ratio:.2f}, "
                  f"CAGR={bt_vb.cagr:.1%}, MaxDD={bt_vb.max_drawdown:.1%}")

            # Percentage mode
            vb_pct = ChampionVolumeBreakout(exit_mode='pct')
            bt_vp = engine.run(vb_pct)
            record_result(history, 0, "VolumeBreakout_Pct",
                          "Volume Breakout with percentage-based TP/SL",
                          bt_vp.sharpe_ratio, bt_vp.cagr, bt_vp.max_drawdown,
                          bt_vp.calmar_ratio,
                          "BASELINE: Volume surge breakout with pct exits", True)
            print(f"   📌 VolumeBreakout Pct: Sharpe={bt_vp.sharpe_ratio:.2f}, "
                  f"CAGR={bt_vp.cagr:.1%}, MaxDD={bt_vp.max_drawdown:.1%}")
        else:
            print("   📌 Volume Breakout baseline already recorded")

    except Exception as e:
        print(f"   ⚠️ Volume Breakout baseline failed: {e}")


def run_crossover_round(data: pd.DataFrame, history: Dict):
    """Run a crossover round to discover best module combinations."""
    print("\n🧬 Running Crossover Round...")
    try:
        from strategy_crossover import run_crossover

        results = run_crossover(data, top_n=5)

        for r in results[:3]:
            name = r["name"]
            record_result(history, 0, name,
                          f"Crossover: {r['regime']}×{r['entry']}×{r['exit']}",
                          r["sharpe"], r["cagr"], r["max_dd"], r["calmar"],
                          "CROSSOVER: modular component combination", True)
            check_hall_of_fame(name, r["sharpe"], r["max_dd"],
                              r["cagr"], r["calmar"],
                              f"Crossover: {r['regime']}×{r['entry']}×{r['exit']}")

    except Exception as e:
        print(f"   ⚠️ Crossover round failed: {e}")


def run_param_sweep_round(data: pd.DataFrame, history: Dict):
    """Run parameter sweep on Champion RVI and Volume Breakout."""
    print("\n🔬 Running Parameter Sweep...")

    # Sweep Champion RVI
    try:
        from param_sweep import sweep_champion

        results = sweep_champion(data)

        if results:
            best = results[0]
            name = "ChampionRVI_Optimized"

            baseline_sharpe = 0
            for s in history.get("strategies", []):
                if s.get("name") == "ChampionRVI":
                    baseline_sharpe = s.get("sharpe", 0)
                    break

            if best["sharpe"] > baseline_sharpe:
                record_result(history, 0, name,
                              f"Optimized Champion RVI: {best['params']}",
                              best["sharpe"], best["cagr"], best["max_dd"],
                              best["calmar"],
                              "SWEEP: parameter optimized champion", True)
                check_hall_of_fame(name, best["sharpe"], best["max_dd"],
                                  best["cagr"], best["calmar"],
                                  f"Optimized params: {best['params']}")
                print(f"   🎯 RVI Sweep found better params: Sharpe={best['sharpe']:.2f}")
            else:
                print(f"   📌 RVI Sweep: base params are already optimal")

    except Exception as e:
        print(f"   ⚠️ RVI Parameter sweep failed: {e}")

    # Sweep Volume Breakout
    try:
        from param_sweep import sweep_volume_breakout

        vb_results = sweep_volume_breakout(data)

        if vb_results:
            best_vb = vb_results[0]
            name_vb = "VolumeBreakout_Optimized"

            vb_baseline_sharpe = 0
            for s in history.get("strategies", []):
                if s.get("name") == "VolumeBreakout_ATR":
                    vb_baseline_sharpe = s.get("sharpe", 0)
                    break

            if best_vb["sharpe"] > vb_baseline_sharpe:
                record_result(history, 0, name_vb,
                              f"Optimized Volume Breakout: {best_vb['params']}",
                              best_vb["sharpe"], best_vb["cagr"], best_vb["max_dd"],
                              best_vb["calmar"],
                              "SWEEP: parameter optimized volume breakout", True)
                check_hall_of_fame(name_vb, best_vb["sharpe"], best_vb["max_dd"],
                                  best_vb["cagr"], best_vb["calmar"],
                                  f"Optimized VB params: {best_vb['params']}")
                print(f"   🎯 VB Sweep found better params: Sharpe={best_vb['sharpe']:.2f}")
            else:
                print(f"   📌 VB Sweep: base params are already optimal")

    except Exception as e:
        print(f"   ⚠️ Volume Breakout sweep failed: {e}")


def main():
    parser = argparse.ArgumentParser(description='TQQQ Automated Strategy Discovery Engine v2')
    parser.add_argument('--iterations', type=int, default=50, help='Number of iterations')
    parser.add_argument('--target-sharpe', type=float, default=2.0, help='Stop if Sharpe >= this')
    parser.add_argument('--notify', type=str, default='file', choices=['file', 'telegram'])
    parser.add_argument('--report-every', type=int, default=25, help='Report frequency')
    parser.add_argument('--refresh-data', action='store_true', help='Force refresh market data')
    parser.add_argument('--skip-baseline', action='store_true', help='Skip champion baseline')
    parser.add_argument('--crossover-every', type=int, default=25, help='Crossover round frequency')
    parser.add_argument('--sweep-every', type=int, default=50, help='Parameter sweep frequency')
    parser.add_argument('--max-cooldown', type=int, default=1800, help='Max cooldown seconds when all APIs exhausted')
    args = parser.parse_args()

    print("=" * 55)
    print("🚀 TQQQ Strategy Discovery Engine v2")
    print(f"   🧬 Champion DNA + State Machine + Crossover")
    print(f"   🔑 Groq 5-Key Pool: idea=K1,K2 | code=K3,K4 | fix=K5")
    print(f"   Iterations: {args.iterations}")
    print(f"   Target Sharpe: {args.target_sharpe}")
    print(f"   Crossover every: {args.crossover_every} iters")
    print(f"   Param sweep every: {args.sweep_every} iters")
    print("=" * 55)

    # Load data with pre-calculated indicators (now includes RVI)
    data = get_enriched_data(force_refresh=args.refresh_data)
    indicator_menu = get_indicator_menu()
    engine = BacktestEngine(data)
    history = load_history()
    llm = LLMClient()

    # ─── Phase 0: Establish Champion baseline ───
    if not args.skip_baseline:
        run_champion_baseline(engine, data, history)

    session_start = datetime.now()
    session_successes = 0
    consec_api_fail = 0

    for i in range(1, args.iterations + 1):
        strategy_id = history["total_iterations"] + 1
        print(f"\n[{i}/{args.iterations}] Iteration {strategy_id}")

        # ─── Crossover round (every N iterations) ───
        if i > 1 and i % args.crossover_every == 0:
            run_crossover_round(data, history)

        # ─── Parameter sweep (every N iterations) ───
        if i > 1 and i % args.sweep_every == 0:
            run_param_sweep_round(data, history)

        # ─── Normal LLM-generated strategy iteration ───
        result = run_iteration(llm, engine, data, history, indicator_menu, strategy_id)

        if result["success"]:
            calmar = result.get("calmar", 0)
            print(f"   ✅ Sharpe={result['sharpe']:.2f} CAGR={result['cagr']:.1%} "
                  f"MaxDD={result['max_dd']:.1%} Calmar={calmar:.2f}")
            session_successes += 1
            consec_api_fail = 0

            if result["sharpe"] >= args.target_sharpe:
                print(f"\n🎯 TARGET ACHIEVED! Sharpe {result['sharpe']:.2f}")
                break
        else:
            error = result["error"] or "Unknown"
            print(f"   ❌ {error[:60]}")

            if "都不可用" in error or "failed to generate" in error.lower():
                consec_api_fail += 1
                if consec_api_fail >= 5:
                    # Long backoff: all APIs likely exhausted, wait for quota reset
                    cooldown = min(300 * (consec_api_fail - 4), args.max_cooldown)
                    print(f"   ⏳ All APIs exhausted, long cooldown {cooldown}s ({cooldown//60}min)...")
                else:
                    cooldown = min(60 * consec_api_fail, 300)
                    print(f"   ⏳ API cooldown {cooldown}s...")
                time.sleep(cooldown)
                continue
            else:
                consec_api_fail = 0

        # Periodic report
        if i % args.report_every == 0:
            stats = {
                "iterations": i,
                "successes": session_successes,
                "duration": str(datetime.now() - session_start).split('.')[0],
                "llm_stats": llm.get_stats(),
            }
            report = generate_report(history, stats)
            print(report)
            if args.notify == "telegram":
                send_telegram(report)

        time.sleep(5)  # Pace API calls

    # Final report
    stats = {
        "iterations": i,
        "successes": session_successes,
        "duration": str(datetime.now() - session_start).split('.')[0],
        "llm_stats": llm.get_stats(),
    }
    report = generate_report(history, stats)
    print(report)

    if args.notify == "telegram":
        send_telegram(report)

    # Save latest report
    with open("latest_report.txt", "w") as f:
        f.write(report)


if __name__ == "__main__":
    main()
