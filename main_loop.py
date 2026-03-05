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
from brief_system import InternalAnalyst, Secretary, SolutionResearcher, AnalysisValidator, ResultChecker, ExternalResearcher


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

# Minimum thresholds for ranking — filter out "do nothing" strategies
RANK_MIN_SHARPE = 0.0   # Must beat risk-free rate
RANK_MIN_CAGR = 0.05    # Must generate at least 5% annual return

# Hard filter thresholds — strategies below these are rejected outright
HARD_FILTER_MAX_DD = -0.50    # MaxDD must be better than -50%
HARD_FILTER_MIN_TRADES = 10   # Must have at least 10 trades
HARD_FILTER_MIN_EXPOSURE = 0.05  # Must be in market at least 5% of time
HARD_FILTER_MIN_SHARPE = 0.3  # Must have Sharpe >= 0.3

# Failure memory size
FAILURE_MEMORY_SIZE = 15  # Keep last N failure reasons in FIFO queue

# Director (投資總監) review settings
DIRECTOR_INTERVAL = 50       # Every N iterations, director reviews & gives guidance
STAGNATION_THRESHOLD = 15    # N consecutive iterations without improvement → early director call

# Composite score weights
COMPOSITE_WEIGHTS = {
    "sharpe": 0.30,
    "calmar": 0.25,
    "sortino": 0.20,
    "max_dd": 0.15,
    "profit_factor": 0.10,
}


def is_rankable(s: Dict) -> bool:
    """Check if a strategy qualifies for ranking (not a 'do nothing' strategy)."""
    return (s.get("success", False)
            and s.get("sharpe", 0) > RANK_MIN_SHARPE
            and s.get("cagr", 0) > RANK_MIN_CAGR)


def calculate_composite_score(sharpe: float, calmar: float, sortino: float,
                              max_dd: float, profit_factor: float) -> float:
    """
    Calculate weighted composite score with 0-1 normalization.
    Formula: Sharpe*0.30 + Calmar*0.25 + Sortino*0.20 + DD*0.15 + PF*0.10
    """
    # Normalize each metric to 0-1 range using sigmoid-like clamping
    def norm(val, low, high):
        """Normalize val from [low, high] to [0, 1], clamped."""
        if high == low:
            return 0.5
        return max(0.0, min(1.0, (val - low) / (high - low)))

    n_sharpe = norm(sharpe, -1.0, 3.0)          # Sharpe: -1 to 3
    n_calmar = norm(calmar, 0.0, 5.0)           # Calmar: 0 to 5
    n_sortino = norm(sortino, -1.0, 5.0)        # Sortino: -1 to 5
    n_dd = norm(max_dd, -0.80, 0.0)             # MaxDD: -80% to 0% (higher=better)
    n_pf = norm(min(profit_factor, 5.0), 0.0, 5.0)  # PF: 0 to 5, cap at 5

    score = (COMPOSITE_WEIGHTS["sharpe"] * n_sharpe
             + COMPOSITE_WEIGHTS["calmar"] * n_calmar
             + COMPOSITE_WEIGHTS["sortino"] * n_sortino
             + COMPOSITE_WEIGHTS["max_dd"] * n_dd
             + COMPOSITE_WEIGHTS["profit_factor"] * n_pf)
    return round(score, 4)


def hard_filter(bt) -> Optional[str]:
    """
    Apply hard rejection filter to a backtest result.
    Returns None if passed, or a rejection reason string if failed.
    """
    if bt.max_drawdown < HARD_FILTER_MAX_DD:
        return f"MaxDD={bt.max_drawdown:.1%} worse than {HARD_FILTER_MAX_DD:.0%}"
    if bt.total_trades < HARD_FILTER_MIN_TRADES:
        return f"Only {bt.total_trades} trades (min={HARD_FILTER_MIN_TRADES})"
    if bt.time_in_market < HARD_FILTER_MIN_EXPOSURE:
        return f"Exposure={bt.time_in_market:.1%} below {HARD_FILTER_MIN_EXPOSURE:.0%}"
    if bt.sharpe_ratio < HARD_FILTER_MIN_SHARPE:
        return f"Sharpe={bt.sharpe_ratio:.2f} below {HARD_FILTER_MIN_SHARPE}"
    return None


def is_duplicate_result(bt, history: dict, tol: float = 1e-4) -> bool:
    """
    Return True if backtest metrics are functionally identical to any existing
    strategy in history (within tolerance). Prevents recording RVI clones.
    """
    sharpe = round(bt.sharpe_ratio, 4)
    cagr   = round(bt.cagr, 4)
    max_dd = round(bt.max_drawdown, 4)
    for s in history.get("strategies", []):
        if (abs(s.get("sharpe", -999) - sharpe) < tol and
                abs(s.get("cagr",   -999) - cagr)   < tol and
                abs(s.get("max_dd", -999) - max_dd) < tol):
            return True
    return False


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
    return {"total_iterations": 0, "best_sharpe": 0.0, "best_calmar": 0.0, "best_composite": 0.0, "best_strategy": None, "strategies": []}


def save_history(history: Dict):
    with open(HISTORY_FILE, 'w') as f:
        json.dump(history, f, indent=2, default=str)


def record_result(history: Dict, strategy_id: int, name: str, idea: str,
                  sharpe: float, cagr: float, max_dd: float, calmar: float,
                  analysis: str, success: bool,
                  composite: float = 0.0,
                  test_sharpe: float = 0.0, test_cagr: float = 0.0,
                  test_max_dd: float = 0.0, test_composite: float = 0.0):
    history["total_iterations"] += 1
    history["strategies"].append({
        "id": strategy_id,
        "name": name,
        "idea": idea[:500],
        "sharpe": sharpe,
        "calmar": calmar,
        "cagr": cagr,
        "max_dd": max_dd,
        "composite": composite,
        "test_sharpe": test_sharpe,
        "test_cagr": test_cagr,
        "test_max_dd": test_max_dd,
        "test_composite": test_composite,
        "failure_analysis": analysis,
        "success": success,
        "timestamp": datetime.now().isoformat(),
    })
    if (composite > history.get("best_composite", 0)
            and sharpe > RANK_MIN_SHARPE and cagr > RANK_MIN_CAGR):
        history["best_sharpe"] = sharpe
        history["best_calmar"] = calmar
        history["best_composite"] = composite
        history["best_strategy"] = name
    save_history(history)


# ═══════════════════════════════════════════════════════════════
# Prompt builders
# ═══════════════════════════════════════════════════════════════
def build_idea_prompt(history: Dict, indicator_menu: str,
                      director_advice: Optional[str] = None,
                      oos_warning: Optional[str] = None,
                      brief: Optional[Dict] = None,
                      current_assignment: Optional[str] = None) -> str:
    """Build the strategy idea generation prompt with Champion DNA + mutation feedback."""

    # Context from history
    total = history["total_iterations"]
    strategies = history["strategies"]
    rankable = [s for s in strategies if is_rankable(s)]
    top3 = sorted(rankable, key=lambda x: x.get("sharpe", 0), reverse=True)[:3]
    recent5 = strategies[-5:] if strategies else []

    # Hall of fame
    hof = load_hall_of_fame()

    best_composite = history.get('best_composite', history.get('best_calmar', history.get('best_sharpe', 0)))
    context = f"Total iterations: {total}\n"
    context += f"Best: {history['best_strategy']} (Composite: {best_composite:.4f})\n"
    context += f"Hall of Fame entries: {len(hof)}\n\n"

    if top3:
        context += "🏆 TOP 3 STRATEGIES (by Sharpe ratio):\n"
        for s in top3:
            cs = s.get('composite', 0)
            context += (f"  {s['name']}: Composite={cs:.4f}, Sharpe={s['sharpe']:.2f}, "
                        f"CAGR={s['cagr']:.1%}, MaxDD={s['max_dd']:.1%}")
            if s.get("test_sharpe"):
                context += f" | OOS: Sharpe={s['test_sharpe']:.2f}, CAGR={s['test_cagr']:.1%}"
            context += "\n"
        context += "\n"

    if recent5:
        context += "📝 LAST 5 ATTEMPTS:\n"
        for s in recent5:
            st = "✅" if s.get("success") else "❌"
            context += f"  {st} {s['name']}: Sharpe={s['sharpe']:.2f} | {s.get('failure_analysis','')[:60]}\n"
        context += "\n"

    # Failure memory — last N failures with rejection reasons
    recent_failures = [s for s in strategies[-30:] if not s.get("success")][-FAILURE_MEMORY_SIZE:]
    if recent_failures:
        context += f"⚠️ RECENT {len(recent_failures)} FAILURES (AVOID THESE PATTERNS):\n"
        for s in recent_failures:
            context += f"  ❌ {s['name']}: {s.get('failure_analysis','')[:80]}\n"
        context += "\n"

    # OOS overfitting warning
    oos_section = ""
    if oos_warning:
        oos_section = f"""
═══════════════════════════════════════════════════════════════
⚠️ OVERFITTING WARNING (MUST HEED)
═══════════════════════════════════════════════════════════════
{oos_warning}

You MUST design a SIMPLER strategy with fewer parameters to avoid overfitting.
Use fewer conditions (max 3), prefer robust indicators, avoid fine-tuned thresholds.
"""

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

    # Directed assignment from execution queue (overrides random mutation mode when set)
    assignment_section = ""
    if current_assignment:
        assignment_section = f"""
═══════════════════════════════════════════════════════════════
🎯 SPECIFIC ASSIGNMENT (HIGHEST PRIORITY — FOLLOW EXACTLY)
═══════════════════════════════════════════════════════════════
{current_assignment}

This is a directed research assignment. Implement it precisely as described.
Use the specified indicators for entry/exit/regime filter as instructed.
"""

    # Director's strategic guidance (if available)
    director_section = ""
    if director_advice:
        director_section = f"""
═══════════════════════════════════════════════════════════════
👔 DIRECTOR'S STRATEGIC GUIDANCE (MUST FOLLOW)
═══════════════════════════════════════════════════════════════
{director_advice}

You MUST incorporate the Director's advice into your strategy design.
"""

    # Secretary brief injection (higher priority than Director's raw advice)
    brief_section = ""
    if brief:
        brief_section = f"""
═══════════════════════════════════════════════════════════════
📋 RESEARCH BRIEF (Secretary Synthesis — HIGHEST PRIORITY)
═══════════════════════════════════════════════════════════════
Theme: {brief.get('focus_theme', '')}
Priority indicators to use: {', '.join(brief.get('required_indicators', []))}
Avoid these patterns: {'; '.join(brief.get('avoid_patterns', []))}
Target exploration: {brief.get('exploration_target', '')}
---
{brief.get('brief_text', '')}

This brief synthesizes Director guidance + historical analysis. Follow it over generic mutation modes.
"""

    # F agent: custom indicators section (new signal sources beyond self.data)
    custom_inds = brief.get("custom_indicators", []) if brief else []
    custom_section = ""
    if custom_inds:
        custom_section = """
═══════════════════════════════════════════════════════════════
🔭 NEW CUSTOM INDICATORS (F agent — call as helper methods in your class)
═══════════════════════════════════════════════════════════════
These indicators are NOT in self.data. Include the helper method and call it in init():
"""
        for ci in custom_inds[:3]:
            custom_section += f"\n• {ci['name']} — {ci.get('description', '')}"
            custom_section += f"\n  Usage: {ci.get('usage', '')}"
            custom_section += f"\n  Signal: {ci.get('entry_hint', '')}\n"
        custom_section += "\n⚠️ NOT in self.data — call as methods: kama = self._calc_kama(self.data['Close'])"
        custom_section += "\nUsing these gives genuinely NEW signal sources beyond self.data columns."

    return f"""You are a Quantitative Research Director designing TQQQ (3x Leveraged Nasdaq) strategies.

{context}
{mutation}
{assignment_section}
{director_section}
{brief_section}
{custom_section}
{oos_section}

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

YOUR TASK — {current_assignment if current_assignment else champion_mutation}

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

⚠️ ENTRY FREQUENCY RULE (CRITICAL — most common failure):
Your entry condition MUST fire on at least 5% of trading days.
FORBIDDEN pattern (fires almost NEVER):
  bad:  RSI > 65 AND MACD > 0 AND Volume > 2.5x AND Aroon > 75  ← 4 ANDs = near-zero frequency
REQUIRED pattern (fires regularly):
  good: (RSI > 55 AND MACD > 0) OR (Volume > 1.5x AND Close > SMA20)  ← use OR between groups
  good: single strong condition + 1 regime filter  ← simpler is better
Rule of thumb: each AND halves your entry frequency. Max 2 AND conditions per entry signal.

⛔ ABSOLUTELY FORBIDDEN — these columns DO NOT EXIST in self.data:
  Yield curve: YC_Invert, YC2Y10Y, 2Y-10Y, 2Y, 10Y, US_10Y, curve_2y10y, YC_2Y10Y
  Fed/macro: FedWatch, FedCutProb, CME_CutProb, CME_3mo_cut_prob, FedPiv, FedFunds
  External data: VIX (only Sim_VIX exists), SPY, QQQ, any ticker not TQQQ
  Anything not in the INDICATOR MENU above = KeyError = WASTED iteration
  The Director's macro suggestions are IDEAS only — translate them to available indicators!

RESPOND WITH:
1. Strategy Name
2. Which Champion module(s) you KEPT vs CHANGED
3. State Machine Logic (what states, what transitions trigger signals)
4. Entry Signal Logic
5. Exit & Risk Logic
6. Short Selling Logic (if any)
7. Key Parameters

Keep response concise and actionable."""


def build_code_prompt(idea: str, strategy_id: int, indicator_menu: str,
                      custom_indicators=None) -> str:
    """Build the strategy code generation prompt with state machine patterns."""
    class_name = f"Strategy_Gen{strategy_id}"

    helper_section = ""
    if custom_indicators:
        helper_section = """
═══════════════════════════════════════════════════════════════
🔭 CUSTOM HELPER METHODS — CRITICAL USAGE RULES
═══════════════════════════════════════════════════════════════
⚠️  These indicators are NOT columns in self.data — self.data['KAMA'] will KeyError!
⚠️  You MUST copy the method into your class body and call it as a method:
    e.g.  kama = self._calc_kama(self.data['Close'])   # ✅ correct
          self.data['KAMA']                             # ❌ KeyError!
"""
        for ci in custom_indicators:
            usage = ci.get("usage", "")
            helper_section += f"\n# {ci['name']}: {ci.get('description', '')}\n"
            helper_section += f"# Usage: {usage}\n"
            helper_section += ci.get("method_code", "") + "\n"
        helper_section += "\nCopy the method(s) above into your class body. Call them in __init__ to pre-compute."

    return f"""Write a Python trading strategy class with STATE MACHINE logic.

STRATEGY IDEA:
{idea}

{indicator_menu}

═══════════════════════════════════════════════════════════════
🧬 REQUIRED CODE STRUCTURE — MUST FOLLOW EXACTLY
═══════════════════════════════════════════════════════════════
```
from strategy_base import BaseStrategy
import pandas as pd
import numpy as np

class {class_name}(BaseStrategy):
    # REQUIRED: __init__ with 3-5 KEY tunable parameters (thresholds, periods, multipliers)
    # Choose parameters that most affect THIS strategy's entry/exit/regime logic
    def __init__(self, bull_threshold=59.0, bear_threshold=42.0, atr_mult=1.5, rsi_period=14):
        super().__init__()
        self.bull_threshold = bull_threshold   # example: RVI bull level
        self.bear_threshold = bear_threshold   # example: RVI bear level
        self.atr_mult = atr_mult               # example: ATR stop multiplier
        self.rsi_period = rsi_period           # example: RSI lookback period

    # REQUIRED: get_params() — enables automatic parameter sweep
    def get_params(self) -> dict:
        return {{
            'bull_threshold': self.bull_threshold,
            'bear_threshold': self.bear_threshold,
            'atr_mult': self.atr_mult,
            'rsi_period': self.rsi_period,
        }}

    def init(self, data: pd.DataFrame):
        self.data = data
        # All indicators PRE-CALCULATED in self.data
        # Use self.bull_threshold etc. instead of hardcoded numbers!

    def _get_state(self) -> pd.Series:
        state = pd.Series('neutral', index=self.data.index)
        state[self.data['RVI_Refined'] > self.bull_threshold] = 'bull'  # use self.param!
        state[self.data['RVI_Refined'] < self.bear_threshold] = 'bear'
        return state

    def generate_signals(self) -> pd.Series:
        state = self._get_state()
        prev_state = state.shift(1).fillna('unknown')
        signals = pd.Series(0.0, index=self.data.index)
        position = 0
        for i in range(len(self.data)):
            curr = state.iloc[i]
            prev = prev_state.iloc[i]
            if prev in ('neutral', 'bear') and curr == 'bull':
                position = 1
            elif position == 1 and <exit_condition>:
                position = 0
            signals.iloc[i] = float(position)
        return signals.clip(-1, 1)

    def get_description(self) -> str:
        return f"Description — thresholds: {{self.bull_threshold}}/{{self.bear_threshold}}"
```
⚠️ ADAPT the parameter names/defaults to YOUR strategy's actual logic.
   Use self.your_param everywhere instead of hardcoded numbers.

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
- __init__ MUST have named parameters with defaults (3-5 key params)
- get_params() MUST return dict of all __init__ parameters
- Use self.param_name throughout — NEVER hardcode threshold numbers
- Indicators are ALREADY in self.data — do NOT recalculate them
- NO look-ahead: no shift(-1), no iloc[i+1], no future data
- Handle NaN with .fillna(0) or .bfill()
- Return pd.Series of floats -1.0 to 1.0 (negative = short)
- Use a for-loop to track position state (stateful logic)

⛔ COLUMNS THAT DO NOT EXIST (using them = immediate KeyError crash):
  YC_Invert, YC2Y10Y, 2Y, 10Y, US_10Y, curve_2y10y, FedWatch, FedCutProb,
  CME_CutProb, CME_3mo_cut_prob, FedPiv, FedFunds, VIX, SPY, QQQ
  If the strategy idea mentions macro/yield curve → use Sim_VIX or ATR_Pct as proxy instead.
{helper_section}
OUTPUT ONLY PYTHON CODE. No markdown, no explanations."""


def build_fix_prompt(code: str, error: str, strategy_id: int) -> str:
    class_name = f"Strategy_Gen{strategy_id}"

    # Special guidance for time_in_market=0 (never enters market)
    entry_fix_section = ""
    if "time_in_market=0" in error or "time_in_market" in error.lower():
        entry_fix_section = """
⚠️ ROOT CAUSE: Strategy NEVER ENTERS THE MARKET (time_in_market=0).
Your entry condition evaluates to False on EVERY single trading day.

DIAGNOSE & FIX — pick the applicable cause:
1. TOO MANY AND CONDITIONS — each AND halves frequency:
   bad:  (RSI > 65) & (MACD > 0) & (Vol > 2x) & (Aroon > 75)  → fires <1% of days
   fix:  (RSI > 52) & (MACD > 0)  OR  (Vol > 1.5x) & (Close > SMA_20)

2. THRESHOLDS TOO EXTREME — loosen them:
   bad:  RSI > 75, Volume > 3.0x, ZScore < -2.5, ATR_Pct > 5%
   fix:  RSI > 55, Volume > 1.3x, ZScore < -1.5, ATR_Pct > 1%

3. TRANSITION LOGIC BUG — shift(1) fills NaN on first row:
   bad:  (signal > 0) & (signal.shift(1) == 0)  → NaN comparison always False
   fix:  (signal > 0) & (signal.shift(1).fillna(0) == 0)

4. INDICATOR NAME TYPO — use exact column names from self.data:
   Always do: entry_mask = self.data['EXACT_COLUMN_NAME'] > threshold

REQUIRED: after fixing, your entry_mask.sum() must be > 20 (fires on 20+ days).
"""

    return f"""Fix this broken Python trading strategy.

BROKEN CODE:
{code}

ERROR:
{error}
{entry_fix_section}
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
    director_advice: Optional[str] = None,
    oos_warning: Optional[str] = None,
    notify_method: str = "file",
    brief: Optional[Dict] = None,
    current_assignment: Optional[str] = None,
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
        "composite": 0.0,
        "test_sharpe": 0.0,
        "test_cagr": 0.0,
        "test_max_dd": 0.0,
        "test_composite": 0.0,
        "error": None,
        "oos_warning": None,
    }

    # Step 1: Generate idea
    print("   💡 Generating idea...")
    idea_prompt = build_idea_prompt(history, indicator_menu,
                                    director_advice=director_advice,
                                    oos_warning=oos_warning,
                                    brief=brief,
                                    current_assignment=current_assignment)
    idea = llm.generate(idea_prompt, task="idea")
    if not idea:
        result["error"] = "LLM failed to generate idea"
        return result

    # Step 2: Generate code
    print("   💻 Generating code...")
    all_custom = brief.get("custom_indicators", []) if brief else []
    # Only inject method code for indicators explicitly mentioned in the idea
    mentioned = [ci for ci in all_custom if ci.get("name", "") in idea]
    custom_indicators = mentioned if mentioned else []
    code_prompt = build_code_prompt(idea, strategy_id, indicator_menu,
                                    custom_indicators=custom_indicators)
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

    # Step 5: Backtest (full + OOS train/test split)
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

        # Hard filter — reject strategies that don't meet minimum thresholds
        rejection = hard_filter(bt)
        if rejection:
            result["error"] = f"Hard filter: {rejection}"
            record_result(history, strategy_id, result["name"], idea,
                          bt.sharpe_ratio, bt.cagr, bt.max_drawdown, bt.calmar_ratio,
                          f"REJECTED: {rejection}", False)
            return result

        # Duplicate detection — reject functionally identical strategies (RVI clones)
        if is_duplicate_result(bt, history):
            msg = (f"Duplicate: Sharpe={bt.sharpe_ratio:.4f} CAGR={bt.cagr:.4f} "
                   f"MaxDD={bt.max_drawdown:.4f} already exists in history")
            print(f"   ⚠️ [DUP] {msg}")
            result["error"] = f"Duplicate result: {msg}"
            record_result(history, strategy_id, result["name"], idea,
                          bt.sharpe_ratio, bt.cagr, bt.max_drawdown, bt.calmar_ratio,
                          f"REJECTED: duplicate_strategy", False)
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

        # Train/Test OOS split
        print("   🧪 Running OOS validation...")
        train_sharpe = test_sharpe = test_cagr = test_max_dd = test_composite = 0.0
        try:
            # Reload strategy for OOS (fresh state)
            strategy_oos = sandbox.load_strategy(str(file_path), class_name)
            train_bt, test_bt = engine.run_with_oos(strategy_oos, train_ratio=0.8)
            train_sharpe = train_bt.sharpe_ratio
            test_sharpe = test_bt.sharpe_ratio
            test_cagr = test_bt.cagr
            test_max_dd = test_bt.max_drawdown
            test_composite = calculate_composite_score(
                test_bt.sharpe_ratio, test_bt.calmar_ratio,
                test_bt.sortino_ratio, test_bt.max_drawdown, test_bt.profit_factor
            )
            print(f"   🧪 OOS: Train Sharpe={train_sharpe:.2f}, Test Sharpe={test_sharpe:.2f}")
        except Exception as e:
            print(f"   ⚠️ OOS validation error (continuing): {str(e)[:60]}")

        # Composite score (on full backtest)
        composite = calculate_composite_score(
            bt.sharpe_ratio, bt.calmar_ratio, bt.sortino_ratio,
            bt.max_drawdown, bt.profit_factor
        )

        # OOS overfitting detection → set warning for next iteration
        if train_sharpe > 0 and test_sharpe < train_sharpe * 0.5:
            result["oos_warning"] = (
                f"Previous strategy {result['name']} showed OVERFITTING: "
                f"Train Sharpe={train_sharpe:.2f} but Test Sharpe={test_sharpe:.2f} "
                f"(test < 50% of train). Simplify your logic and use fewer parameters."
            )
            print(f"   ⚠️ OOS overfitting detected: train={train_sharpe:.2f} > test={test_sharpe:.2f}")

        result["success"] = True
        result["sharpe"] = bt.sharpe_ratio
        result["cagr"] = bt.cagr
        result["max_dd"] = bt.max_drawdown
        result["calmar"] = bt.calmar_ratio
        result["composite"] = composite
        result["test_sharpe"] = test_sharpe
        result["test_cagr"] = test_cagr
        result["test_max_dd"] = test_max_dd
        result["test_composite"] = test_composite
        result["idea"] = idea[:200]

        # ── Auto param sweep if strategy exposes get_params() ──
        if hasattr(strategy, 'get_params') and callable(strategy.get_params):
            try:
                from param_sweep import sweep_generated_strategy
                base_params = strategy.get_params()
                print(f"   🔬 Auto-sweep: {len(base_params)} params {list(base_params.keys())}")
                sweep_results = sweep_generated_strategy(strategy.__class__, data, base_params)
                if sweep_results and sweep_results[0]['sharpe'] > bt.sharpe_ratio + 0.05:
                    best = sweep_results[0]
                    print(f"   🏆 Sweep improved: Sharpe {bt.sharpe_ratio:.2f}→{best['sharpe']:.2f} "
                          f"CAGR {bt.cagr:.1%}→{best['cagr']:.1%} | params={best['params']}")
                    # Promote swept result
                    bt_sharpe = best['sharpe']
                    bt_cagr = best['cagr']
                    bt_maxdd = best['max_dd']
                    bt_calmar = best['calmar']
                    swept_composite = calculate_composite_score(
                        bt_sharpe, bt_calmar, best.get('sortino', bt_sharpe),
                        bt_maxdd, best.get('profit_factor', 1.0)
                    )
                    result["sharpe"] = bt_sharpe
                    result["cagr"] = bt_cagr
                    result["max_dd"] = bt_maxdd
                    result["calmar"] = bt_calmar
                    result["composite"] = swept_composite
                    result["swept_params"] = best['params']
                    composite = swept_composite
                else:
                    print(f"   ✅ Sweep done — base params already near-optimal")
            except Exception as _se:
                print(f"   ⚠️ Auto-sweep failed: {_se}")

        analysis = bt.get_failure_analysis()
        record_result(history, strategy_id, result["name"], idea,
                      result["sharpe"], result["cagr"], result["max_dd"], result["calmar"],
                      analysis, True,
                      composite=composite,
                      test_sharpe=test_sharpe, test_cagr=test_cagr,
                      test_max_dd=test_max_dd, test_composite=test_composite)

        # Check hall of fame — use result[] values (may be swept, higher than bt)
        check_hall_of_fame(result["name"], result["sharpe"], result["max_dd"],
                          result["cagr"], result["calmar"], idea)

        # Instant Telegram + git push on new record (composite score)
        if composite > history.get("best_composite", 0):
            record_msg = (
                f"🏆 NEW RECORD! {result['name']}\n"
                f"Composite: {composite:.4f}\n"
                f"Sharpe={bt.sharpe_ratio:.2f} CAGR={bt.cagr:.1%} MaxDD={bt.max_drawdown:.1%}\n"
                f"Calmar={bt.calmar_ratio:.2f} Sortino={bt.sortino_ratio:.2f}\n"
                f"--- OOS Test ---\n"
                f"Test Sharpe={test_sharpe:.2f} Test CAGR={test_cagr:.1%} Test MaxDD={test_max_dd:.1%}\n"
                f"Test Composite={test_composite:.4f}"
            )
            if notify_method == "telegram":
                send_telegram(record_msg)
            strategy_file = GENERATED_DIR / f"strategy_gen_{strategy_id}.py"
            git_push(
                f"[record] {result['name']}: Comp={composite:.4f} Sharpe={result['sharpe']:.2f} CAGR={result['cagr']:.1%}",
                files=[strategy_file, HISTORY_FILE, HALL_OF_FAME_FILE]
            )

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
    rankable = [s for s in strategies if is_rankable(s)]
    top5 = sorted(rankable, key=lambda x: x.get("sharpe", 0), reverse=True)[:5]
    recent10 = strategies[-10:]
    hof = load_hall_of_fame()

    best_composite = history.get('best_composite', history.get('best_calmar', history.get('best_sharpe', 0)))
    report = f"""
{'='*55}
📊 TQQQ Strategy Discovery Report
   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*55}

📈 Overall
  Total iterations: {total}
  Successful: {len(successful)} ({len(successful)/total*100:.0f}%)
  Best: {history['best_strategy']} (Composite: {best_composite:.4f})
  Hall of Fame: {len(hof)} strategies

  This session: {session_stats['iterations']} iters, {session_stats['successes']} ok
  Duration: {session_stats['duration']}
  {session_stats['llm_stats']}

🏆 Top 5 Strategies (by Sharpe ratio)
{'─'*55}"""
    for i, s in enumerate(top5, 1):
        cs = s.get('composite', 0)
        report += (f"\n  #{i} {s['name']}: Comp={cs:.4f} Sharpe={s['sharpe']:.2f} "
                   f"CAGR={s['cagr']:.1%} MaxDD={s['max_dd']:.1%}")
        if s.get("test_sharpe"):
            report += f" | OOS Sharpe={s['test_sharpe']:.2f}"

    if hof:
        report += f"\n\n🥇 Hall of Fame ({len(hof)})\n{'─'*55}"
        for h in hof[:5]:
            report += f"\n  {h['name']}: Calmar={h.get('calmar',0):.2f} Sharpe={h['sharpe']:.2f} MaxDD={h['max_dd']:.1%}"

    report += f"\n\n📝 Last 10 Iterations\n{'─'*55}"
    for s in recent10:
        st = "✅" if s.get("success") else "❌"
        if s.get("success"):
            cs = s.get('composite', 0)
            info = f"Comp={cs:.4f} Sharpe={s['sharpe']:.2f} CAGR={s['cagr']:.1%} MaxDD={s['max_dd']:.1%}"
        else:
            info = s.get("failure_analysis", "")[:40]
        report += f"\n  {st} {s['name']}: {info}"

    report += f"\n\n{'='*55}\n"
    return report


def send_telegram(report: str):
    try:
        import requests
        bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
        chat_id = os.getenv("TELEGRAM_CHAT_ID")
        if not bot_token or not chat_id:
            return
        # Split into pages if > 4000 chars
        PAGE_SIZE = 3800
        lines = report.split('\n')
        pages, current, cur_len = [], [], 0
        for line in lines:
            if cur_len + len(line) + 1 > PAGE_SIZE and current:
                pages.append('\n'.join(current))
                current, cur_len = [line], len(line) + 1
            else:
                current.append(line)
                cur_len += len(line) + 1
        if current:
            pages.append('\n'.join(current))
        total = len(pages)
        for idx, page in enumerate(pages, 1):
            header = f"[{idx}/{total}] " if total > 1 else ""
            resp = requests.post(
                f"https://api.telegram.org/bot{bot_token}/sendMessage",
                data={"chat_id": chat_id, "text": header + page},
                timeout=15
            )
            if resp.status_code == 200:
                print(f"📱 Telegram [{idx}/{total}] sent")
            if total > 1 and idx < total:
                time.sleep(1)
    except Exception as e:
        print(f"⚠️ Telegram failed: {e}")


def git_push(message: str = None, files: list = None):
    """Commit changed files and push to remote. Silently skips if nothing to do."""
    import subprocess
    repo_dir = Path(__file__).parent
    try:
        # Stage files
        if files:
            for f in files:
                subprocess.run(['git', 'add', str(f)], cwd=repo_dir,
                               capture_output=True, timeout=30)
        else:
            subprocess.run(['git', 'add', '-u'], cwd=repo_dir,
                           capture_output=True, timeout=30)
        # Commit
        msg = message or f"[auto] iteration update {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        commit = subprocess.run(
            ['git', 'commit', '-m', msg],
            cwd=repo_dir, capture_output=True, text=True, timeout=30
        )
        if commit.returncode != 0 and 'nothing to commit' not in commit.stdout + commit.stderr:
            print(f"   ⚠️ git commit: {commit.stderr[:80]}")
            return
        if commit.returncode == 0:
            print(f"   📝 git committed: {msg[:60]}")
        # Push with token if available
        token = os.getenv('GITHUB_TOKEN', '')
        url_res = subprocess.run(['git', 'remote', 'get-url', 'origin'],
                                 cwd=repo_dir, capture_output=True, text=True, timeout=10)
        remote_url = url_res.stdout.strip()
        if token and remote_url.startswith('https://') and '@' not in remote_url:
            authed = remote_url.replace('https://', f'https://jackcanhelp:{token}@')
            push = subprocess.run(['git', 'push', authed, 'HEAD'],
                                  cwd=repo_dir, capture_output=True, text=True, timeout=60)
        else:
            push = subprocess.run(['git', 'push'],
                                  cwd=repo_dir, capture_output=True, text=True, timeout=60)
        if push.returncode == 0:
            print("   ☁️ git pushed to remote")
        else:
            print(f"   ⚠️ git push failed: {push.stderr[:80]}")
    except Exception as e:
        print(f"   ⚠️ git_push error: {e}")


# ═══════════════════════════════════════════════════════════════
# Director (投資總監) — Strategic review & guidance
# ═══════════════════════════════════════════════════════════════
def get_director_advice(llm: 'LLMClient', history: Dict) -> Optional[str]:
    """
    AI Director of Quantitative Research reviews top strategies and recent failures,
    then provides tactical guidance for the next iterations.
    """
    strategies = history.get("strategies", [])
    rankable = [s for s in strategies if is_rankable(s)]
    top5 = sorted(rankable, key=lambda x: x.get("sharpe", 0), reverse=True)[:5]
    recent_failures = [s for s in strategies[-20:] if not s.get("success")][-5:]
    recent_all = strategies[-10:]

    if not top5:
        return None

    # Build review report for director
    report = "═══ CURRENT TOP STRATEGIES (ranked by Sharpe ratio) ═══\n"
    for i, s in enumerate(top5, 1):
        report += (
            f"\n--- Rank {i}: {s['name']} ---\n"
            f"Composite: {s.get('composite', 0):.4f} | Sharpe: {s['sharpe']:.2f} | "
            f"CAGR: {s.get('cagr', 0):.1%} | MaxDD: {s['max_dd']:.1%} | "
            f"Calmar: {s.get('calmar', 0):.2f}\n"
        )
        if s.get("test_sharpe"):
            report += f"OOS: Test Sharpe={s['test_sharpe']:.2f}, Test CAGR={s['test_cagr']:.1%}\n"
        report += f"Idea: {s.get('idea', '')[:200]}\n"

    report += "\n═══ RECENT 10 ITERATIONS ═══\n"
    for s in recent_all:
        st = "✅" if s.get("success") else "❌"
        report += f"  {st} {s['name']}: Sharpe={s['sharpe']:.2f}, MaxDD={s['max_dd']:.1%}\n"

    if recent_failures:
        report += "\n═══ RECENT FAILURE PATTERNS ═══\n"
        for s in recent_failures:
            report += f"  - {s['name']}: {s.get('failure_analysis', '')[:100]}\n"

    report += f"\nTotal iterations: {history['total_iterations']}\n"
    report += f"Best strategy: {history.get('best_strategy', 'N/A')} (Composite: {history.get('best_composite', 0):.4f})\n"

    director_prompt = f"""You are the Director of Quantitative Research, reviewing your team's TQQQ strategy evolution progress.

{report}

Based on the above data, provide a concise strategic directive (2-4 sentences) for the next batch of iterations:

1. What SPECIFIC weakness do you see in the current top strategies? (e.g., high drawdown during crisis, low CAGR, overfitting signs)
2. What SPECIFIC indicator, risk management technique, or logic pattern should the team explore NEXT?
3. What approaches should be AVOIDED based on recent failure patterns?

⚠️ CONSTRAINT — only recommend indicators from this AVAILABLE list:
  Trend: SMA_20/50/200, EMA_20/50/200, MACD, MACD_signal, MACD_hist, ADX, Supertrend, Ichimoku_conv, Ichimoku_base, DI_Plus, DI_Minus, DI_Diff, Aroon_Up, Aroon_Down, Aroon_Osc, TRIX, PPO
  Momentum: RSI, RSI_7, Stoch_K, Stoch_D, Williams_R, CCI, ROC, ROC_5, ROC_20, MFI, RVI, RVI_Refined, RVI_State, TSI, AO, UO
  Volatility: ATR, ATR_Pct, BB_upper/middle/lower/width/pct, KC_upper/lower, Donchian_upper/lower, Sim_VIX, HV_10, HV_30, BB_Squeeze, VoV
  Volume: OBV, OBV_SMA, CMF, Force_Index, Vol_Ratio, VWAP_Ratio
  Structure: Drawdown, Days_Up, Days_Down, Gap_Pct, ZScore, SMA50_Dist, SMA200_Dist
  Elder: Elder_Bull, Elder_Bear
  ML Regime (Phase 3A): HMM_Regime (0=bear/1=neutral/2=bull), HMM_Prob_Bull (0-1 bull probability), GARCH_Vol (annualized conditional volatility %), CP_Distance (days since last structural break)
  New Momentum (Phase 3B): QQE, STC, KDJ_K, KDJ_D, KDJ_J, CTI, SMI, Squeeze_Pro_Hist, Squeeze_Pro_On
DO NOT suggest: yield curve, Fed funds rate, VIX (external), SPY, QQQ, or any macro data.
Be specific and actionable.
DO NOT write code. ONLY provide strategic direction."""

    advice = llm.generate(director_prompt, task="director")
    if advice:
        print(f"\n   👔 [投資總監] 戰術指導：{advice[:200]}...")
    return advice


# ═══════════════════════════════════════════════════════════════
# Slow Path — full A→B→C1→C2 pipeline
# ═══════════════════════════════════════════════════════════════
def run_slow_path(analyst: InternalAnalyst, researcher: SolutionResearcher,
                  external_researcher: ExternalResearcher,
                  validator: AnalysisValidator, secretary: Secretary,
                  llm: LLMClient, history: Dict,
                  director_advice: Optional[str]) -> tuple:
    """
    Full A→B→C1→C2→F pipeline. Returns (brief, execution_queue).

    A: analyze (pure Python) + explain (LLM, task=director)
    B: research proposals (LLM, task=director)
    C1: validate indicators (pure Python, LLM only if corrections needed)
    C2: secretary brief + execution_queue (LLM, task=secretary)
    F: external researcher proposes new indicators (LLM, task=director)

    Total LLM calls: ~4-5 (A.explain, B.research, [C1 correction], C2.brief, F.propose)
    """
    print("   🔍 [A] Problem Explorer: analyzing history...")
    stats_report = analyst.analyze(history)

    print("   🧠 [A] Generating LLM narrative (root cause analysis)...")
    a_narrative = analyst.explain(stats_report, llm)
    print(f"   🔍 [A] Narrative: {a_narrative[:120]}...")

    print("   🔬 [B] Solution Researcher: proposing concrete approaches...")
    b_proposals = researcher.research(
        a_narrative, stats_report, InternalAnalyst.KNOWN_INDICATORS, llm
    )
    n_proposals = len(b_proposals.get("proposals", []))
    print(f"   🔬 [B] {n_proposals} proposals generated")

    print("   ✅ [C1] Validator: checking proposal indicator names...")
    c1_result = validator.validate(
        b_proposals, InternalAnalyst.KNOWN_INDICATORS, llm
    )
    if c1_result.get("approved"):
        print("   ✅ [C1] All indicators valid — pure Python pass")
    else:
        corrections = len(c1_result.get("corrections", []))
        print(f"   ⚠️ [C1] {corrections} corrections applied")

    print("   📋 [C2] Secretary: creating execution brief...")
    brief = secretary.create_brief(
        director_advice, stats_report, llm,
        researcher_proposals=c1_result.get("validated_proposals"),
        validator_result=c1_result,
    )

    print("   🔭 [F] External Researcher: proposing new indicators...")
    f_result = external_researcher.propose(stats_report, InternalAnalyst.KNOWN_INDICATORS, llm)
    brief["custom_indicators"] = f_result.get("new_indicators", [])
    n_custom = len(brief["custom_indicators"])
    print(f"   🔭 [F] {n_custom} custom indicators ready for injection")

    eq = brief.get("execution_queue", [])
    print(f"   📋 Slow path complete — execution queue: {len(eq)} assignments")
    return brief, eq


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

        results = run_crossover(data, top_n=5, max_combos=300)

        for r in results[:3]:
            name = r["name"]
            # Skip if already in history (crossover runs every 25 iters)
            if any(s.get("name") == name for s in history.get("strategies", [])):
                print(f"   ⏭ {name} already recorded, skipping")
                continue
            # Apply same quality gates as LLM-generated strategies
            from backtest import BacktestResult
            _bt_mock = type('_BT', (), {
                'sharpe_ratio': r["sharpe"], 'cagr': r["cagr"],
                'max_drawdown': r["max_dd"], 'calmar_ratio': r["calmar"],
                'total_trades': r.get("trades", 99),
                'time_in_market': r.get("time_in_market", 0.5),
            })()
            if hard_filter(_bt_mock):
                print(f"   ⚠️ {name} failed hard filter, skipping")
                continue
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
    print("🚀 TQQQ Strategy Discovery Engine v4")
    print(f"   🧬 Champion DNA + State Machine + Crossover")
    print(f"   🔑 Groq 5-Key Pool: idea/director/secretary=K1,K2 | code=K3,K4 | fix=K5")
    print(f"   🤖 Multi-Agent: Director→InternalAnalyst→Secretary→IdeaGen")
    print(f"   👔 Director review every {DIRECTOR_INTERVAL} iters (stagnation: {STAGNATION_THRESHOLD})")
    print(f"   📋 Brief refresh every {25} iters (Secretary synthesis)")
    print(f"   📊 Composite Score ranking + OOS validation")
    print(f"   🛡️ Hard Filter: MaxDD>{HARD_FILTER_MAX_DD:.0%}, Trades>={HARD_FILTER_MIN_TRADES}, Sharpe>={HARD_FILTER_MIN_SHARPE}")
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

    # ─── Multi-agent pipeline components (ABCDEF) ───
    analyst = InternalAnalyst()           # A: stats + narrative
    researcher = SolutionResearcher()     # B: concrete proposals
    validator = AnalysisValidator()       # C1: indicator validation
    secretary = Secretary()               # C2: brief + execution_queue
    external_researcher = ExternalResearcher()  # F: new indicator proposals
    result_checker = ResultChecker()      # E: monitors iteration results

    # ─── Execution queue state (from slow path C2 output) ───
    execution_queue: List[str] = []   # Directed assignments from C2
    eq_idx: int = 0                   # Current position in queue
    e_flag_count: int = 0             # E flags accumulated since last slow path
    E_FLAG_THRESHOLD = 3              # Trigger slow path after N E-flag events

    # ─── Phase 0: Establish Champion baseline ───
    if not args.skip_baseline:
        run_champion_baseline(engine, data, history)

    session_start = datetime.now()
    session_successes = 0
    consec_api_fail = 0
    iters_since_improvement = 0
    best_composite_at_start = history.get("best_composite", 0)
    director_advice = None  # Current director guidance (injected into idea prompt)
    oos_warning = None      # OOS overfitting warning (injected into next idea prompt)
    current_brief = None    # Secretary Brief (highest priority guidance for idea gen)

    for i in range(1, args.iterations + 1):
        strategy_id = history["total_iterations"] + 1
        print(f"\n[{i}/{args.iterations}] Iteration {strategy_id}")

        # ─── Stagnation detection → early director call + slow path ───
        if (iters_since_improvement >= STAGNATION_THRESHOLD
                and i % DIRECTOR_INTERVAL != 0):
            print(f"\n   ⚠️ [停滯偵測] 已 {iters_since_improvement} 輪未突破，觸發慢速路徑 A→B→C1→C2")
            director_advice = get_director_advice(llm, history)
            current_brief, execution_queue = run_slow_path(
                analyst, researcher, external_researcher, validator, secretary, llm, history, director_advice
            )
            eq_idx = 0
            e_flag_count = 0
            if director_advice and args.notify == "telegram":
                brief_summary = current_brief.get("focus_theme", "") if current_brief else ""
                eq_count = len(execution_queue)
                send_telegram(
                    f"⚠️ 【停滯警報：{iters_since_improvement} 代未突破】\n"
                    f"👔 總監緊急指導：\n{director_advice[:300]}\n"
                    f"📋 研究簡報主題：{brief_summary[:150]}\n"
                    f"🎯 執行隊列：{eq_count} 項任務"
                )
            iters_since_improvement = 0

        # ─── Director review (every N iterations) + slow path A→B→C1→C2 ───
        if i > 1 and i % DIRECTOR_INTERVAL == 0:
            print(f"\n   👔 [定期總監審查] 第 {i} 輪 — 觸發慢速路徑")
            director_advice = get_director_advice(llm, history)
            current_brief, execution_queue = run_slow_path(
                analyst, researcher, external_researcher, validator, secretary, llm, history, director_advice
            )
            eq_idx = 0
            e_flag_count = 0
            if director_advice and args.notify == "telegram":
                brief_summary = current_brief.get("focus_theme", "") if current_brief else ""
                eq_count = len(execution_queue)
                send_telegram(
                    f"👔 【第 {strategy_id} 代 — 總監戰術指導】\n{director_advice[:300]}\n"
                    f"📋 研究簡報主題：{brief_summary[:150]}\n"
                    f"🎯 執行隊列：{eq_count} 項任務"
                )

        # ─── Crossover round (every N iterations) ───
        if i > 1 and i % args.crossover_every == 0:
            run_crossover_round(data, history)

        # ─── Parameter sweep (every N iterations) ───
        if i > 1 and i % args.sweep_every == 0:
            run_param_sweep_round(data, history)

        # ─── Consume next assignment from execution queue (fast path) ───
        current_assignment = None
        if execution_queue and eq_idx < len(execution_queue):
            current_assignment = execution_queue[eq_idx]
            eq_idx += 1
            print(f"   🎯 Queue assignment [{eq_idx}/{len(execution_queue)}]: {current_assignment[:70]}...")

        # ─── Normal LLM-generated strategy iteration (D) ───
        result = run_iteration(llm, engine, data, history, indicator_menu, strategy_id,
                               director_advice=director_advice,
                               oos_warning=oos_warning,
                               notify_method=args.notify,
                               brief=current_brief,
                               current_assignment=current_assignment)

        # ─── E: Result checker — detect patterns, trigger slow path if needed ───
        e_result = result_checker.check(result)
        if e_result["trigger_slow_path"]:
            e_flag_count += 1
            print(f"   ⚡ [E] Flag detected: {e_result['summary']} (count={e_flag_count}/{E_FLAG_THRESHOLD})")
            if e_flag_count >= E_FLAG_THRESHOLD:
                print(f"   ⚠️ [E] Threshold reached — triggering slow path A→B→C1→C2")
                director_advice = get_director_advice(llm, history)
                current_brief, execution_queue = run_slow_path(
                    analyst, researcher, external_researcher, validator, secretary, llm, history, director_advice
                )
                eq_idx = 0
                e_flag_count = 0
                if current_brief and args.notify == "telegram":
                    brief_summary = current_brief.get("focus_theme", "")
                    send_telegram(
                        f"⚡ 【E 觸發慢速路徑】\n原因：{e_result['summary']}\n"
                        f"📋 新簡報：{brief_summary[:150]}\n"
                        f"🎯 執行隊列：{len(execution_queue)} 項任務"
                    )

        # Consume OOS warning (only inject once)
        oos_warning = None

        if result["success"]:
            composite = result.get("composite", 0)
            print(f"   ✅ Composite={composite:.4f} Sharpe={result['sharpe']:.2f} "
                  f"CAGR={result['cagr']:.1%} MaxDD={result['max_dd']:.1%}")
            if result.get("test_sharpe"):
                print(f"      OOS: Test Sharpe={result['test_sharpe']:.2f} "
                      f"Test CAGR={result['test_cagr']:.1%}")
            session_successes += 1
            consec_api_fail = 0

            # Propagate OOS overfitting warning for next iteration
            if result.get("oos_warning"):
                oos_warning = result["oos_warning"]

            # Track improvement for stagnation detection (now using composite)
            current_best = history.get("best_composite", 0)
            if current_best > best_composite_at_start:
                iters_since_improvement = 0
                best_composite_at_start = current_best
            else:
                iters_since_improvement += 1

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
                iters_since_improvement += 1

        # Periodic report + brief refresh + git push
        if i % args.report_every == 0:
            stats = {
                "iterations": i,
                "successes": session_successes,
                "duration": str(datetime.now() - session_start).split('.')[0],
                "llm_stats": llm.get_stats(),
            }
            report = generate_report(history, stats)
            print(report)
            with open("latest_report.txt", "w") as f:
                f.write(report)
            if args.notify == "telegram":
                send_telegram(report)
            git_push(
                f"[auto] report iter={strategy_id} ok={session_successes}",
                files=[HISTORY_FILE, HALL_OF_FAME_FILE, Path("latest_report.txt")]
            )
            # Brief is refreshed by slow path (every 50 iters or stagnation/E-flag).
            # Avoid calling secretary.create_brief() here — without researcher_proposals
            # it would produce an inferior brief (empty execution_queue) that overwrites
            # the valid brief from the last slow path.

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

    with open("latest_report.txt", "w") as f:
        f.write(report)

    if args.notify == "telegram":
        send_telegram(report)

    git_push(
        f"[final] iter={strategy_id} ok={session_successes}",
        files=[HISTORY_FILE, HALL_OF_FAME_FILE, Path("latest_report.txt")]
    )


if __name__ == "__main__":
    main()
