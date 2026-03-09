"""
Multi-Agent Brief System — ABCDE Pipeline
==========================================
A  (InternalAnalyst): Pure Python stats + LLM narrative WHY analysis
B  (SolutionResearcher): Proposes concrete strategy approaches
C1 (AnalysisValidator): Validates B's proposals for indicator correctness
C2 (Secretary): Creates Brief JSON + execution_queue (3 directed assignments)
E  (ResultChecker): Monitors iteration results, triggers slow path

Pipeline:
  Slow path: A.analyze → A.explain → B.research → C1.validate → C2.brief → execution_queue
  Fast path: pop assignment → D (run_iteration) → E.check
"""

import json
import re
from collections import Counter
from typing import Dict, List, Optional
from datetime import datetime


def _parse_llm_json(raw: Optional[str]) -> Optional[dict]:
    """Strip markdown fences from LLM response and parse as JSON. Returns None on failure."""
    if not raw:
        return None
    try:
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r'^```(?:json)?\s*', '', cleaned, flags=re.MULTILINE)
            cleaned = re.sub(r'\s*```\s*$', '', cleaned)
        match = re.search(r'\{[\s\S]+\}', cleaned)
        if match:
            cleaned = match.group()
        return json.loads(cleaned)
    except (json.JSONDecodeError, ValueError):
        return None


# ═══════════════════════════════════════════════════════════════
# InternalAnalyst (A) — Pure Python stats + LLM narrative
# ═══════════════════════════════════════════════════════════════
class InternalAnalyst:
    """
    A: Analyzes history_of_thoughts.json to extract statistical insights,
    then generates an LLM narrative explaining WHY the system is failing.

    analyze() = pure Python stats (deterministic, fast, no LLM)
    explain() = LLM root cause narrative (task="director")
    """

    # All known indicators for usage tracking
    KNOWN_INDICATORS = [
        "SMA_20", "SMA_50", "SMA_200", "EMA_20", "EMA_50", "EMA_200",
        "MACD", "MACD_signal", "MACD_hist", "ADX", "Supertrend",
        "Ichimoku_conv", "Ichimoku_base",
        "RSI", "RSI_7", "Stoch_K", "Stoch_D", "Williams_R", "CCI",
        "ROC", "ROC_5", "ROC_20", "MFI",
        "ATR", "ATR_Pct", "BB_upper", "BB_middle", "BB_lower",
        "BB_width", "BB_pct", "BB_Squeeze", "KC_upper", "KC_lower",
        "Donchian_upper", "Donchian_lower", "Sim_VIX", "HV_10", "HV_30", "VoV",
        "RVI", "RVI_Refined", "RVI_State",
        "OBV", "OBV_SMA", "CMF", "Force_Index", "Vol_Ratio", "VWAP_Ratio",
        "DI_Plus", "DI_Minus", "DI_Diff",
        "Aroon_Up", "Aroon_Down", "Aroon_Osc",
        "TRIX", "PPO", "TSI", "AO", "UO",
        "Elder_Bull", "Elder_Bear",
        "Drawdown", "Days_Up", "Days_Down", "Gap_Pct",
        "ZScore", "SMA50_Dist", "SMA200_Dist",
        # Phase 3A: ML Regime
        "HMM_Regime", "HMM_Prob_Bull", "GARCH_Vol", "CP_Distance",
        # Phase 3B: New Momentum (pandas-ta-classic)
        "QQE", "STC", "KDJ_K", "KDJ_D", "KDJ_J", "CTI", "SMI",
        "Squeeze_Pro_Hist", "Squeeze_Pro_On",
    ]

    # Category mapping for usage analysis
    CATEGORIES = {
        "Trend": ["SMA_20","SMA_50","SMA_200","EMA_20","EMA_50","EMA_200",
                  "MACD","MACD_signal","MACD_hist","ADX","Supertrend",
                  "Ichimoku_conv","Ichimoku_base"],
        "Momentum": ["RSI","RSI_7","Stoch_K","Stoch_D","Williams_R","CCI",
                     "ROC","ROC_5","ROC_20","MFI","RVI","RVI_Refined","RVI_State"],
        "Volatility": ["ATR","ATR_Pct","BB_upper","BB_middle","BB_lower",
                       "BB_width","BB_pct","BB_Squeeze","KC_upper","KC_lower",
                       "Donchian_upper","Donchian_lower","Sim_VIX","HV_10","HV_30","VoV"],
        "Volume": ["OBV","OBV_SMA","CMF","Force_Index","Vol_Ratio","VWAP_Ratio"],
        "TrendQuality": ["DI_Plus","DI_Minus","DI_Diff","Aroon_Up","Aroon_Down","Aroon_Osc","TRIX","PPO"],
        "AdvancedMomentum": ["TSI","AO","UO","Elder_Bull","Elder_Bear"],
        "Structure": ["Drawdown","Days_Up","Days_Down","Gap_Pct","ZScore","SMA50_Dist","SMA200_Dist"],
        "MLRegime": ["HMM_Regime","HMM_Prob_Bull","GARCH_Vol","CP_Distance"],
        "NewMomentum": ["QQE","STC","KDJ_K","KDJ_D","KDJ_J","CTI","SMI",
                        "Squeeze_Pro_Hist","Squeeze_Pro_On"],
    }

    def analyze(self, history: Dict) -> str:
        """
        Perform statistical analysis of strategy history.
        Returns a plain-text report string for use by explain() and Secretary.
        """
        strategies = history.get("strategies", [])
        if not strategies:
            return "No strategy history available yet."

        total = len(strategies)
        recent_n = min(50, total)
        recent = strategies[-recent_n:]
        window_100 = strategies[-100:]

        # ── 1. Success rates ──
        overall_success = sum(1 for s in strategies if s.get("success")) / total
        recent_success = sum(1 for s in recent if s.get("success")) / recent_n

        # ── 2. Failure pattern analysis ──
        failures = [s for s in recent if not s.get("success")]
        failure_patterns: Dict[str, int] = {}
        for s in failures:
            reason = s.get("failure_analysis", "")
            if "MaxDD" in reason or "drawdown" in reason.lower():
                failure_patterns["MaxDD_too_deep"] = failure_patterns.get("MaxDD_too_deep", 0) + 1
            elif "trade" in reason.lower() or "Only" in reason:
                failure_patterns["too_few_trades"] = failure_patterns.get("too_few_trades", 0) + 1
            elif "Exposure" in reason or "time_in_market" in reason.lower():
                failure_patterns["low_exposure"] = failure_patterns.get("low_exposure", 0) + 1
            elif "Sharpe" in reason:
                failure_patterns["low_sharpe"] = failure_patterns.get("low_sharpe", 0) + 1
            elif "KeyError" in reason or "not in" in reason.lower():
                failure_patterns["key_error"] = failure_patterns.get("key_error", 0) + 1
            elif "look-ahead" in reason.lower() or "bias" in reason.lower():
                failure_patterns["look_ahead_bias"] = failure_patterns.get("look_ahead_bias", 0) + 1
            elif "failed to generate" in reason.lower() or "LLM" in reason:
                failure_patterns["llm_failure"] = failure_patterns.get("llm_failure", 0) + 1
            elif "duplicate" in reason.lower():
                failure_patterns["duplicate_strategy"] = failure_patterns.get("duplicate_strategy", 0) + 1
            else:
                failure_patterns["other"] = failure_patterns.get("other", 0) + 1

        sorted_failures = sorted(failure_patterns.items(), key=lambda x: x[1], reverse=True)

        # ── 3. Indicator usage frequency (last 100 strategies) ──
        indicator_counts: Dict[str, int] = {ind: 0 for ind in self.KNOWN_INDICATORS}
        for s in window_100:
            idea = s.get("idea", "")
            for ind in self.KNOWN_INDICATORS:
                if ind in idea:
                    indicator_counts[ind] += 1

        sorted_by_usage = sorted(indicator_counts.items(), key=lambda x: x[1], reverse=True)
        overused = [(k, v) for k, v in sorted_by_usage if v >= 8][:10]
        underused = [(k, v) for k, v in sorted_by_usage if v <= 2][:15]

        # ── 4. Score progression (last 20 successful) ──
        recent_successful = [s for s in strategies[-40:] if s.get("success")]
        if len(recent_successful) >= 4:
            half = len(recent_successful) // 2
            first_half = recent_successful[:half]
            second_half = recent_successful[half:]
            avg_early = sum(s.get("sharpe", 0) for s in first_half) / len(first_half)
            avg_late = sum(s.get("sharpe", 0) for s in second_half) / len(second_half)
            score_trend = "IMPROVING" if avg_late > avg_early * 1.02 else "STAGNATING"
            trend_detail = f" (early_avg={avg_early:.4f} → recent_avg={avg_late:.4f})"
        else:
            score_trend = "INSUFFICIENT_DATA"
            trend_detail = ""

        # ── 5. Top strategies and their dominant indicators ──
        rankable = [s for s in strategies if s.get("success") and s.get("sharpe", 0) > 0]
        top5 = sorted(rankable, key=lambda x: x.get("sharpe", 0), reverse=True)[:5]

        top5_info = []
        for s in top5:
            idea = s.get("idea", "")
            used_inds = [ind for ind in self.KNOWN_INDICATORS if ind in idea]
            top5_info.append({
                "name": s["name"],
                "composite": s.get("composite", 0),
                "sharpe": s.get("sharpe", 0),
                "max_dd": s.get("max_dd", 0),
                "indicators": used_inds[:6],
            })

        # ── 6. Consecutive failures recently ──
        consec_failures = 0
        for s in reversed(strategies[-20:]):
            if not s.get("success"):
                consec_failures += 1
            else:
                break

        # ── 7. Category usage totals ──
        cat_usage: Dict[str, int] = {}
        for cat, inds in self.CATEGORIES.items():
            cat_usage[cat] = sum(indicator_counts.get(ind, 0) for ind in inds)

        sorted_cats = sorted(cat_usage.items(), key=lambda x: x[1])

        # ── 7b. Indicator diversity entropy (Phase 3C) ──
        import math as _math
        _counts = [v for v in indicator_counts.values() if v > 0]
        _total_uses = sum(_counts)
        if _total_uses > 0 and len(_counts) > 1:
            _probs = [c / _total_uses for c in _counts]
            _entropy = -sum(p * _math.log2(p) for p in _probs)
            _max_entropy = _math.log2(len(indicator_counts))
            diversity_pct = _entropy / _max_entropy if _max_entropy > 0 else 0.0
        else:
            diversity_pct = 0.0
        # New pool indicators with zero usage (Phase 3A/3B) — immediate candidates
        new_pool_inds = (self.CATEGORIES.get("MLRegime", []) +
                         self.CATEGORIES.get("NewMomentum", []))
        untried_new = [ind for ind in new_pool_inds if indicator_counts.get(ind, 0) == 0]

        # ── Compose report ──
        report = f"""=== INTERNAL ANALYST REPORT ===
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}
Total strategies analyzed: {total}

--- SUCCESS RATES ---
Overall: {overall_success:.1%} ({sum(1 for s in strategies if s.get('success'))}/{total})
Recent {recent_n}: {recent_success:.1%} ({sum(1 for s in recent if s.get('success'))}/{recent_n})
Consecutive recent failures: {consec_failures}

--- FAILURE PATTERNS (last {recent_n}) ---"""
        if sorted_failures:
            total_failures = len(failures)
            for pattern, count in sorted_failures:
                pct = count / max(total_failures, 1) * 100
                report += f"\n  {pattern}: {count} ({pct:.0f}%)"
        else:
            report += "\n  No failures in recent window"

        report += f"""

--- SCORE PROGRESSION ---
Trend: {score_trend}{trend_detail}

--- TOP 5 STRATEGIES ---"""
        for t in top5_info:
            inds = ", ".join(t["indicators"]) if t["indicators"] else "N/A"
            report += (f"\n  {t['name']}: Comp={t['composite']:.4f} Sharpe={t['sharpe']:.2f} "
                       f"MaxDD={t['max_dd']:.1%} | Inds: {inds}")

        report += f"""

--- INDICATOR USAGE (last {len(window_100)} strategies) ---
OVERUSED (dominant, may cause stagnation — try alternatives):"""
        for ind, cnt in overused[:8]:
            report += f"\n  {ind}: {cnt} times"

        report += f"\nUNDEREXPLORED (rarely tried — high novelty potential):"
        for ind, cnt in underused[:10]:
            report += f"\n  {ind}: {cnt} times"

        report += f"\n\nCATEGORY USAGE (lowest = least explored):"
        for cat, cnt in sorted_cats[:5]:
            report += f"\n  {cat}: {cnt} total uses"

        report += f"\n\nINDICATOR DIVERSITY: {diversity_pct:.1%} (100%=perfect spread, low%=monoculture)"
        if untried_new:
            report += f"\nNEW POOL (Phase 3, zero usage — try these first): {', '.join(untried_new[:8])}"

        # ── 8. Recommendations ──
        recommendations = []
        if failure_patterns.get("MaxDD_too_deep", 0) > len(failures) * 0.3:
            recommendations.append(
                "PRIORITY: MaxDD failures dominate — strategies lack drawdown protection. "
                "Use ATR-based trailing stops or volatility-regime cash periods."
            )
        if failure_patterns.get("too_few_trades", 0) > len(failures) * 0.2:
            recommendations.append(
                "NOTE: Too-few-trades failures — entry conditions too strict. "
                "Loosen signal criteria or add alternative entry paths."
            )
        if failure_patterns.get("low_exposure", 0) > len(failures) * 0.2:
            recommendations.append(
                "NOTE: Low exposure failures — regime filter too conservative. "
                "Reduce strictness or widen the 'bull' state threshold."
            )
        if failure_patterns.get("duplicate_strategy", 0) > 0:
            dup_count = failure_patterns["duplicate_strategy"]
            recommendations.append(
                f"CRITICAL: {dup_count} strategies were EXACT DUPLICATES of existing ones (same Sharpe/CAGR/MaxDD). "
                "The LLM is generating functionally identical RVI code despite different descriptions. "
                "MANDATE: next strategies MUST NOT use RVI/RVI_Refined/RVI_State as primary entry signal. "
                "Force use of: HMM_Regime, QQE, STC, KDJ_J, CTI, GARCH_Vol, or CP_Distance instead."
            )
        if score_trend == "STAGNATING":
            recommendations.append(
                "ALERT: Composite score stagnating — current indicator combinations exhausted. "
                f"Pivot to unexplored categories: {sorted_cats[0][0]}, {sorted_cats[1][0] if len(sorted_cats) > 1 else ''}."
            )
        if consec_failures >= 5:
            recommendations.append(
                f"ALERT: {consec_failures} consecutive failures — recent approach is broken. "
                "Need radical strategy change, not incremental mutation."
            )

        # Push underexplored category recommendation
        if sorted_cats:
            least_cat = sorted_cats[0][0]
            least_inds = [ind for ind in self.CATEGORIES.get(least_cat, [])
                          if indicator_counts.get(ind, 0) <= 2]
            if least_inds:
                recommendations.append(
                    f"EXPLORE: '{least_cat}' category barely tried. "
                    f"Candidate indicators: {', '.join(least_inds[:4])}"
                )

        # New pool adoption recommendation (Phase 3C)
        if untried_new:
            recommendations.append(
                f"NEW POOL: {len(untried_new)} Phase-3 indicators have NEVER been used: "
                f"{', '.join(untried_new[:5])}. "
                "These provide genuinely new signal sources (ML regime, advanced momentum)."
            )
        if diversity_pct < 0.5:
            recommendations.append(
                f"MONOCULTURE ALERT: Indicator diversity only {diversity_pct:.1%} — "
                "system over-concentrates on a few indicators. Force usage of untried categories."
            )

        if recommendations:
            report += "\n\n--- RECOMMENDATIONS ---"
            for rec in recommendations:
                report += f"\n  * {rec}"

        report += "\n=== END ANALYST REPORT ==="
        return report

    def explain(self, stats_report: str, llm) -> str:
        """
        A-LLM: Given the statistical analysis, generate an LLM narrative
        explaining WHY the system is failing (root cause analysis).
        Uses task="director" for kimi-k2 / K1,K2 pool.
        """
        prompt = f"""You are A, a Problem Explorer for a quantitative trading system (TQQQ 3x leveraged ETF).

Here is a statistical analysis of strategy backtests:
{stats_report}

Provide a ROOT CAUSE analysis in 5-6 sentences:
1. PRIMARY failure reason (not just "MaxDD too deep" — explain WHY strategies have high drawdown structurally)
2. SECONDARY failure mode
3. Why the system is STUCK (what pattern keeps repeating despite variations)
4. What SPECIFIC structural change is needed (indicator logic, not just "try new indicators")
5. One overlooked opportunity visible in the data

Be specific and analytical. Mention exact indicator names. No generic advice."""
        result = llm.generate(prompt, task="director")
        return result or "Unable to generate narrative analysis."


# ═══════════════════════════════════════════════════════════════
# SolutionResearcher (B) — Proposes concrete strategy approaches
# ═══════════════════════════════════════════════════════════════
class SolutionResearcher:
    """
    B: Solution Researcher — proposes 3 concrete strategy approaches
    that directly address A's identified root causes.
    Uses task="director" for kimi-k2 / K1,K2 pool.
    """

    def research(self, a_narrative: str, stats_report: str,
                 known_indicators: list, llm) -> dict:
        """Propose 3 concrete strategy approaches addressing A's identified problems."""
        indicator_sample = ", ".join(known_indicators[:40])
        prompt = f"""You are B, a Solution Researcher for quantitative trading strategy development.

⚠️ LONG-ONLY CONSTRAINT: All strategies use TQQQ signals 0 (cash) or 1 (long) ONLY.
NEVER propose short entries. "Protective" = go to cash (signal=0), not short (signal=-1).

PROBLEM ANALYSIS from A:
{a_narrative}

KEY STATISTICS:
{stats_report[:800]}

AVAILABLE INDICATORS (exact names only): {indicator_sample}...

Propose exactly 3 concrete strategy approaches that DIRECTLY address the identified problems.
For each, focus on novel combinations NOT relying on RVI (overused).

Output ONLY valid JSON:
{{
  "proposals": [
    {{
      "name": "2-4 word strategy name",
      "core_logic": "1 sentence describing entry + exit + regime filter",
      "entry_indicator": "exact_column_name",
      "exit_indicator": "exact_column_name",
      "regime_indicator": "exact_column_name",
      "rationale": "why this addresses A's identified problem"
    }}
  ]
}}"""
        raw = llm.generate(prompt, task="director")
        return self._parse_json(raw, known_indicators)

    def _parse_json(self, raw: Optional[str], known_indicators: list) -> dict:
        """Parse JSON response and validate structure."""
        data = _parse_llm_json(raw)
        if data:
            proposals = data.get("proposals", [])
            if isinstance(proposals, list) and len(proposals) > 0:
                return {"proposals": proposals}
        return self._default_proposals()

    def _default_proposals(self) -> dict:
        """Fallback proposals using underused indicators."""
        return {
            "proposals": [
                {
                    "name": "TSI Elder Combo",
                    "core_logic": "TSI zero-cross for entry with Elder_Bull confirmation, exit on Elder_Bear dominance with ATR trailing stop",
                    "entry_indicator": "TSI",
                    "exit_indicator": "Elder_Bear",
                    "regime_indicator": "Elder_Bull",
                    "rationale": "Underused advanced momentum indicators break RVI over-dependence",
                },
                {
                    "name": "Aroon Structure Filter",
                    "core_logic": "Aroon_Osc trending signal with ZScore mean reversion entries, ADX regime filter",
                    "entry_indicator": "Aroon_Osc",
                    "exit_indicator": "ZScore",
                    "regime_indicator": "ADX",
                    "rationale": "Structure + trend quality combination rarely explored",
                },
                {
                    "name": "VoV Volatility Regime",
                    "core_logic": "VoV detects unstable regimes (cash), AO oscillator for entry in stable markets, ATR-based exit",
                    "entry_indicator": "AO",
                    "exit_indicator": "ATR",
                    "regime_indicator": "VoV",
                    "rationale": "Volatility-of-volatility filter directly addresses drawdown issues",
                },
            ]
        }


# ═══════════════════════════════════════════════════════════════
# AnalysisValidator (C1) — Validates B's proposals
# ═══════════════════════════════════════════════════════════════
class AnalysisValidator:
    """
    C1: Checks B's proposals for valid indicator names against the known pool.
    Pure Python first — LLM correction only if indicators are invalid (max 1 round).
    """

    def validate(self, b_proposals: dict, known_indicators: list, llm) -> dict:
        """
        Validate proposal indicator names against known pool.
        Returns validated result dict. Only calls LLM if corrections needed.
        """
        proposals = b_proposals.get("proposals", [])
        invalid = []

        for p in proposals:
            p_name = p.get("name", "?")
            for field in ["entry_indicator", "exit_indicator", "regime_indicator"]:
                val = p.get(field, "")
                if val and val not in known_indicators:
                    invalid.append(f"{p_name}.{field}: '{val}' not in pool")

        if not invalid:
            # All valid — pure Python pass, no LLM call
            return {
                "approved": True,
                "validated_proposals": proposals,
                "corrections": [],
            }

        # LLM correction round (max 1)
        print(f"   ⚠️ [C1] {len(invalid)} invalid indicators — requesting correction...")
        correction_prompt = f"""Fix these indicator names to valid column names from the pool:

Invalid indicators:
{chr(10).join(invalid[:10])}

Valid column names: {', '.join(known_indicators)}

Return corrected proposals as JSON only:
{{
  "proposals": [
    {{
      "name": "...",
      "core_logic": "...",
      "entry_indicator": "exact_valid_column_name",
      "exit_indicator": "exact_valid_column_name",
      "regime_indicator": "exact_valid_column_name",
      "rationale": "..."
    }}
  ]
}}"""
        corrected_raw = llm.generate(correction_prompt, task="secretary")

        if corrected_raw:
            data = _parse_llm_json(corrected_raw)
            if data:
                corrected = data.get("proposals", [])
                if corrected:
                    return {
                        "approved": True,
                        "validated_proposals": corrected,
                        "corrections": invalid,
                    }

        # Correction failed — return originals with approved=False
        return {
            "approved": False,
            "validated_proposals": proposals,
            "corrections": invalid,
        }


# ═══════════════════════════════════════════════════════════════
# Secretary (C2) — LLM-powered Brief synthesizer + execution_queue
# ═══════════════════════════════════════════════════════════════
class Secretary:
    """
    C2: Synthesizes Director advice + InternalAnalyst report (+ B's proposals if available)
    into a structured Brief JSON with an execution_queue of 3 directed assignments.
    Uses kimi-k2 via Groq K1/K2 pool (task="secretary").
    """

    def create_brief(self, director_advice: Optional[str], analyst_report: str, llm,
                     researcher_proposals: Optional[list] = None,
                     validator_result: Optional[dict] = None) -> Dict:
        """
        Call LLM to synthesize inputs into a validated Brief JSON.

        When researcher_proposals are available (full slow path), output includes
        execution_queue of 3 specific directed assignments.

        Returns dict with keys:
          focus_theme: str
          required_indicators: list[str]
          avoid_patterns: list[str]
          exploration_target: str
          brief_text: str
          execution_queue: list[str]  — 3 directed assignments (empty if no proposals)
        """
        director_section = (
            f"DIRECTOR'S STRATEGIC GUIDANCE:\n{director_advice}"
            if director_advice
            else "DIRECTOR'S GUIDANCE: None available — rely on analyst insights."
        )

        # B's validated proposals section (optional — only in full slow path)
        proposals_section = ""
        if researcher_proposals:
            proposals_section = "\n\nSOLUTION RESEARCHER PROPOSALS (validated by C1):\n"
            for idx, p in enumerate(researcher_proposals[:3], 1):
                proposals_section += (
                    f"  {idx}. {p.get('name', '')}: {p.get('core_logic', '')}\n"
                    f"     Entry: {p.get('entry_indicator', '')} | "
                    f"Exit: {p.get('exit_indicator', '')} | "
                    f"Regime: {p.get('regime_indicator', '')}\n"
                    f"     Rationale: {p.get('rationale', '')}\n"
                )

        # execution_queue JSON schema (only when proposals available)
        has_proposals = bool(researcher_proposals)
        eq_schema = ""
        eq_rules = ""
        if has_proposals:
            eq_schema = """,
  "execution_queue": [
    "Assignment 1: [Complete sentence — which indicators for entry/exit/regime, key parameters, why]",
    "Assignment 2: ...",
    "Assignment 3: ..."
  ]"""
            eq_rules = (
                "\n- execution_queue: 3 specific strategy assignments derived from Solution Researcher "
                "proposals. Each is a complete actionable sentence specifying entry/exit/regime indicators "
                "and key logic patterns. CRITICAL: ALL assignments must be LONG-ONLY (signal=0 or 1). "
                "Never include short entries. Use 'exit to cash' not 'enter short'."
            )

        prompt = f"""You are the Secretary (C2) of a quantitative research team. Your job is to synthesize research inputs into a structured strategy brief.

⚠️ SYSTEM CONSTRAINT — LONG-ONLY: All strategies trade TQQQ with signals 0 (cash) or 1 (long) ONLY.
NEVER include short entries, short-biased logic, or short-swing hedges in any assignment.
Risk reduction = better exits (ATR stops, trailing stops) + regime filters (go to cash), NOT shorts.

{director_section}

{analyst_report}{proposals_section}

Based on these inputs, create a concise research brief.
Output ONLY valid JSON (no markdown, no explanation, no code blocks):
{{
  "focus_theme": "One clear sentence describing strategic focus (e.g., 'Combine volume-regime awareness with mean-reversion entries to reduce MaxDD while maintaining CAGR')",
  "required_indicators": ["IndicatorName1", "IndicatorName2"],
  "avoid_patterns": ["Pattern1 — reason", "Pattern2 — reason"],
  "exploration_target": "Specific 1-2 indicator combination to try next (e.g., 'TSI zero-cross with Elder_Bull confirmation')",
  "brief_text": "2-3 sentence actionable directive for strategy generation. Be specific about logic patterns, not general advice."{eq_schema}
}}

Rules:
- required_indicators: pick 2-3 from the UNDEREXPLORED list in the analyst report — exact column names only
- avoid_patterns: match the top 2-3 failure patterns from analyst report
- exploration_target: name specific indicators from UNDEREXPLORED or LEAST-USED CATEGORY
- brief_text: must reference specific indicator names and logic patterns (state transitions, ATR stops, etc.)
- All indicator names must be exact column names (e.g., 'Vol_Ratio', not 'volume ratio'; 'HV_10', not 'historical volatility'){eq_rules}"""

        raw = llm.generate(prompt, task="secretary")
        if not raw:
            print("   ⚠️ Secretary LLM call failed, using default brief")
            return self._default_brief(analyst_report)

        # Parse JSON
        brief = _parse_llm_json(raw)
        if brief:
            required_keys = ["focus_theme", "required_indicators", "avoid_patterns",
                             "exploration_target", "brief_text"]
            if all(k in brief for k in required_keys):
                print(f"   📋 Secretary Brief: {brief['focus_theme'][:80]}")
                eq = brief.get("execution_queue", [])
                if eq:
                    print(f"   🎯 Execution queue: {len(eq)} assignments ready")
                return brief
            missing = [k for k in required_keys if k not in brief]
            print(f"   ⚠️ Secretary JSON missing keys: {missing}, using default brief")
        else:
            print("   ⚠️ Secretary JSON parse failed, using default brief")

        return self._default_brief(analyst_report)

    def _default_brief(self, analyst_report: str) -> Dict:
        """Fallback brief when LLM fails or returns invalid JSON."""
        # Extract underexplored indicators from analyst report
        unexplored_inds = []
        match = re.search(r'UNDEREXPLORED.*?(?=\n\n|CATEGORY|=== END)', analyst_report, re.DOTALL)
        if match:
            for line in match.group().split('\n'):
                line = line.strip()
                if ': ' in line and 'UNDEREXPLORED' not in line and line:
                    ind = line.split(':')[0].strip()
                    if ind and ind in InternalAnalyst.KNOWN_INDICATORS:
                        unexplored_inds.append(ind)

        # Extract top failure from analyst report
        avoid = []
        if "MaxDD_too_deep" in analyst_report:
            avoid.append("High drawdown without ATR-based stop — use adaptive exits")
        if "too_few_trades" in analyst_report:
            avoid.append("Overly selective entry — reduce condition count")

        return {
            "focus_theme": "Explore underused indicators to break out of current local maximum",
            "required_indicators": unexplored_inds[:3] if unexplored_inds else ["TSI", "Elder_Bull", "VoV"],
            "avoid_patterns": avoid if avoid else ["MaxDD too deep without stops", "Too many parameters"],
            "exploration_target": (
                f"Combine {unexplored_inds[0]} with ATR-based trailing stop"
                if unexplored_inds else "Try TSI zero-cross with Elder_Bull power confirmation"
            ),
            "brief_text": (
                "Focus on unexplored indicators and novel combinations to break stagnation. "
                "Prioritize drawdown control with adaptive ATR stops. "
                "Use state machine transitions, not raw threshold comparisons."
            ),
            "execution_queue": [],  # Empty queue for default brief (no B proposals)
        }

    @staticmethod
    def format_for_prompt(brief: Dict) -> str:
        """Format brief dict into a compact prompt injection string."""
        if not brief:
            return ""
        required = ", ".join(brief.get("required_indicators", []))
        avoid = "; ".join(brief.get("avoid_patterns", []))
        return f"""📋 RESEARCH BRIEF (Secretary Synthesis):
Theme: {brief.get('focus_theme', '')}
Priority indicators: {required}
Avoid: {avoid}
Explore: {brief.get('exploration_target', '')}
---
{brief.get('brief_text', '')}"""


# ═══════════════════════════════════════════════════════════════
# ExternalResearcher (F) — Proposes new indicators (inline helpers)
# ═══════════════════════════════════════════════════════════════
class ExternalResearcher:
    """
    F: External Researcher — proposes NEW technical indicators not in the 78-indicator pool.
    Generates inline helper methods for strategy classes (no indicator_pool.py changes).
    Uses task="director" for kimi-k2 / K1,K2 pool.
    """

    # Hardcoded fallback proposals — validated, reliable Python code
    # These are distinct from both the 78-indicator pool AND the Phase 3B additions (QQE/STC/KDJ etc.)
    FALLBACK_PROPOSALS = [
        {
            "name": "KAMA",
            "full_name": "Kaufman Adaptive Moving Average",
            "description": "Adapts MA speed to market noise — fast in trends, slow in chop",
            "method_name": "_calc_kama",
            "method_code": """def _calc_kama(self, prices, fast=2, slow=30):
    fast_sc = 2.0 / (fast + 1)
    slow_sc = 2.0 / (slow + 1)
    arr = prices.ffill().bfill().values.astype(float)
    kama = arr.copy()
    for i in range(10, len(arr)):
        direction = abs(arr[i] - arr[i - 10])
        volatility = float(np.sum(np.abs(np.diff(arr[i - 10:i + 1]))))
        er = direction / volatility if volatility > 0 else 0
        sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
        kama[i] = kama[i - 1] + sc * (arr[i] - kama[i - 1])
    return pd.Series(kama, index=prices.index)""",
            "usage": "kama = self._calc_kama(self.data['Close'], fast=2, slow=30)",
            "suitable_for": "regime",
            "entry_hint": "Bullish regime: Close > KAMA and KAMA slope positive (kama.diff() > 0)",
        },
        {
            "name": "Hurst",
            "full_name": "Rolling Hurst Exponent",
            "description": "Fractal market dimension: >0.5=trending, 0.5=random walk, <0.5=mean-reverting",
            "method_name": "_calc_hurst",
            "method_code": """def _calc_hurst(self, prices, window=60):
    def _h(w):
        log_w = np.log(np.abs(w) + 1e-10)
        lags = [l for l in [2, 4, 8, 16] if 2 * l < len(w)]
        if not lags:
            return 0.5
        vs = [np.var(log_w[l:] - log_w[:-l]) for l in lags]
        ll = np.log(lags)
        lv = np.log([max(v, 1e-10) for v in vs])
        return float(np.clip(np.polyfit(ll, lv, 1)[0] / 2, 0.0, 1.0))
    return prices.rolling(window).apply(_h, raw=True).fillna(0.5)""",
            "usage": "hurst = self._calc_hurst(self.data['Close'], window=60)",
            "suitable_for": "regime",
            "entry_hint": "Enter long only when Hurst > 0.55 (trending regime). Avoid mean-reversion noise when Hurst < 0.45.",
        },
        {
            "name": "ZLEMA",
            "full_name": "Zero-Lag Exponential Moving Average",
            "description": "EMA with lag compensation: faster signal with minimal delay vs standard EMA",
            "method_name": "_calc_zlema",
            "method_code": """def _calc_zlema(self, prices, period=20):
    lag = (period - 1) // 2
    adjusted = prices + (prices - prices.shift(lag))
    return adjusted.ewm(span=period, adjust=False).mean().bfill().fillna(0)""",
            "usage": "zlema = self._calc_zlema(self.data['Close'], period=20)",
            "suitable_for": "entry",
            "entry_hint": "Buy when Close crosses above ZLEMA(20). Faster than EMA crossover with less whipsaw.",
        },
    ]

    def propose(self, stats_report: str, known_indicators: list, llm) -> dict:
        """Propose 2-3 new indicators with inline Python calculation code."""
        # Extract overused section from analyst report for context
        overused_section = ""
        match = re.search(r'OVERUSED.*?(?=\nUNDEREXPLORED|\n\n)', stats_report, re.DOTALL)
        if match:
            overused_section = match.group()[:300]

        prompt = f"""You are F, an External Researcher specializing in technical analysis.

CURRENT INDICATOR POOL ({len(known_indicators)} indicators — DO NOT repropose these):
{', '.join(known_indicators)}

OVERUSED (especially avoid):
{overused_section}

TASK: Propose exactly 3 NEW technical indicators that are:
1. NOT already in the pool above
2. Suitable for TQQQ (3x leveraged ETF momentum)
3. Implementable with pandas/numpy only (no TA-Lib, no yfinance, no imports)
4. Each is a self-contained Python method

Good candidates: TEMA, CMO, Zero-Lag EMA, Fisher Transform, Laguerre RSI,
McGinley Dynamic, KST, PVT, VIDYA, Hurst Exponent, Kalman RSI, Fractal Adaptive MA,
Ehlers Instantaneous Trendline, Recursive Bands, Stochastic RSI smoothed, etc.
(QQE, STC, KDJ, SMI, Squeeze_Pro, KAMA are already implemented — do NOT propose these)

Output ONLY valid JSON:
{{
  "new_indicators": [
    {{
      "name": "SHORT_NAME",
      "full_name": "Full Indicator Name",
      "description": "1 sentence: what it measures and why useful for TQQQ",
      "method_name": "_calc_shortname",
      "method_code": "def _calc_shortname(self, prices, period=14):\\n    ...\\n    return result_series",
      "usage": "varname = self._calc_shortname(self.data['Close'], period=14)",
      "suitable_for": "regime|entry|exit",
      "entry_hint": "Specific signal logic: e.g. 'Buy when X crosses above Y'"
    }}
  ]
}}

CRITICAL requirements for method_code:
- Valid Python method (def _calc_xxx(self, prices, ...))
- prices is a pd.Series; returns pd.Series of same length
- Only use pd and np (already imported in strategy file)
- Include .fillna(0) or .bfill() on output
- NO external imports inside method body"""

        raw = llm.generate(prompt, task="director")
        return self._parse_and_validate(raw, known_indicators)

    def _parse_and_validate(self, raw: Optional[str], known_indicators: list) -> dict:
        """Parse JSON, exec-test each method, supplement with fallbacks if needed."""
        data = _parse_llm_json(raw)
        proposals = data.get("new_indicators", []) if data else []

        # Filter out overlaps with known pool + exec-test each
        validated = []
        known_lower = {k.lower() for k in known_indicators}
        for p in proposals:
            name = p.get("name", "")
            if name.lower() in known_lower:
                print(f"   ⚠️ [F] {name} overlaps with existing pool, skipping")
                continue
            code = p.get("method_code", "")
            mname = p.get("method_name", "")
            if code and mname and self._test_method(code, mname):
                validated.append(p)
                print(f"   ✅ [F] {name}: code validated")
            else:
                print(f"   ⚠️ [F] {name}: code failed validation, skipping")

        # Supplement with fallbacks if < 2 validated
        if len(validated) < 2:
            print(f"   🔄 [F] {len(validated)} valid, supplementing with fallbacks...")
            existing = {p["name"] for p in validated}
            for fb in self.FALLBACK_PROPOSALS:
                if fb["name"] not in existing and len(validated) < 3:
                    validated.append(fb)

        return {"new_indicators": validated[:3]}

    def _test_method(self, method_code: str, method_name: str) -> bool:
        """Exec-test the method with 300 dummy price points. Returns True if valid."""
        indented = "\n".join("    " + line for line in method_code.split("\n"))
        test_src = f"""
import pandas as pd
import numpy as np

class _T:
{indented}

_t = _T()
_prices = pd.Series([float(100 + i * 0.1 + (i % 7) * 0.5) for i in range(300)])
_result = _t.{method_name}(_prices)
assert isinstance(_result, pd.Series)
assert len(_result) == len(_prices)
"""
        try:
            exec(test_src, {})
            return True
        except Exception:
            return False


# ═══════════════════════════════════════════════════════════════
# ResultChecker (E) — Monitors iteration results
# ═══════════════════════════════════════════════════════════════
class ResultChecker:
    """
    E: Result Checker — monitors each iteration result and signals when the
    slow path (A→B→C1→C2) should be triggered.

    Pure Python, no LLM calls. Detects:
      - 4+ consecutive failures
      - Same error type 3x in last 5 results
      - OOS composite degrading (3+ successful iters with worsening test_composite)
    """

    def __init__(self):
        self._recent: list = []   # Sliding window of last 10 results

    def check(self, result: dict) -> dict:
        """
        Check result for patterns signaling need for slow path.
        Returns dict with: flags (list), trigger_slow_path (bool), summary (str).
        """
        self._recent.append(result)
        if len(self._recent) > 10:
            self._recent.pop(0)

        flags = []

        # Flag 1: 4+ consecutive failures
        consec = 0
        for r in reversed(self._recent):
            if not r.get("success"):
                consec += 1
            else:
                break
        if consec >= 4:
            flags.append(f"consec_failures={consec}")

        # Flag 2: same error type 3x in last 5 results
        recent5 = self._recent[-5:]
        error_types = [self._classify_error(r) for r in recent5 if not r.get("success")]
        if error_types:
            most_common, count = Counter(error_types).most_common(1)[0]
            if count >= 3:
                flags.append(f"repeated_error={most_common}(x{count})")

        # Flag 3: OOS composite degrading across last 3 successful iterations
        # Use > 0 (not truthiness) since test_composite defaults to 0.0 (falsy)
        successful = [r for r in self._recent
                      if r.get("success") and r.get("test_composite", 0) > 0]
        if len(successful) >= 3:
            composites = [r["test_composite"] for r in successful[-3:]]
            if composites[2] < composites[0] * 0.7:
                flags.append("oos_degrading")

        trigger = len(flags) > 0
        return {
            "flags": flags,
            "trigger_slow_path": trigger,
            "summary": "; ".join(flags) if flags else "ok",
        }

    def _classify_error(self, result: dict) -> str:
        """Classify the type of failure for pattern detection."""
        err = result.get("error", "") or ""
        if "MaxDD" in err:
            return "maxdd"
        if "Sharpe" in err:
            return "sharpe"
        if "trade" in err.lower():
            return "trades"
        if "Exposure" in err:
            return "exposure"
        if "KeyError" in err:
            return "keyerror"
        return "other"
