"""
Multi-Agent Brief System — Phase 1
====================================
InternalAnalyst: Pure Python statistical analysis of history_of_thoughts.json
Secretary: LLM-powered synthesizer → generates structured Brief JSON for next iterations

Pipeline: Director advice → InternalAnalyst report → Secretary Brief → idea prompts
"""

import json
import re
from typing import Dict, List, Optional
from datetime import datetime


# ═══════════════════════════════════════════════════════════════
# InternalAnalyst — Pure Python, no LLM needed
# ═══════════════════════════════════════════════════════════════
class InternalAnalyst:
    """
    Analyzes history_of_thoughts.json to extract statistical insights.
    No LLM required — pure computation for fast, deterministic analysis.
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
    }

    def analyze(self, history: Dict) -> str:
        """
        Perform statistical analysis of strategy history.
        Returns a plain-text report string for the Secretary.
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
            avg_early = sum(s.get("composite", 0) for s in first_half) / len(first_half)
            avg_late = sum(s.get("composite", 0) for s in second_half) / len(second_half)
            score_trend = "IMPROVING" if avg_late > avg_early * 1.02 else "STAGNATING"
            trend_detail = f" (early_avg={avg_early:.4f} → recent_avg={avg_late:.4f})"
        else:
            score_trend = "INSUFFICIENT_DATA"
            trend_detail = ""

        # ── 5. Top strategies and their dominant indicators ──
        rankable = [s for s in strategies if s.get("success") and s.get("composite", 0) > 0]
        top5 = sorted(rankable, key=lambda x: x.get("composite", 0), reverse=True)[:5]

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
        for cat, cnt in sorted_cats[:4]:
            report += f"\n  {cat}: {cnt} total uses"

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

        if recommendations:
            report += "\n\n--- RECOMMENDATIONS ---"
            for rec in recommendations:
                report += f"\n  * {rec}"

        report += "\n=== END ANALYST REPORT ==="
        return report


# ═══════════════════════════════════════════════════════════════
# Secretary — LLM-powered Brief synthesizer
# ═══════════════════════════════════════════════════════════════
class Secretary:
    """
    Synthesizes Director advice + InternalAnalyst report into a structured Brief JSON.
    Uses kimi-k2 via Groq K1/K2 pool (task="secretary").
    """

    def create_brief(self, director_advice: Optional[str], analyst_report: str, llm) -> Dict:
        """
        Call LLM to synthesize inputs into a validated Brief JSON.

        Returns dict with keys:
          focus_theme: str — one-sentence strategic direction for next batch
          required_indicators: list[str] — indicators to prioritize (from underexplored)
          avoid_patterns: list[str] — failure patterns to avoid
          exploration_target: str — specific novel combination to try
          brief_text: str — compact prompt injection for idea generator
        """
        director_section = (
            f"DIRECTOR'S STRATEGIC GUIDANCE:\n{director_advice}"
            if director_advice
            else "DIRECTOR'S GUIDANCE: None available — rely on analyst insights."
        )

        prompt = f"""You are the Secretary of a quantitative research team. Your job is to synthesize research inputs into a structured strategy brief for the next batch of automated strategy iterations.

{director_section}

{analyst_report}

Based on these inputs, create a concise research brief.
Output ONLY valid JSON (no markdown, no explanation, no code blocks):
{{
  "focus_theme": "One clear sentence describing strategic focus (e.g., 'Combine volume-regime awareness with mean-reversion entries to reduce MaxDD while maintaining CAGR')",
  "required_indicators": ["IndicatorName1", "IndicatorName2"],
  "avoid_patterns": ["Pattern1 — reason", "Pattern2 — reason"],
  "exploration_target": "Specific 1-2 indicator combination to try next (e.g., 'TSI zero-cross with Elder_Bull confirmation')",
  "brief_text": "2-3 sentence actionable directive for strategy generation. Be specific about logic patterns, not general advice."
}}

Rules:
- required_indicators: pick 2-3 from the UNDEREXPLORED list in the analyst report — exact column names only
- avoid_patterns: match the top 2-3 failure patterns from analyst report
- exploration_target: name specific indicators from UNDEREXPLORED or LEAST-USED CATEGORY
- brief_text: must reference specific indicator names and logic patterns (state transitions, ATR stops, etc.)
- All indicator names must be exact column names (e.g., 'Vol_Ratio', not 'volume ratio'; 'HV_10', not 'historical volatility')"""

        raw = llm.generate(prompt, task="secretary")
        if not raw:
            print("   ⚠️ Secretary LLM call failed, using default brief")
            return self._default_brief(analyst_report)

        # Parse JSON
        try:
            cleaned = raw.strip()
            if cleaned.startswith("```"):
                cleaned = re.sub(r'^```(?:json)?\s*', '', cleaned, flags=re.MULTILINE)
                cleaned = re.sub(r'\s*```\s*$', '', cleaned)
            # Find JSON object
            match = re.search(r'\{[\s\S]+\}', cleaned)
            if match:
                cleaned = match.group()
            brief = json.loads(cleaned)
            required_keys = ["focus_theme", "required_indicators", "avoid_patterns",
                             "exploration_target", "brief_text"]
            if all(k in brief for k in required_keys):
                print(f"   📋 Secretary Brief: {brief['focus_theme'][:80]}")
                return brief
            else:
                missing = [k for k in required_keys if k not in brief]
                print(f"   ⚠️ Secretary JSON missing keys: {missing}, using default brief")
        except (json.JSONDecodeError, KeyError) as e:
            print(f"   ⚠️ Secretary JSON parse failed ({e}), using default brief")

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
