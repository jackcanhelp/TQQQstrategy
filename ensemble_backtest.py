"""
Ensemble Backtest
=================
Combine multiple strategies via majority voting.
Enters TQQQ only when >= min_votes strategies signal long simultaneously.

Expected benefit:
  - Reduce MaxDD (consensus required → fewer false entries)
  - Improve Sharpe (filter noise)
  - Some CAGR loss acceptable (miss solo-signal trades)

Usage:
    # Auto-test all promising combinations from top strategies
    PYTHONIOENCODING=utf-8 PYTHONUTF8=1 py -3.14 ensemble_backtest.py

    # Test a specific set of strategies
    PYTHONIOENCODING=utf-8 PYTHONUTF8=1 py -3.14 ensemble_backtest.py \
        --strategies 1381 3671 4415 --min-votes 2

    # Exhaustive search over all top-N strategies
    PYTHONIOENCODING=utf-8 PYTHONUTF8=1 py -3.14 ensemble_backtest.py \
        --top 15 --combo-size 3 --min-votes 2
"""

import argparse
import itertools
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

from indicator_pool import get_enriched_data
from backtest import BacktestEngine

GENERATED_DIR = Path("generated_strategies")
HISTORY_FILE  = Path("history_of_thoughts.json")
ENSEMBLE_LOG  = Path("ensemble_results.json")


# ─────────────────────────────────────────────────────────────
# Strategy loader
# ─────────────────────────────────────────────────────────────

def load_strategy_signals(sid: str, data: pd.DataFrame) -> Optional[pd.Series]:
    """Load a strategy by ID, run it, return its signal Series (0/1)."""
    fpath = GENERATED_DIR / f"strategy_gen_{sid}.py"
    if not fpath.exists():
        return None
    code = fpath.read_text(encoding="utf-8", errors="ignore")
    m = re.search(r"class (Strategy_Gen\w+)", code)
    if not m:
        return None
    class_name = m.group(1)
    try:
        ns: dict = {}
        exec(code, ns)  # nosec
        cls = ns[class_name]
        if hasattr(cls, "__init__") and "swept_params" in str(cls.__init__.__code__.co_varnames):
            inst = cls()
        else:
            inst = cls()
        inst.init(data)
        signals = inst.generate_signals()
        return signals.clip(0, 1)
    except Exception as e:
        print(f"   ⚠️  Failed to load Gen{sid}: {e}")
        return None


def load_champion_signals(data: pd.DataFrame) -> pd.Series:
    """Load ChampionRVI signals as baseline."""
    from champion_rvi import ChampionRVI
    c = ChampionRVI(enable_short=False)
    c.init(data)
    return c.generate_signals().clip(0, 1)


# ─────────────────────────────────────────────────────────────
# Ensemble signal combiner
# ─────────────────────────────────────────────────────────────

def majority_vote(signals_list: List[pd.Series], min_votes: int) -> pd.Series:
    """
    Combine signals by majority voting.
    Returns 1 when >= min_votes strategies signal long, else 0.
    """
    combined = pd.concat(signals_list, axis=1).fillna(0)
    vote_count = combined.sum(axis=1)
    return (vote_count >= min_votes).astype(float)


def weighted_vote(signals_list: List[pd.Series],
                  weights: List[float],
                  threshold: float = 0.5) -> pd.Series:
    """
    Weighted voting: enter when weighted average signal >= threshold.
    """
    combined = pd.concat(signals_list, axis=1).fillna(0)
    w = np.array(weights) / sum(weights)
    weighted = combined.values @ w
    return pd.Series((weighted >= threshold).astype(float), index=combined.index)


# ─────────────────────────────────────────────────────────────
# Single ensemble backtest
# ─────────────────────────────────────────────────────────────

def run_ensemble(
    sids: List[str],
    data: pd.DataFrame,
    min_votes: int = 2,
    include_champion: bool = False,
    label: Optional[str] = None,
) -> Optional[Dict]:
    """
    Load strategies by SID, combine via majority vote, backtest.
    Returns result dict or None.
    """
    signals_list = []
    names_loaded = []

    if include_champion:
        sig = load_champion_signals(data)
        signals_list.append(sig)
        names_loaded.append("ChampionRVI")

    for sid in sids:
        sig = load_strategy_signals(sid, data)
        if sig is not None:
            signals_list.append(sig)
            names_loaded.append(f"Gen{sid}")

    if len(signals_list) < 2:
        print(f"   ⚠️  Only {len(signals_list)} strategies loaded, skipping")
        return None

    ensemble_sig = majority_vote(signals_list, min_votes=min_votes)
    exposure = ensemble_sig.mean()

    if exposure < 0.02:
        print(f"   ⚠️  Exposure too low ({exposure:.1%}), min_votes too strict")
        return None

    lbl = label or f"Ensemble({'+'.join(names_loaded)}, v>={min_votes})"
    engine = BacktestEngine(data)

    # Create a minimal strategy-like wrapper for the engine
    _name = lbl[:50]

    class _EnsembleWrapper:
        name = _name
        def generate_signals(self_inner):
            return ensemble_sig
        def init(self_inner, d):
            pass

    wrapper = _EnsembleWrapper()
    bt = engine.run(wrapper)
    return {
        "label":          lbl,
        "strategies":     names_loaded,
        "min_votes":      min_votes,
        "n_strategies":   len(signals_list),
        "sharpe":         bt.sharpe_ratio,
        "cagr":           bt.cagr,
        "max_dd":         bt.max_drawdown,
        "calmar":         bt.calmar_ratio,
        "sortino":        bt.sortino_ratio,
        "trades":         bt.total_trades,
        "time_in_market": bt.time_in_market,
        "exposure":       float(exposure),
    }


# ─────────────────────────────────────────────────────────────
# Individual baselines
# ─────────────────────────────────────────────────────────────

def run_individual_baselines(sids: List[str], data: pd.DataFrame) -> List[Dict]:
    """Backtest each strategy individually for comparison."""
    results = []
    engine = BacktestEngine(data)

    # ChampionRVI baseline
    from champion_rvi import ChampionRVI
    c = ChampionRVI(enable_short=False)
    c.init(data)
    bt = engine.run(c)
    results.append({
        "label": "ChampionRVI (baseline)",
        "sharpe": bt.sharpe_ratio, "cagr": bt.cagr,
        "max_dd": bt.max_drawdown, "calmar": bt.calmar_ratio,
        "time_in_market": bt.time_in_market,
    })

    for sid in sids:
        sig = load_strategy_signals(sid, data)
        if sig is None:
            continue

        class _W:
            name = f"Gen{sid}"
            def generate_signals(self_inner): return sig
            def init(self_inner, d): pass

        bt = engine.run(_W())
        results.append({
            "label": f"Gen{sid} (individual)",
            "sharpe": bt.sharpe_ratio, "cagr": bt.cagr,
            "max_dd": bt.max_drawdown, "calmar": bt.calmar_ratio,
            "time_in_market": bt.time_in_market,
        })
    return results


# ─────────────────────────────────────────────────────────────
# Auto-search: test all 3-strategy combinations from top N
# ─────────────────────────────────────────────────────────────

def search_best_ensemble(
    candidate_sids: List[str],
    data: pd.DataFrame,
    combo_size: int = 3,
    min_votes: int = 2,
    include_champion: bool = False,
    top_n: int = 10,
) -> List[Dict]:
    """
    Exhaustively test all combinations of combo_size strategies.
    Returns top_n results sorted by Sharpe.
    """
    combos = list(itertools.combinations(candidate_sids, combo_size))
    print(f"\n🔍 Testing {len(combos)} combinations "
          f"(size={combo_size}, min_votes={min_votes})...")

    all_results = []
    for i, combo in enumerate(combos, 1):
        r = run_ensemble(
            list(combo), data, min_votes=min_votes,
            include_champion=include_champion,
        )
        if r:
            all_results.append(r)
        if i % 20 == 0:
            print(f"   [{i}/{len(combos)}] tested, "
                  f"{len(all_results)} valid ensembles so far")

    all_results.sort(key=lambda x: x["sharpe"], reverse=True)
    return all_results[:top_n]


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Ensemble Backtest")
    parser.add_argument("--strategies", nargs="+", type=str, default=None,
                        help="Specific strategy IDs to combine (e.g. 1381 3671 4415)")
    parser.add_argument("--top", type=int, default=12,
                        help="Use top N strategies from history (default: 12)")
    parser.add_argument("--combo-size", type=int, default=3,
                        help="Number of strategies per ensemble (default: 3)")
    parser.add_argument("--min-votes", type=int, default=2,
                        help="Min votes required to enter (default: 2)")
    parser.add_argument("--include-champion", action="store_true",
                        help="Include ChampionRVI in every ensemble")
    parser.add_argument("--min-sharpe", type=float, default=1.0,
                        help="Min individual Sharpe to be a candidate (default: 1.0)")
    args = parser.parse_args()

    print("=" * 62)
    print("🎯 Ensemble Backtest")
    print("=" * 62)

    data = get_enriched_data()
    with open(HISTORY_FILE, encoding="utf-8") as f:
        history = json.load(f)

    # Build candidate list
    if args.strategies:
        candidate_sids = args.strategies
        print(f"Mode: specific strategies {candidate_sids}")
    else:
        successful = [s for s in history.get("strategies", [])
                      if s.get("success") and s.get("sharpe", 0) >= args.min_sharpe]
        successful.sort(key=lambda s: s.get("sharpe", 0), reverse=True)
        # Keep only those with .py files, deduplicate by name
        seen = set()
        candidate_sids = []
        for s in successful:
            name = s.get("name", "")
            if name in seen:
                continue
            seen.add(name)
            sid = name.replace("Strategy_Gen", "")
            if (GENERATED_DIR / f"strategy_gen_{sid}.py").exists():
                candidate_sids.append(sid)
            if len(candidate_sids) >= args.top:
                break
        print(f"Mode: top {len(candidate_sids)} strategies (Sharpe>={args.min_sharpe})")
        print(f"Candidates: {['Gen'+s for s in candidate_sids]}")

    # ── Individual baselines ──────────────────────────────────
    print("\n📊 Individual Baselines:")
    print("─" * 62)
    baselines = run_individual_baselines(candidate_sids, data)
    for b in baselines:
        print(f"  {b['label']:<40} "
              f"S={b['sharpe']:.2f}  CAGR={b['cagr']:.1%}  "
              f"MaxDD={b['max_dd']:.1%}  TiM={b['time_in_market']:.0%}")

    # ── Ensemble search ───────────────────────────────────────
    if args.strategies and len(args.strategies) >= 2:
        # Test the specific combination with all vote thresholds
        print(f"\n🗳️  Testing specific ensemble: {args.strategies}")
        print("─" * 62)
        ensemble_results = []
        n = len(args.strategies)
        for min_v in range(1, n + 1):
            r = run_ensemble(args.strategies, data, min_votes=min_v,
                             include_champion=args.include_champion)
            if r:
                ensemble_results.append(r)
    else:
        # Auto-search all combinations
        ensemble_results = search_best_ensemble(
            candidate_sids, data,
            combo_size=args.combo_size,
            min_votes=args.min_votes,
            include_champion=args.include_champion,
        )

    # ── Print results ─────────────────────────────────────────
    print(f"\n🏆 Top Ensemble Results (sorted by Sharpe):")
    print("─" * 62)
    for i, r in enumerate(ensemble_results[:15], 1):
        print(f"  #{i:2d}  {r['label']}")
        print(f"       S={r['sharpe']:.2f}  CAGR={r['cagr']:.1%}  "
              f"MaxDD={r['max_dd']:.1%}  Calmar={r['calmar']:.2f}  "
              f"TiM={r['time_in_market']:.0%}")

    # ── Save results ──────────────────────────────────────────
    existing = []
    if ENSEMBLE_LOG.exists():
        with open(ENSEMBLE_LOG, encoding="utf-8") as f:
            existing = json.load(f)
    from datetime import datetime
    existing.append({
        "run_at":    datetime.now().isoformat(),
        "baselines": baselines,
        "ensembles": ensemble_results,
        "args":      vars(args),
    })
    with open(ENSEMBLE_LOG, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2, default=str)
    print(f"\n💾 Results saved to {ENSEMBLE_LOG}")
    print("=" * 62)


if __name__ == "__main__":
    main()
