"""
Retroactive Parameter Sweep
============================
Sweeps existing generated strategies that have NO get_params().
Uses regex substitution to vary numeric constants (thresholds, spans, multipliers)
found in the strategy source code, then backtests all variants.

Usage:
    PYTHONIOENCODING=utf-8 PYTHONUTF8=1 py -3.14 retroactive_param_sweep.py
    PYTHONIOENCODING=utf-8 PYTHONUTF8=1 py -3.14 retroactive_param_sweep.py --top 20
    PYTHONIOENCODING=utf-8 PYTHONUTF8=1 py -3.14 retroactive_param_sweep.py --name Strategy_Gen983
"""

import re
import ast
import json
import argparse
import itertools
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

from indicator_pool import get_enriched_data
from backtest import BacktestEngine
from main_loop import calculate_composite_score, hard_filter, is_duplicate_result

GENERATED_DIR  = Path("generated_strategies")
HISTORY_FILE   = Path("history_of_thoughts.json")
RETRO_LOG_FILE = Path("retroactive_sweep_log.json")


# ─────────────────────────────────────────────────────────────
# Numeric constant extraction
# ─────────────────────────────────────────────────────────────

def extract_sweep_params(code: str) -> Dict[str, List]:
    """
    Scan strategy source code for numeric constants that are worth sweeping.
    Returns a dict of {param_label: [candidate_values]}.

    Targets:
      - Comparison thresholds: `> 59`, `< 30`, `>= 0.8`
      - Rolling windows:       `rolling(20)`, `rolling(14)`
      - EWM spans:             `span=20`, `span=34`
      - Multipliers:           `* 1.5`, `* 2.0`  (in ATR/position-sizing context)
    """
    params = {}

    # 1. Comparison thresholds (integer or float, range 1..200)
    thresh_pattern = re.compile(r'([><=!]+)\s*([\d]+(?:\.\d+)?)')
    thresholds = set()
    for match in thresh_pattern.finditer(code):
        op, val_str = match.group(1), match.group(2)
        val = float(val_str)
        if 1 < val < 200 and op in ('>', '<', '>=', '<='):
            thresholds.add(val)

    for val in sorted(thresholds):
        label = f"thresh_{int(val)}" if val == int(val) else f"thresh_{val}"
        candidates = _make_int_range(val, pct=0.20, n_steps=4) if val == int(val) else \
                     _make_float_range(val, pct=0.20, n_steps=4)
        if len(candidates) > 1:
            params[label] = candidates

    # 2. Rolling windows: rolling(N)
    roll_pattern = re.compile(r'rolling\((\d+)\)')
    for match in roll_pattern.finditer(code):
        val = int(match.group(1))
        if 3 <= val <= 200:
            label = f"window_{val}"
            if label not in params:
                params[label] = _make_int_range(val, pct=0.30, n_steps=3)

    # 3. EWM spans: span=N, span=N,
    span_pattern = re.compile(r'span\s*=\s*(\d+)')
    for match in span_pattern.finditer(code):
        val = int(match.group(1))
        if 3 <= val <= 200:
            label = f"span_{val}"
            if label not in params:
                params[label] = _make_int_range(val, pct=0.30, n_steps=3)

    # 4. ATR multipliers: * 1.5, * 2.0 etc. (float between 0.5 and 5.0)
    mult_pattern = re.compile(r'\*\s*([\d]\.\d+)')
    for match in mult_pattern.finditer(code):
        val = float(match.group(1))
        if 0.5 <= val <= 5.0:
            label = f"mult_{val}"
            if label not in params:
                params[label] = _make_float_range(val, pct=0.30, n_steps=3)

    # Limit to 4 most impactful params to keep combos manageable
    # Prioritise: thresholds > multipliers > windows > spans
    priority_order = (
        [k for k in params if k.startswith('thresh_')] +
        [k for k in params if k.startswith('mult_')] +
        [k for k in params if k.startswith('window_')] +
        [k for k in params if k.startswith('span_')]
    )
    seen = set()
    ordered = [k for k in priority_order if not (k in seen or seen.add(k))]
    params = {k: params[k] for k in ordered[:4]}

    return params


def _make_int_range(base: float, pct: float, n_steps: int) -> List[int]:
    base = int(round(base))
    step = max(1, int(base * pct))
    candidates = sorted({max(1, base + i * step) for i in range(-n_steps // 2, n_steps // 2 + 1)})
    return candidates


def _make_float_range(base: float, pct: float, n_steps: int) -> List[float]:
    step = max(0.05, round(base * pct, 2))
    candidates = sorted({round(base + i * step, 2) for i in range(-n_steps // 2, n_steps // 2 + 1)
                         if round(base + i * step, 2) > 0})
    return candidates


# ─────────────────────────────────────────────────────────────
# Code substitution + dynamic class loading
# ─────────────────────────────────────────────────────────────

def make_variant_class(original_code: str, class_name: str,
                       substitutions: Dict[str, float]):
    """
    Create a strategy class variant by substituting numeric literals.
    substitutions: {label: new_value} where label encodes the original value.
    Returns the class object, or None if exec fails.
    """
    modified = original_code
    for label, new_val in substitutions.items():
        # Extract original value from label (e.g. 'thresh_59' → 59)
        parts = label.split('_')
        try:
            orig_val = float(parts[-1]) if '.' in parts[-1] else int(parts[-1])
        except ValueError:
            continue

        # Replace the numeric literal precisely using word boundaries
        # For integers: replace `59` but not `590` or `159`
        orig_str = str(orig_val) if orig_val == int(orig_val) else str(orig_val)
        new_str  = str(int(new_val)) if new_val == int(new_val) else str(new_val)
        pattern = r'(?<!\d)' + re.escape(orig_str) + r'(?!\d)'
        modified = re.sub(pattern, new_str, modified)

    try:
        namespace = {}
        exec(modified, namespace)           # nosec — controlled internal use
        cls = namespace.get(class_name)
        if cls is None:
            return None
        return cls
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────
# Main sweep logic
# ─────────────────────────────────────────────────────────────

def sweep_strategy_retroactively(
    strategy_name: str,
    code: str,
    data: pd.DataFrame,
    history: dict,
    max_combos: int = 200,
) -> Optional[Dict]:
    """
    Sweep a single strategy by varying its numeric constants.
    Returns best result dict if improved, else None.
    """
    class_name = strategy_name.replace('Strategy_', 'Strategy_')
    sweep_params = extract_sweep_params(code)

    if not sweep_params:
        print(f"   ⚠️ No sweepable params found in {strategy_name}")
        return None

    print(f"   🔍 Sweep params: { {k: len(v) for k,v in sweep_params.items()} }")

    # Build grid
    labels  = list(sweep_params.keys())
    grid    = list(itertools.product(*[sweep_params[k] for k in labels]))
    if len(grid) > max_combos:
        import random
        grid = random.sample(grid, max_combos)
    print(f"   🔢 Testing {len(grid)} combinations...")

    engine  = BacktestEngine(data)
    results = []

    for combo in grid:
        subs = dict(zip(labels, combo))
        cls  = make_variant_class(code, class_name, subs)
        if cls is None:
            continue
        try:
            inst = cls()
            inst.init(data)
            signals = inst.generate_signals()
            if signals.isna().all() or (signals == 0).all():
                continue
            bt = engine.run(inst)
            if hard_filter(bt):
                continue
            results.append({
                "params": subs,
                "sharpe": bt.sharpe_ratio,
                "cagr": bt.cagr,
                "max_dd": bt.max_drawdown,
                "calmar": bt.calmar_ratio,
                "composite": calculate_composite_score(
                    bt.sharpe_ratio, bt.calmar_ratio, bt.sortino_ratio,
                    bt.max_drawdown, getattr(bt, 'profit_factor', 1.0)
                ),
            })
        except Exception:
            continue

    if not results:
        print(f"   ❌ No valid variants found")
        return None

    results.sort(key=lambda r: r['composite'], reverse=True)
    best = results[0]
    print(f"   🏆 Best: Sharpe={best['sharpe']:.2f} CAGR={best['cagr']:.1%} "
          f"MaxDD={best['max_dd']:.1%} Comp={best['composite']:.4f}")
    print(f"          Params: {best['params']}")
    return best


# ─────────────────────────────────────────────────────────────
# History update
# ─────────────────────────────────────────────────────────────

def update_history_with_swept_result(history: dict, strategy_name: str, best: Dict):
    """Overwrite the strategy's metrics in history with the swept best result."""
    for s in history.get("strategies", []):
        if s["name"] == strategy_name:
            old_comp = s.get("composite", 0)
            s["sharpe"]    = best["sharpe"]
            s["cagr"]      = best["cagr"]
            s["max_dd"]    = best["max_dd"]
            s["calmar"]    = best["calmar"]
            s["composite"] = best["composite"]
            s["swept_params"] = best["params"]
            s["swept_at"]  = datetime.now().isoformat()
            print(f"   ✅ History updated: Comp {old_comp:.4f} → {best['composite']:.4f}")
            return
    print(f"   ⚠️ {strategy_name} not found in history")


# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Retroactive param sweep for existing strategies")
    parser.add_argument('--top',  type=int, default=15, help='Sweep top N strategies by composite score')
    parser.add_argument('--name', type=str, default=None, help='Sweep a specific strategy by name')
    parser.add_argument('--min-composite', type=float, default=0.25, help='Min composite score to include')
    parser.add_argument('--max-combos', type=int, default=200, help='Max param combinations per strategy')
    args = parser.parse_args()

    print("=" * 60)
    print("🔬 Retroactive Parameter Sweep")
    print("=" * 60)

    # Load data and history
    data    = get_enriched_data()
    with open(HISTORY_FILE, encoding='utf-8') as f:
        history = json.load(f)

    strategies = history.get("strategies", [])
    rankable   = [s for s in strategies if s.get("success") and
                  s.get("composite", 0) >= args.min_composite]

    # Pick targets
    if args.name:
        targets = [s for s in rankable if s["name"] == args.name]
        if not targets:
            print(f"❌ Strategy '{args.name}' not found or below threshold")
            return
    else:
        targets = sorted(rankable, key=lambda x: x.get("composite", 0), reverse=True)[:args.top]

    print(f"Targets: {len(targets)} strategies | max_combos={args.max_combos}")
    print()

    log_entries = []
    improved    = 0

    for rank, s in enumerate(targets, 1):
        name = s["name"]
        num  = name.replace("Strategy_Gen", "")
        fpath = GENERATED_DIR / f"strategy_gen_{num}.py"

        print(f"[{rank}/{len(targets)}] {name}  "
              f"Sharpe={s.get('sharpe',0):.2f} Comp={s.get('composite',0):.4f}")

        if not fpath.exists():
            print(f"   ⚠️ .py file not found, skipping")
            continue

        code = fpath.read_text(encoding='utf-8', errors='ignore')

        # Skip already-swept strategies (params already optimal)
        if 'swept_params' in s:
            print(f"   ⏭ Already swept ({s['swept_params']}), skipping")
            continue

        best = sweep_strategy_retroactively(name, code, data, history,
                                            max_combos=args.max_combos)

        entry = {
            "name": name,
            "original_composite": s.get("composite", 0),
            "original_sharpe": s.get("sharpe", 0),
        }

        # Compare on Sharpe (in-sample) — composites are not comparable because
        # history composites use OOS-weighted scoring while sweep runs IS only.
        if best and best["sharpe"] > s.get("sharpe", 0) + 0.02:
            update_history_with_swept_result(history, name, best)
            entry.update({"status": "improved", "best": best})
            improved += 1
        else:
            entry["status"] = "no_improvement"
            print(f"   — No improvement over base Sharpe={s.get('sharpe',0):.2f}")

        log_entries.append(entry)
        print()

    # Save updated history
    with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2, default=str)
    print(f"💾 History saved.")

    # Save sweep log
    existing_log = []
    if RETRO_LOG_FILE.exists():
        with open(RETRO_LOG_FILE, encoding='utf-8') as f:
            existing_log = json.load(f)
    existing_log.append({
        "run_at": datetime.now().isoformat(),
        "targets": len(targets),
        "improved": improved,
        "entries": log_entries,
    })
    with open(RETRO_LOG_FILE, 'w', encoding='utf-8') as f:
        json.dump(existing_log, f, indent=2, default=str)

    print("=" * 60)
    print(f"✅ Done: {improved}/{len(targets)} strategies improved")
    print("=" * 60)


if __name__ == "__main__":
    main()
