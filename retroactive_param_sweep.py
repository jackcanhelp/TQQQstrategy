"""
Retroactive Parameter Sweep
============================
Sweeps existing generated strategies — both old (regex) and new (get_params) formats.

Usage:
    # Sweep top 15 by Sharpe (default)
    PYTHONIOENCODING=utf-8 PYTHONUTF8=1 py -3.14 retroactive_param_sweep.py

    # Sweep ALL strategies with Sharpe >= 0.4 (40 combos each)
    PYTHONIOENCODING=utf-8 PYTHONUTF8=1 py -3.14 retroactive_param_sweep.py \
        --all --min-sharpe 0.4 --max-combos 40

    # Deep-sweep top 30 (200 combos, re-sweep already swept)
    PYTHONIOENCODING=utf-8 PYTHONUTF8=1 py -3.14 retroactive_param_sweep.py \
        --top 30 --min-sharpe 0.4 --max-combos 200 --force

    # Sweep a specific strategy by name
    PYTHONIOENCODING=utf-8 PYTHONUTF8=1 py -3.14 retroactive_param_sweep.py \
        --name Strategy_Gen2323 --max-combos 20
"""

import re
import ast
import json
import time
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


def save_history_merge(in_memory: dict):
    """
    Merge-safe save: reads disk first, applies our swept updates on top,
    then atomically writes back. Safe to run concurrently with main_loop.py.
    """
    # Build a map of in-memory changes (only strategies we actually updated)
    mem_updates = {s["name"]: s for s in in_memory.get("strategies", [])
                   if s.get("swept_at")}

    for attempt in range(5):
        try:
            tmp = HISTORY_FILE.with_suffix('.tmp')
            # Read fresh from disk to pick up any new strategies added by main_loop
            with open(HISTORY_FILE, encoding='utf-8') as f:
                disk = json.load(f)
            # Apply our swept updates onto the disk copy
            updated = []
            for s in disk.get("strategies", []):
                name = s.get("name", "")
                updated.append(mem_updates.get(name, s))
            disk["strategies"] = updated
            with open(tmp, 'w', encoding='utf-8') as f:
                json.dump(disk, f, indent=2, default=str)
            tmp.replace(HISTORY_FILE)
            return
        except OSError as e:
            if attempt < 4:
                time.sleep(1 + attempt)
            else:
                print(f"⚠️  save_history_merge failed: {e}")


# ─────────────────────────────────────────────────────────────
# Numeric constant extraction (regex path)
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
# Code substitution + dynamic class loading (regex path)
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
# Sweep — new-style strategies (get_params path)
# ─────────────────────────────────────────────────────────────

def _sweep_with_get_params(
    strategy_name: str,
    code: str,
    data: pd.DataFrame,
    max_combos: int,
) -> Optional[Dict]:
    """
    For strategies that have `get_params()`: exec the code, instantiate the
    class, call get_params() to get base params, then use run_sweep().
    Much more precise than regex substitution.
    """
    namespace = {}
    try:
        exec(code, namespace)               # nosec — controlled internal use
    except Exception as e:
        print(f"   ❌ exec failed: {e}")
        return None

    cls = namespace.get(strategy_name)
    if cls is None:
        print(f"   ❌ Class '{strategy_name}' not found after exec")
        return None

    # Get base params
    try:
        inst = cls()
        inst.init(data)
        base_params = inst.get_params()
    except Exception as e:
        print(f"   ❌ get_params() failed: {e}")
        return None

    from param_sweep import run_sweep
    results = run_sweep(
        cls, data, base_params=base_params,
        metric="sharpe_ratio", max_combos=max_combos,
        strategy_source=(code, strategy_name),  # allow parallel workers to re-exec
    )
    if not results:
        return None

    best = results[0]
    return {
        "params": best["params"],
        "sharpe": best["sharpe"],
        "cagr":   best["cagr"],
        "max_dd": best["max_dd"],
        "calmar": best["calmar"],
        "composite": calculate_composite_score(
            best["sharpe"], best["calmar"],
            best.get("sortino", best["sharpe"]),
            best["max_dd"], best.get("profit_factor", 1.0),
        ),
    }


# ─────────────────────────────────────────────────────────────
# Sweep — old-style strategies (regex path)
# ─────────────────────────────────────────────────────────────

def _sweep_with_regex(
    strategy_name: str,
    code: str,
    data: pd.DataFrame,
    history: dict,
    max_combos: int,
) -> Optional[Dict]:
    """
    Sweep a single old-style strategy by varying its numeric constants via regex.
    Returns best result dict if improved, else None.
    """
    class_name = strategy_name  # already e.g. "Strategy_Gen983"
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

    results.sort(key=lambda r: r['sharpe'], reverse=True)
    best = results[0]
    print(f"   🏆 Best: Sharpe={best['sharpe']:.2f} CAGR={best['cagr']:.1%} "
          f"MaxDD={best['max_dd']:.1%} Comp={best['composite']:.4f}")
    print(f"          Params: {best['params']}")
    return best


# ─────────────────────────────────────────────────────────────
# Unified sweep dispatcher
# ─────────────────────────────────────────────────────────────

def sweep_strategy_retroactively(
    strategy_name: str,
    code: str,
    data: pd.DataFrame,
    history: dict,
    max_combos: int = 200,
) -> Optional[Dict]:
    """
    Dispatch to the appropriate sweep path based on strategy format.
    New-style (has get_params): use run_sweep() for precision.
    Old-style: use regex substitution.
    """
    if 'def get_params' in code:
        print(f"   → New-style strategy: using get_params() path")
        return _sweep_with_get_params(strategy_name, code, data, max_combos)
    else:
        print(f"   → Old-style strategy: using regex path")
        return _sweep_with_regex(strategy_name, code, data, history, max_combos)


# ─────────────────────────────────────────────────────────────
# Write best params back to .py __init__ defaults
# ─────────────────────────────────────────────────────────────

def update_py_file_defaults(fpath: Path, best_params: Dict):
    """
    Write swept best params as new __init__ defaults in the .py file.
    Only applied to new-style strategies (those with get_params).
    Uses targeted regex substitution on the __init__ signature.
    """
    code = fpath.read_text(encoding='utf-8', errors='ignore')
    changed = []
    for param_name, param_value in best_params.items():
        # Format value: drop .0 suffix for whole numbers
        if isinstance(param_value, float) and param_value == int(param_value):
            new_str = str(int(param_value))
        else:
            new_str = str(param_value)
        # Match: param_name=OLD_VAL inside __init__ signature
        pattern = rf'(\b{re.escape(param_name)}\s*=\s*)[^\s,)]*'
        new_code, n = re.subn(pattern, rf'\g<1>{new_str}', code, count=1)
        if n:
            changed.append(f"{param_name}={param_value}")
            code = new_code
    if changed:
        fpath.write_text(code, encoding='utf-8')
        print(f"   ✏️  Updated __init__ defaults: {', '.join(changed)}")
    else:
        print(f"   ⚠️  No __init__ defaults updated (params not found in signature)")


# ─────────────────────────────────────────────────────────────
# History update
# ─────────────────────────────────────────────────────────────

def update_history_with_swept_result(history: dict, strategy_name: str, best: Dict):
    """Overwrite the strategy's metrics in history with the swept best result."""
    for s in history.get("strategies", []):
        if s["name"] == strategy_name:
            old_sharpe = s.get("sharpe", 0)
            s["sharpe"]       = best["sharpe"]
            s["cagr"]         = best["cagr"]
            s["max_dd"]       = best["max_dd"]
            s["calmar"]       = best["calmar"]
            s["composite"]    = best["composite"]
            s["swept_params"] = best["params"]
            s["swept_at"]     = datetime.now().isoformat()
            print(f"   ✅ History updated: Sharpe {old_sharpe:.2f} → {best['sharpe']:.2f}")
            return
    print(f"   ⚠️ {strategy_name} not found in history")


# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Retroactive param sweep for existing strategies")
    parser.add_argument('--top',  type=int, default=15,
                        help='Sweep top N strategies by Sharpe (ignored if --all)')
    parser.add_argument('--all',  action='store_true',
                        help='Sweep ALL qualifying strategies (not just top N)')
    parser.add_argument('--name', type=str, default=None,
                        help='Sweep a specific strategy by name')
    parser.add_argument('--min-sharpe',    type=float, default=0.40,
                        help='Min Sharpe to qualify (default: 0.40)')
    parser.add_argument('--min-composite', type=float, default=None,
                        help='(Legacy) Min composite score — overrides --min-sharpe if set')
    parser.add_argument('--max-combos',    type=int,   default=200,
                        help='Max param combinations per strategy')
    parser.add_argument('--update-file',   action='store_true', default=True,
                        help='Write best params back to .py __init__ defaults (new-style only)')
    parser.add_argument('--no-update-file', action='store_true',
                        help='Disable writing best params back to .py files')
    parser.add_argument('--force', action='store_true',
                        help='Re-sweep strategies already swept (otherwise skipped)')
    args = parser.parse_args()

    # --no-update-file overrides --update-file
    if args.no_update_file:
        args.update_file = False

    print("=" * 60)
    print("🔬 Retroactive Parameter Sweep")
    print("=" * 60)

    # Load data and history
    data    = get_enriched_data()
    with open(HISTORY_FILE, encoding='utf-8') as f:
        history = json.load(f)

    strategies = history.get("strategies", [])

    # Filter qualifying strategies
    if args.min_composite is not None:
        # Legacy composite-based filter
        rankable = [s for s in strategies if s.get("success") and
                    s.get("composite", 0) >= args.min_composite]
        print(f"Filter: composite >= {args.min_composite} (legacy mode)")
    else:
        rankable = [s for s in strategies if s.get("success") and
                    s.get("sharpe", 0) >= args.min_sharpe]
        print(f"Filter: Sharpe >= {args.min_sharpe}")

    # Deduplicate by name — keep highest Sharpe per strategy name
    seen_names: Dict[str, dict] = {}
    for s in rankable:
        name = s.get("name", "")
        if name not in seen_names or s.get("sharpe", 0) > seen_names[name].get("sharpe", 0):
            seen_names[name] = s
    rankable = list(seen_names.values())
    print(f"Unique strategies after dedup: {len(rankable)}")

    # Pick targets
    if args.name:
        targets = [s for s in strategies if s["name"] == args.name]
        if not targets:
            print(f"❌ Strategy '{args.name}' not found in history")
            return
    elif args.all:
        targets = sorted(rankable, key=lambda x: x.get("sharpe", 0), reverse=True)
        print(f"Mode: ALL ({len(targets)} strategies)")
    else:
        targets = sorted(rankable, key=lambda x: x.get("sharpe", 0), reverse=True)[:args.top]
        print(f"Mode: top {args.top}")

    print(f"Targets: {len(targets)} strategies | max_combos={args.max_combos} | "
          f"update_file={args.update_file} | force={args.force}")
    print()

    log_entries = []
    improved    = 0
    skipped     = 0

    for rank, s in enumerate(targets, 1):
        name = s["name"]
        num  = name.replace("Strategy_Gen", "")
        fpath = GENERATED_DIR / f"strategy_gen_{num}.py"

        print(f"[{rank}/{len(targets)}] {name}  "
              f"Sharpe={s.get('sharpe',0):.2f} Comp={s.get('composite',0):.4f}")

        if not fpath.exists():
            print(f"   ⚠️ .py file not found, skipping")
            skipped += 1
            continue

        code = fpath.read_text(encoding='utf-8', errors='ignore')

        # Skip already-swept strategies unless --force
        if 'swept_at' in s and not args.force:
            print(f"   ⏭ Already swept at {s['swept_at'][:10]}, skipping (use --force to re-sweep)")
            skipped += 1
            continue

        best = sweep_strategy_retroactively(name, code, data, history,
                                            max_combos=args.max_combos)

        entry = {
            "name": name,
            "original_sharpe":    s.get("sharpe", 0),
            "original_composite": s.get("composite", 0),
        }

        # Improvement threshold: +0.02 Sharpe
        if best and best["sharpe"] > s.get("sharpe", 0) + 0.02:
            update_history_with_swept_result(history, name, best)
            # Write best params back to .py file (new-style only)
            if args.update_file and 'def get_params' in code:
                update_py_file_defaults(fpath, best["params"])
            entry.update({"status": "improved", "best": best})
            improved += 1
            # Incremental save after each improvement (crash-safe)
            save_history_merge(history)
        else:
            entry["status"] = "no_improvement"
            print(f"   — No improvement over base Sharpe={s.get('sharpe',0):.2f}")

        log_entries.append(entry)
        print()

    # Save updated history (merge-safe: won't lose strategies added by main_loop)
    save_history_merge(history)
    print(f"💾 History saved (merge-safe).")

    # Save sweep log
    existing_log = []
    if RETRO_LOG_FILE.exists():
        with open(RETRO_LOG_FILE, encoding='utf-8') as f:
            existing_log = json.load(f)
    existing_log.append({
        "run_at":   datetime.now().isoformat(),
        "targets":  len(targets),
        "improved": improved,
        "skipped":  skipped,
        "entries":  log_entries,
    })
    with open(RETRO_LOG_FILE, 'w', encoding='utf-8') as f:
        json.dump(existing_log, f, indent=2, default=str)

    print("=" * 60)
    print(f"✅ Done: {improved}/{len(targets)} improved, {skipped} skipped")
    print("=" * 60)


if __name__ == "__main__":
    main()
