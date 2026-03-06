"""
MaxDD Rescue Sweep
==================
Find historical strategies that were rejected ONLY due to MaxDD > -50%,
run a parameter sweep to find better risk settings, and promote them to
successful strategies in history.

Target: strategies with MaxDD -50% to -65%, Sharpe >= 0.5, have get_params()

Usage:
    py -3.14 maxdd_rescue_sweep.py                     # rescue top-Sharpe candidates
    py -3.14 maxdd_rescue_sweep.py --max-combos 120    # more thorough sweep
    py -3.14 maxdd_rescue_sweep.py --min-sharpe 0.7    # only higher-quality
    py -3.14 maxdd_rescue_sweep.py --name Strategy_Gen2846  # single strategy
"""

import argparse
import json
import sys
import time
from pathlib import Path
from datetime import datetime

import pandas as pd

HISTORY_FILE = Path("history_of_thoughts.json")
GENERATED_DIR = Path("generated_strategies")

# Hard filter thresholds (must match main_loop.py)
HARD_FILTER_MAX_DD    = -0.50
HARD_FILTER_MIN_SHARPE = 0.3
HARD_FILTER_MIN_TRADES = 10
HARD_FILTER_MIN_EXPOSURE = 0.05


def load_history():
    with open(HISTORY_FILE, encoding='utf-8') as f:
        return json.load(f)


def save_history(history):
    for attempt in range(5):
        try:
            tmp = HISTORY_FILE.with_suffix('.tmp')
            with open(tmp, 'w', encoding='utf-8') as f:
                json.dump(history, f, indent=2, default=str)
            tmp.replace(HISTORY_FILE)
            return
        except OSError as e:
            if attempt < 4:
                time.sleep(1 + attempt)
            else:
                print(f"⚠️ save_history failed: {e}")


def passes_hard_filter(r: dict) -> bool:
    return (r.get('max_dd', -99) > HARD_FILTER_MAX_DD
            and r.get('sharpe', 0) >= HARD_FILTER_MIN_SHARPE
            and r.get('trades', 0) >= HARD_FILTER_MIN_TRADES
            and r.get('time_in_market', 0) >= HARD_FILTER_MIN_EXPOSURE)


def find_rescue_candidates(history, min_sharpe=0.5, max_dd_range=(-0.65, -0.50), name_filter=None):
    """Return list of strategies eligible for MaxDD rescue."""
    strategies = history.get('strategies', [])
    candidates = []
    for s in strategies:
        if s.get('success'):
            continue
        fa = s.get('failure_analysis', '')
        if 'MaxDD' not in fa and 'REJECTED' not in fa:
            continue
        # Only target MaxDD rejections (not other reasons)
        if 'MaxDD' not in fa:
            continue
        sharpe = s.get('sharpe', 0)
        max_dd = s.get('max_dd', -99)
        if sharpe < min_sharpe:
            continue
        if not (max_dd_range[0] <= max_dd <= max_dd_range[1]):
            continue
        name = s.get('name', '')
        if name_filter and name != name_filter:
            continue
        sid = name.replace('Strategy_Gen', '')
        fpath = GENERATED_DIR / f'strategy_gen_{sid}.py'
        if not fpath.exists():
            continue
        code = fpath.read_text(encoding='utf-8', errors='ignore')
        if 'def get_params' not in code:
            continue
        candidates.append({
            'name': name,
            'sid': sid,
            'fpath': fpath,
            'sharpe': sharpe,
            'max_dd': max_dd,
            'cagr': s.get('cagr', 0),
            'history_entry': s,
        })
    candidates.sort(key=lambda x: x['sharpe'], reverse=True)
    return candidates


def load_strategy_class(fpath: Path, class_name: str):
    code = fpath.read_text(encoding='utf-8', errors='ignore')
    namespace = {}
    exec(compile(code, str(fpath), 'exec'), namespace)
    cls = namespace.get(class_name)
    if cls is None:
        raise ValueError(f"Class {class_name} not found in {fpath}")
    return cls


def run_rescue_sweep(candidate: dict, data: pd.DataFrame, max_combos: int) -> dict | None:
    """
    Run param sweep on a MaxDD-failed strategy.
    Returns best passing result dict or None.
    """
    from param_sweep import run_sweep
    name = candidate['name']
    fpath = candidate['fpath']
    try:
        cls = load_strategy_class(fpath, name)
        instance = cls()
        base_params = instance.get_params()
    except Exception as e:
        print(f"   ⚠️ Failed to load {name}: {e}")
        return None

    results = run_sweep(cls, data, base_params, max_combos=max_combos, top_n=20)
    if not results:
        return None

    # Find first result that passes hard filter
    for r in results:
        if passes_hard_filter(r):
            return r
    return None


def promote_to_success(entry: dict, best: dict, original_max_dd: float):
    """Update history entry to mark strategy as rescued."""
    from main_loop import calculate_composite_score
    composite = calculate_composite_score(
        best['sharpe'], best['calmar'], best.get('sortino', best['sharpe']),
        best['max_dd'], best.get('profit_factor', 1.0)
    )
    entry['success'] = True
    entry['sharpe']  = best['sharpe']
    entry['cagr']    = best['cagr']
    entry['max_dd']  = best['max_dd']
    entry['calmar']  = best['calmar']
    entry['composite'] = composite
    entry['swept_params'] = best['params']
    entry['failure_analysis'] = (
        f"RESCUED: MaxDD swept from {original_max_dd:.1%} → {best['max_dd']:.1%} "
        f"| Sharpe={best['sharpe']:.2f} | params={best['params']}"
    )
    entry['rescued_from_maxdd'] = True
    entry['rescued_at'] = datetime.now().isoformat()
    return composite


def main():
    parser = argparse.ArgumentParser(description="MaxDD Rescue Sweep")
    parser.add_argument('--min-sharpe',  type=float, default=0.5,
                        help='Min Sharpe of failed strategy to attempt rescue (default: 0.5)')
    parser.add_argument('--max-dd-range', type=float, default=-0.65,
                        help='Worst MaxDD to attempt rescue (default: -0.65)')
    parser.add_argument('--max-combos',  type=int,   default=80,
                        help='Param combinations per strategy (default: 80)')
    parser.add_argument('--max-rescue',  type=int,   default=9999,
                        help='Max strategies to process (default: all)')
    parser.add_argument('--name',        type=str,   default=None,
                        help='Target a specific strategy name only')
    args = parser.parse_args()

    print("=" * 60)
    print("MaxDD Rescue Sweep")
    print(f"  min_sharpe={args.min_sharpe}, max_dd_range=[{args.max_dd_range:.0%}, -50%]")
    print(f"  max_combos={args.max_combos}, max_rescue={args.max_rescue}")
    print("=" * 60)

    # Load data
    print("\n📊 Loading data...")
    from indicator_pool import get_enriched_data
    data = get_enriched_data()
    print(f"   Data: {len(data)} rows, {data.columns.tolist()[:5]}...")

    # Load history
    history = load_history()
    total_iters = history.get('total_iterations', 0)
    print(f"   History: {total_iters} iterations loaded")

    # Find candidates
    candidates = find_rescue_candidates(
        history,
        min_sharpe=args.min_sharpe,
        max_dd_range=(args.max_dd_range, -0.50),
        name_filter=args.name,
    )
    candidates = candidates[:args.max_rescue]
    print(f"\n🎯 Rescue candidates: {len(candidates)} strategies")
    print(f"   (Sharpe>={args.min_sharpe}, MaxDD between {args.max_dd_range:.0%} and -50%, has get_params())\n")

    rescued = 0
    failed  = 0
    start_time = time.time()

    for i, cand in enumerate(candidates, 1):
        name    = cand['name']
        sharpe0 = cand['sharpe']
        maxdd0  = cand['max_dd']
        cagr0   = cand['cagr']
        elapsed = time.time() - start_time

        print(f"\n[{i}/{len(candidates)}] {name} | Sharpe={sharpe0:.2f} MaxDD={maxdd0:.1%} CAGR={cagr0:.1%}")
        print(f"   ⏱ Elapsed: {elapsed:.0f}s | Rescued so far: {rescued}")

        best = run_rescue_sweep(cand, data, max_combos=args.max_combos)

        if best:
            composite = promote_to_success(cand['history_entry'], best, maxdd0)
            save_history(history)
            rescued += 1
            print(f"   ✅ RESCUED! MaxDD {maxdd0:.1%}→{best['max_dd']:.1%} "
                  f"Sharpe {sharpe0:.2f}→{best['sharpe']:.2f} "
                  f"Composite={composite:.4f}")
        else:
            failed += 1
            print(f"   ❌ No passing params found (MaxDD cannot be rescued below -50%)")

    elapsed_total = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"MaxDD Rescue Complete")
    print(f"  Processed: {i if candidates else 0} strategies in {elapsed_total:.0f}s")
    print(f"  Rescued: {rescued} ✅")
    print(f"  Not rescuable: {failed} ❌")
    print("=" * 60)

    if rescued > 0:
        # Update best_composite in history if any rescued strategy beats it
        best_comp = history.get('best_composite', 0)
        all_successful = [s for s in history['strategies'] if s.get('success')]
        for s in all_successful:
            if s.get('composite', 0) > best_comp:
                best_comp = s['composite']
                history['best_composite'] = best_comp
                history['best_strategy']  = s['name']
        save_history(history)
        print(f"\n📈 History updated. Best composite now: {history.get('best_composite', 0):.4f}")


if __name__ == '__main__':
    main()
