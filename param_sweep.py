"""
Parameter Sweep Engine
======================
Once a good strategy structure is found, systematically optimize
its parameters via grid search to find the best combination.

Works with any strategy that has a `get_params()` method.
"""

import itertools
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from copy import deepcopy

import pandas as pd
import numpy as np

from backtest import BacktestEngine, BacktestResult


SWEEP_RESULTS_FILE = Path("sweep_results.json")


def generate_param_grid(base_params: Dict, sweep_config: Optional[Dict] = None) -> List[Dict]:
    """
    Generate parameter combinations for grid search.

    Args:
        base_params: Current parameter values (from strategy.get_params())
        sweep_config: Optional custom ranges. If None, auto-generates ±20% variations.

    Returns:
        List of parameter dictionaries to test
    """
    if sweep_config is None:
        sweep_config = _auto_sweep_config(base_params)

    # Generate all combinations
    param_names = list(sweep_config.keys())
    param_values = [sweep_config[k] for k in param_names]
    combinations = list(itertools.product(*param_values))

    param_dicts = []
    for combo in combinations:
        params = base_params.copy()
        for name, value in zip(param_names, combo):
            params[name] = value
        param_dicts.append(params)

    return param_dicts


def _auto_sweep_config(base_params: Dict) -> Dict:
    """
    Auto-generate sweep ranges: test ±20% around each parameter.
    Integer params get integer variations, floats get float variations.
    """
    config = {}
    for name, value in base_params.items():
        if isinstance(value, bool):
            config[name] = [True, False]
        elif isinstance(value, int):
            step = max(1, int(value * 0.1))
            low = max(1, value - 2 * step)
            high = value + 2 * step
            config[name] = list(range(low, high + 1, step))
        elif isinstance(value, float):
            step = max(0.1, value * 0.1)
            values = [round(value + i * step, 2) for i in range(-2, 3)]
            config[name] = [v for v in values if v > 0]
    return config


def run_sweep(
    strategy_class,
    data: pd.DataFrame,
    base_params: Dict,
    sweep_config: Optional[Dict] = None,
    metric: str = "sharpe_ratio",
    top_n: int = 10,
    max_combos: int = 500,
) -> List[Dict]:
    """
    Run parameter sweep on a strategy class.

    Args:
        strategy_class: The strategy class (not instance)
        data: OHLCV DataFrame
        base_params: Base parameter values
        sweep_config: Custom sweep ranges (optional)
        metric: Optimization metric ('sharpe_ratio', 'calmar_ratio', 'cagr')
        top_n: Number of top results to return
        max_combos: Maximum combinations to test (random sample if exceeded)

    Returns:
        List of top results sorted by metric, each containing params and metrics
    """
    param_grid = generate_param_grid(base_params, sweep_config)

    # Cap combinations
    if len(param_grid) > max_combos:
        import random
        # Always include the base params
        param_grid = [base_params] + random.sample(
            [p for p in param_grid if p != base_params],
            max_combos - 1
        )

    engine = BacktestEngine(data)
    results = []
    total = len(param_grid)

    print(f"   🔍 Parameter sweep: testing {total} combinations...")

    for idx, params in enumerate(param_grid):
        try:
            strategy = strategy_class(**params)
            strategy.init(data)
            signals = strategy.generate_signals()

            # Quick sanity check
            if signals.isna().all() or (signals == 0).all():
                continue

            bt = engine.run(strategy)

            results.append({
                "params": params,
                "sharpe": bt.sharpe_ratio,
                "cagr": bt.cagr,
                "max_dd": bt.max_drawdown,
                "calmar": bt.calmar_ratio,
                "sortino": bt.sortino_ratio,
                "trades": bt.total_trades,
                "time_in_market": bt.time_in_market,
            })

            if (idx + 1) % 50 == 0:
                best_so_far = max(results, key=lambda r: r.get(metric.replace('_ratio', ''), r.get('sharpe', 0)))
                print(f"      [{idx+1}/{total}] Best so far: "
                      f"Sharpe={best_so_far['sharpe']:.2f}, "
                      f"MaxDD={best_so_far['max_dd']:.1%}")

        except Exception as e:
            continue

    if not results:
        print("   ❌ No valid results from sweep")
        return []

    # Sort by target metric
    metric_key = metric.replace('_ratio', '')
    results.sort(key=lambda r: r.get(metric_key, 0), reverse=True)

    top_results = results[:top_n]

    print(f"\n   🏆 Top {min(top_n, len(top_results))} parameter sets:")
    for i, r in enumerate(top_results[:5], 1):
        print(f"      #{i}: Sharpe={r['sharpe']:.2f}, CAGR={r['cagr']:.1%}, "
              f"MaxDD={r['max_dd']:.1%}, Calmar={r['calmar']:.2f}")
        changed = {k: v for k, v in r['params'].items()
                   if v != base_params.get(k)}
        if changed:
            print(f"          Changed: {changed}")

    return top_results


def sweep_champion(data: pd.DataFrame) -> List[Dict]:
    """
    Run parameter sweep specifically on the Champion RVI strategy.
    Tests key parameters that most affect performance.
    """
    from champion_rvi import ChampionRVI

    base = ChampionRVI()
    base_params = base.get_params()

    # Focused sweep config for Champion RVI
    sweep_config = {
        'stdev_length': [20, 26, 30, 34, 40, 50],
        'smooth_length': [10, 14, 20, 26, 30],
        'buy_trigger': [50, 55, 59, 63, 68],
        'sell_high': [70, 73, 76, 80, 85],
        'sell_low': [35, 38, 42, 46, 50],
        'volatility_factor': [1.2, 1.5, 1.8, 2.0, 2.5],
        'atr_period': [10, 14, 20],
    }

    print("═══════════════════════════════════════════")
    print("🔬 Champion RVI Parameter Sweep")
    print(f"   Base params: {base_params}")
    print("═══════════════════════════════════════════")

    results = run_sweep(
        ChampionRVI, data, base_params,
        sweep_config=sweep_config,
        metric="sharpe_ratio",
        max_combos=400,
    )

    # Save results
    _save_sweep_results("ChampionRVI", results)

    return results


def sweep_volume_breakout(data: pd.DataFrame) -> List[Dict]:
    """
    Run parameter sweep on the Volume Breakout strategy.
    Tests key parameters that most affect performance.
    """
    from champion_volume_breakout import ChampionVolumeBreakout

    base = ChampionVolumeBreakout(exit_mode='atr')
    base_params = base.get_params()

    # Focused sweep config for Volume Breakout
    sweep_config = {
        'ma_length': [10, 15, 20, 30, 50],
        'vol_multiplier': [1.5, 2.0, 2.5, 3.0],
        'atr_tp_mult': [1.5, 2.0, 2.5, 3.0],
        'atr_sl_mult': [1.0, 1.5, 2.0],
    }

    print("═══════════════════════════════════════════")
    print("🔬 Volume Breakout Parameter Sweep")
    print(f"   Base params: {base_params}")
    print("═══════════════════════════════════════════")

    results = run_sweep(
        ChampionVolumeBreakout, data, base_params,
        sweep_config=sweep_config,
        metric="sharpe_ratio",
        max_combos=400,
    )

    _save_sweep_results("VolumeBreakout", results)

    return results


def sweep_generated_strategy(
    strategy_class,
    data: pd.DataFrame,
    base_params: Optional[Dict] = None,
) -> List[Dict]:
    """
    Sweep a generated strategy that has get_params().
    Auto-detects parameter ranges.
    """
    if base_params is None:
        instance = strategy_class()
        if hasattr(instance, 'get_params'):
            base_params = instance.get_params()
        else:
            print("   ⚠️ Strategy has no get_params() method, skipping sweep")
            return []

    results = run_sweep(
        strategy_class, data, base_params,
        metric="sharpe_ratio",
        max_combos=300,
    )

    _save_sweep_results(strategy_class.__name__, results)
    return results


def _save_sweep_results(strategy_name: str, results: List[Dict]):
    """Save sweep results to JSON."""
    existing = []
    if SWEEP_RESULTS_FILE.exists():
        with open(SWEEP_RESULTS_FILE) as f:
            existing = json.load(f)

    existing.append({
        "strategy": strategy_name,
        "timestamp": datetime.now().isoformat(),
        "total_tested": len(results),
        "top_results": results[:10],
    })

    with open(SWEEP_RESULTS_FILE, 'w') as f:
        json.dump(existing, f, indent=2, default=str)


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    from indicator_pool import get_enriched_data

    data = get_enriched_data()
    results = sweep_champion(data)

    if results:
        best = results[0]
        print(f"\n🎯 BEST PARAMETERS FOUND:")
        print(f"   Sharpe: {best['sharpe']:.2f}")
        print(f"   CAGR: {best['cagr']:.1%}")
        print(f"   MaxDD: {best['max_dd']:.1%}")
        print(f"   Calmar: {best['calmar']:.2f}")
        print(f"   Params: {best['params']}")
