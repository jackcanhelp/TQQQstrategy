#!/usr/bin/env python3
"""
QuantEvolution TQQQ
====================
Autonomous AI Research Lab for TQQQ Trading Strategies

This system uses Gemini AI to evolve trading strategies by:
1. Generating new strategy ideas
2. Writing Python code for those strategies
3. Backtesting on historical TQQQ data
4. Learning from failures and iterating
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv

from strategy_base import BaseStrategy, BuyAndHold, SimpleSMA
from backtest import BacktestEngine, BacktestResult, compare_strategies
from researcher import StrategyGenerator, StrategySandbox

load_dotenv()


def download_data(ticker: str = "TQQQ", force_refresh: bool = False) -> pd.DataFrame:
    """
    Download or load cached TQQQ data.
    """
    cache_file = Path(f"{ticker}_data.pkl")

    # Check cache (refresh if older than 1 day)
    if cache_file.exists() and not force_refresh:
        cached_data = pd.read_pickle(cache_file)
        cache_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
        if cache_age.days < 1:
            print(f"📊 Loaded cached {ticker} data: {len(cached_data)} rows")
            return cached_data

    # Download fresh data
    print(f"📥 Downloading {ticker} data from Yahoo Finance...")
    ticker_obj = yf.Ticker(ticker)
    data = ticker_obj.history(period="max", auto_adjust=True)

    if len(data) == 0:
        raise ValueError(f"No data returned for {ticker}")

    # Clean up
    data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
    data = data.dropna()

    # Cache
    data.to_pickle(cache_file)
    print(f"✅ Downloaded {len(data)} rows ({data.index[0].date()} to {data.index[-1].date()})")

    return data


def run_baseline_benchmarks(data: pd.DataFrame) -> dict:
    """
    Run baseline strategies for comparison.
    """
    print("\n" + "="*60)
    print("📈 RUNNING BASELINE BENCHMARKS")
    print("="*60)

    engine = BacktestEngine(data)
    results = {}

    # Buy and Hold
    bh = BuyAndHold()
    bh_result = engine.run(bh)
    results['BuyAndHold'] = bh_result
    print(f"\n🔹 Buy & Hold: CAGR={bh_result.cagr:.1%}, Sharpe={bh_result.sharpe_ratio:.2f}, MaxDD={bh_result.max_drawdown:.1%}")

    # Simple SMA
    sma = SimpleSMA(fast_period=20, slow_period=50)
    sma_result = engine.run(sma)
    results['SimpleSMA'] = sma_result
    print(f"🔹 SMA(20,50): CAGR={sma_result.cagr:.1%}, Sharpe={sma_result.sharpe_ratio:.2f}, MaxDD={sma_result.max_drawdown:.1%}")

    return results


def run_evolution_loop(
    data: pd.DataFrame,
    max_iterations: int = 10,
    target_sharpe: float = 1.5,
    max_fix_attempts: int = 3
) -> None:
    """
    Main AI evolution loop.
    """
    print("\n" + "="*60)
    print("🧬 STARTING AI EVOLUTION LOOP")
    print(f"   Target Sharpe: {target_sharpe}")
    print(f"   Max Iterations: {max_iterations}")
    print("="*60)

    generator = StrategyGenerator()
    engine = BacktestEngine(data)
    sandbox = StrategySandbox()

    for i in range(max_iterations):
        strategy_id = generator.get_next_strategy_id()
        print(f"\n{'─'*50}")
        print(f"🔬 ITERATION {strategy_id}")
        print(f"{'─'*50}")

        # Step 1: Generate strategy idea
        print("\n💡 Generating strategy idea...")
        try:
            idea = generator.generate_strategy_idea()
            print(f"   Idea: {idea[:200]}...")
        except Exception as e:
            print(f"❌ Failed to generate idea: {e}")
            continue

        # Step 2: Generate code
        print("\n💻 Generating strategy code...")
        try:
            code, file_path = generator.generate_strategy_code(idea, strategy_id)
            print(f"   Saved to: {file_path}")
        except Exception as e:
            print(f"❌ Failed to generate code: {e}")
            continue

        # Step 3: Load and test strategy
        class_name = f"Strategy_Gen{strategy_id}"
        strategy = None
        success = False
        error_msg = ""

        for attempt in range(max_fix_attempts):
            try:
                print(f"\n🧪 Testing strategy (attempt {attempt + 1}/{max_fix_attempts})...")
                strategy = sandbox.load_strategy(file_path, class_name)
                success, error_msg = sandbox.test_strategy(strategy, data)

                if success:
                    print("   ✅ Strategy loaded successfully!")
                    break
                else:
                    print(f"   ⚠️ Test failed: {error_msg[:100]}")
                    if attempt < max_fix_attempts - 1:
                        print("   🔧 Asking AI to fix...")
                        code, file_path = generator.fix_strategy_code(code, error_msg, strategy_id)

            except Exception as e:
                error_msg = str(e)
                print(f"   ❌ Load error: {error_msg[:100]}")
                if attempt < max_fix_attempts - 1:
                    print("   🔧 Asking AI to fix...")
                    try:
                        code, file_path = generator.fix_strategy_code(code, error_msg, strategy_id)
                    except Exception as fix_e:
                        print(f"   ❌ Fix failed: {fix_e}")
                        break

        if not success or strategy is None:
            print(f"\n❌ Strategy {strategy_id} failed after {max_fix_attempts} attempts")
            generator.record_result(
                strategy_id=strategy_id,
                strategy_name=class_name,
                idea=idea,
                sharpe=0.0,
                cagr=0.0,
                max_dd=0.0,
                failure_analysis=f"Code error: {error_msg[:200]}",
                success=False
            )
            continue

        # Step 4: Run backtest
        print("\n📊 Running backtest...")
        try:
            result = engine.run(strategy)
            failure_analysis = result.get_failure_analysis()

            print(f"\n📈 RESULTS for {class_name}:")
            print(f"   CAGR:       {result.cagr:.1%}")
            print(f"   Sharpe:     {result.sharpe_ratio:.2f}")
            print(f"   Sortino:    {result.sortino_ratio:.2f}")
            print(f"   Max DD:     {result.max_drawdown:.1%}")
            print(f"   Win Rate:   {result.win_rate:.1%}")
            print(f"   Analysis:   {failure_analysis}")

            # Record result
            generator.record_result(
                strategy_id=strategy_id,
                strategy_name=class_name,
                idea=idea,
                sharpe=result.sharpe_ratio,
                cagr=result.cagr,
                max_dd=result.max_drawdown,
                failure_analysis=failure_analysis,
                success=True
            )

            # Check if we hit target
            if result.sharpe_ratio >= target_sharpe:
                print(f"\n🎯 TARGET SHARPE ACHIEVED! ({result.sharpe_ratio:.2f} >= {target_sharpe})")
                print("   Stopping evolution loop.")
                break

        except Exception as e:
            print(f"❌ Backtest failed: {e}")
            generator.record_result(
                strategy_id=strategy_id,
                strategy_name=class_name,
                idea=idea,
                sharpe=0.0,
                cagr=0.0,
                max_dd=0.0,
                failure_analysis=f"Backtest error: {str(e)[:200]}",
                success=False
            )

    # Final summary
    print("\n" + "="*60)
    print("📋 EVOLUTION COMPLETE")
    print("="*60)
    print(f"Total iterations: {generator.history['total_iterations']}")
    print(f"Best Sharpe: {generator.history['best_sharpe']:.2f}")
    print(f"Best Strategy: {generator.history['best_strategy']}")


def plot_results(data: pd.DataFrame, results: dict) -> None:
    """
    Plot equity curves for comparison.
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Equity curves
    ax1 = axes[0]
    for name, result in results.items():
        ax1.plot(result.equity_curve.index, result.equity_curve.values, label=name)
    ax1.set_title('Equity Curves')
    ax1.set_ylabel('Portfolio Value')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Drawdowns
    ax2 = axes[1]
    for name, result in results.items():
        ax2.fill_between(result.drawdown_series.index, result.drawdown_series.values, 0, alpha=0.3, label=name)
    ax2.set_title('Drawdowns')
    ax2.set_ylabel('Drawdown %')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('results.png', dpi=150)
    print("\n📊 Results saved to results.png")


def main():
    parser = argparse.ArgumentParser(description='QuantEvolution TQQQ - AI Strategy Evolution')
    parser.add_argument('--iterations', type=int, default=10, help='Max evolution iterations')
    parser.add_argument('--target-sharpe', type=float, default=1.5, help='Target Sharpe ratio')
    parser.add_argument('--refresh-data', action='store_true', help='Force refresh data from Yahoo')
    parser.add_argument('--baseline-only', action='store_true', help='Only run baseline benchmarks')

    args = parser.parse_args()

    print("="*60)
    print("🚀 QuantEvolution TQQQ")
    print("   Autonomous AI Research Lab for TQQQ Trading")
    print("="*60)

    # Check API key
    if not os.getenv("GOOGLE_API_KEY"):
        print("\n❌ Error: GOOGLE_API_KEY not set!")
        print("   Create a .env file with your API key:")
        print("   GOOGLE_API_KEY=your_key_here")
        sys.exit(1)

    # Download data
    data = download_data(force_refresh=args.refresh_data)

    # Run baselines
    baseline_results = run_baseline_benchmarks(data)

    if args.baseline_only:
        print("\n✅ Baseline benchmarks complete.")
        return

    # Run evolution
    run_evolution_loop(
        data=data,
        max_iterations=args.iterations,
        target_sharpe=args.target_sharpe
    )

    # Plot
    try:
        plot_results(data, baseline_results)
    except Exception as e:
        print(f"⚠️ Could not generate plot: {e}")


if __name__ == "__main__":
    main()
