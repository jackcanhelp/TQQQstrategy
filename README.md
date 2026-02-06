# QuantEvolution TQQQ

Autonomous AI Research Lab for TQQQ Trading Strategies

## Overview

This system uses **Gemini 2.5-flash-lite** to autonomously evolve trading strategies for TQQQ (3x Leveraged NASDAQ-100 ETF). It doesn't just optimize parameters—it **evolves logic** by writing new Python code, testing it, analyzing failures, and iterating.

## Features

- **AI-Driven Strategy Generation**: Gemini writes complete Python strategy classes
- **Sandboxed Execution**: Safely load and run AI-generated code
- **Comprehensive Backtesting**: CAGR, Sharpe, Sortino, Max Drawdown, Crisis Analysis
- **Learning Loop**: Failures are analyzed and fed back to improve next iteration
- **Full TQQQ History**: Uses all available data since inception (2010)

## Project Structure

```
TQQQstrategy/
├── main.py                 # Main entry point & evolution loop
├── strategy_base.py        # Abstract base class for strategies
├── backtest.py             # Backtesting engine & metrics
├── researcher.py           # AI code generation with Gemini
├── generated_strategies/   # AI-generated strategy code
├── history_of_thoughts.json # Evolution history & learning
├── requirements.txt        # Python dependencies
└── .env                    # API keys (not tracked)
```

## Quick Start

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up API key**
   ```bash
   cp .env.example .env
   # Edit .env and add your GOOGLE_API_KEY
   ```

3. **Run evolution**
   ```bash
   python main.py --iterations 10 --target-sharpe 1.5
   ```

## Usage

```bash
# Run with default settings (10 iterations, target Sharpe 1.5)
python main.py

# Run more iterations with higher target
python main.py --iterations 50 --target-sharpe 2.0

# Only run baseline benchmarks
python main.py --baseline-only

# Force refresh data from Yahoo Finance
python main.py --refresh-data
```

## How It Works

### The Evolution Loop

1. **Idea Generation**: Gemini analyzes past results and proposes a new strategy concept
2. **Code Writing**: Gemini writes a complete Python class implementing the strategy
3. **Sandboxed Testing**: The code is dynamically loaded and validated
4. **Backtesting**: Full historical simulation with comprehensive metrics
5. **Learning**: Failures are analyzed (e.g., "2022 crash caused 60% drawdown") and context is fed back to Gemini
6. **Iteration**: Repeat with improved understanding

### Strategy Interface

All strategies must inherit from `BaseStrategy`:

```python
class MyStrategy(BaseStrategy):
    def init(self, data: pd.DataFrame) -> None:
        # Calculate indicators
        self.data = data
        self.sma = data['Close'].rolling(20).mean()

    def generate_signals(self) -> pd.Series:
        # Return 0.0-1.0 position weights
        return (self.data['Close'] > self.sma).astype(float)
```

## Metrics Tracked

- **CAGR**: Compound Annual Growth Rate
- **Sharpe Ratio**: Risk-adjusted returns
- **Sortino Ratio**: Downside risk-adjusted returns
- **Max Drawdown**: Worst peak-to-trough decline
- **Win Rate**: Percentage of profitable days
- **Crisis Analysis**: Performance during known crashes (2020 COVID, 2022 Bear)

## License

MIT
