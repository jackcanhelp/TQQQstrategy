"""
Strategy Base Class
====================
All AI-generated strategies MUST inherit from this class.
This enforces a strict interface for the backtester.
"""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np


class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies.

    Strategies must implement:
    - init(data): Initialize indicators and state
    - generate_signals(): Return position weights (0.0 to 1.0)
    """

    def __init__(self):
        self.data: pd.DataFrame = None
        self.signals: pd.Series = None
        self.name: str = self.__class__.__name__

    @abstractmethod
    def init(self, data: pd.DataFrame) -> None:
        """
        Initialize the strategy with OHLCV data.

        Args:
            data: DataFrame with columns ['Open', 'High', 'Low', 'Close', 'Volume']
                  Index should be DatetimeIndex

        This method should:
        - Store self.data = data
        - Calculate any indicators needed
        - Prepare internal state
        """
        pass

    @abstractmethod
    def generate_signals(self) -> pd.Series:
        """
        Generate position signals for each day.

        Returns:
            pd.Series: Position weights between 0.0 (no position) and 1.0 (full position)
                      Index should match self.data.index

        Notes:
        - Return 0.0 for days you want to be in cash
        - Return 1.0 for days you want full TQQQ exposure
        - Values between 0-1 represent partial positions
        - Do NOT return values > 1.0 (no leverage beyond TQQQ)
        - Do NOT return negative values (no shorting)
        """
        pass

    def validate_signals(self, signals: pd.Series) -> pd.Series:
        """
        Validate and clip signals to [0, 1] range.
        Called automatically by the backtester.
        """
        signals = signals.clip(0.0, 1.0)
        signals = signals.fillna(0.0)
        return signals

    def get_description(self) -> str:
        """
        Return a description of the strategy logic.
        Override this to provide context for the AI researcher.
        """
        return f"Strategy: {self.name}"


class BuyAndHold(BaseStrategy):
    """
    Baseline strategy: Always hold 100% TQQQ.
    Used as a benchmark for comparison.
    """

    def init(self, data: pd.DataFrame) -> None:
        self.data = data

    def generate_signals(self) -> pd.Series:
        return pd.Series(1.0, index=self.data.index)

    def get_description(self) -> str:
        return "Buy and Hold: 100% TQQQ at all times"


class SimpleSMA(BaseStrategy):
    """
    Simple SMA crossover strategy.
    Used as a baseline AI example.
    """

    def __init__(self, fast_period: int = 10, slow_period: int = 50):
        super().__init__()
        self.fast_period = fast_period
        self.slow_period = slow_period

    def init(self, data: pd.DataFrame) -> None:
        self.data = data
        self.sma_fast = data['Close'].rolling(self.fast_period).mean()
        self.sma_slow = data['Close'].rolling(self.slow_period).mean()

    def generate_signals(self) -> pd.Series:
        signals = (self.sma_fast > self.sma_slow).astype(float)
        return signals

    def get_description(self) -> str:
        return f"SMA Crossover: Fast={self.fast_period}, Slow={self.slow_period}"
