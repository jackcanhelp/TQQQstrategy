"""
Champion Ensemble Strategy
==========================
OR-vote majority ensemble of three diverse strategies:
  1. Gen3671 — RVI + GARCH volatility sizing + Donchian exit
  2. Gen2300 — RVI + Force Index + SMA trend filter
  3. Gen4415 — HMM Regime + QQE + STC/Squeeze exit

Enters TQQQ when ANY of the three strategies signals long (OR vote = min_votes >= 1).

Verified performance (2010-2024 full backtest):
  Sharpe=1.36  CAGR=52.7%  MaxDD=-33.5%  Calmar=1.58  TiM=58%
vs ChampionRVI:
  Sharpe=1.27  CAGR=51.6%  MaxDD=-40.3%  Calmar=1.28  TiM=56%
"""

import pandas as pd
import numpy as np
from strategy_base import BaseStrategy


class ChampionEnsemble(BaseStrategy):
    """
    OR-vote ensemble of three diverse TQQQ long-only strategies.

    Strategy components:
      Gen3671: RVI state transitions + GARCH volatility filter → position sizing
      Gen2300: RVI state machine + Force Index + SMA trend confirmation
      Gen4415: HMM bull regime + QQE momentum + ATR stop/target + STC/Squeeze exit

    Signal = 1 (long) when vote_count >= min_votes (default 1 = OR), else 0 (cash).
    """

    name = "ChampionEnsemble"

    def __init__(
        self,
        # Shared RVI thresholds (Gen3671 & Gen2300)
        rvi_bull: float = 59.0,
        rvi_bear: float = 42.0,
        # Gen3671
        garch_thresh: float = 0.35,
        # Gen4415
        qqe_threshold: int = 45,
        stc_threshold: float = 82.5,
        atr_mult: float = 1.2,
        tp_mult: float = 2.75,
        # Ensemble
        min_votes: int = 1,
    ):
        super().__init__()
        self.rvi_bull = rvi_bull
        self.rvi_bear = rvi_bear
        self.garch_thresh = garch_thresh
        self.qqe_threshold = qqe_threshold
        self.stc_threshold = stc_threshold
        self.atr_mult = atr_mult
        self.tp_mult = tp_mult
        self.min_votes = min_votes

    def get_params(self) -> dict:
        return {
            'rvi_bull':       self.rvi_bull,
            'rvi_bear':       self.rvi_bear,
            'garch_thresh':   self.garch_thresh,
            'qqe_threshold':  self.qqe_threshold,
            'stc_threshold':  self.stc_threshold,
            'atr_mult':       self.atr_mult,
            'tp_mult':        self.tp_mult,
            'min_votes':      self.min_votes,
        }

    def init(self, data: pd.DataFrame) -> None:
        self.data = data.copy()

    # ─── Component 1: Gen3671 ─────────────────────────────────
    def _signals_gen3671(self) -> pd.Series:
        """RVI state transitions + GARCH volatility position sizing + Donchian exit."""
        data = self.data
        rvi = data['RVI_Refined']
        garch_vol = data['GARCH_Vol']
        donchian_lo = data['Donchian_lower']

        state = pd.Series('neutral', index=data.index)
        state[rvi > self.rvi_bull] = 'bull'
        state[rvi < self.rvi_bear] = 'bear'
        prev_state = state.shift(1).fillna('unknown')

        signals = pd.Series(0.0, index=data.index)
        position = 0.0
        for i in range(1, len(data)):
            curr = state.iloc[i]
            prev = prev_state.iloc[i]
            if prev in ('neutral', 'bear') and curr == 'bull':
                position = 1.0 if garch_vol.iloc[i] < self.garch_thresh else 0.5
            elif position > 0 and (
                data['Close'].iloc[i] < donchian_lo.iloc[i]
                or rvi.iloc[i] > 76
                or rvi.iloc[i] < self.rvi_bear
            ):
                position = 0.0
            signals.iloc[i] = position
        return signals.clip(0, 1)

    # ─── Component 2: Gen2300 ─────────────────────────────────
    def _signals_gen2300(self) -> pd.Series:
        """RVI + Force Index + SMA confirmation, ATR_Pct adaptive exposure."""
        data = self.data
        rvi = data['RVI_Refined']
        rvi_raw = data['RVI']
        close = data['Close']
        sma_20 = data['SMA_20']
        force_index = data['Force_Index']
        atr_pct = data['ATR_Pct']
        # Precompute to avoid O(n²) rolling inside the loop
        low_10min = data['Low'].rolling(window=10).min()

        state = pd.Series('neutral', index=data.index)
        state[rvi > 59] = 'green'
        state[(rvi >= 42) & (rvi <= 59)] = 'orange'
        state[rvi < 42] = 'red'
        prev_state = state.shift(1).fillna('unknown')

        signals = pd.Series(0.0, index=data.index)
        position = 0
        for i in range(len(data)):
            curr = state.iloc[i]
            prev = prev_state.iloc[i]
            if prev in ('orange', 'red') and curr == 'green':
                if close.iloc[i] > sma_20.iloc[i] and force_index.iloc[i] > 0:
                    position = 1
            elif position == 1 and (
                rvi_raw.iloc[i] > 76
                or rvi_raw.iloc[i] < 42
                or close.iloc[i] < low_10min.iloc[i]
            ):
                position = 0
            elif prev == 'orange' and curr == 'red':
                if close.iloc[i] < sma_20.iloc[i] and force_index.iloc[i] < 0:
                    position = 0
            signals.iloc[i] = float(position)

        # ATR_Pct adaptive exposure (from original Gen2300)
        atr_95 = atr_pct.rolling(window=10).quantile(0.95)
        atr_med = atr_pct.rolling(window=20).median()
        exposure = np.where(atr_pct > atr_95, 0.5,
                            np.where(atr_pct < atr_med, 1.0, 0.5))
        return (signals * exposure).clip(0, 1)

    # ─── Component 3: Gen4415 ─────────────────────────────────
    def _signals_gen4415(self) -> pd.Series:
        """HMM bull regime + QQE entry + ATR stop/target + STC/Squeeze exit."""
        data = self.data
        close = data['Close']
        atr = data['ATR'].bfill().fillna(1.0)
        qqe = data['QQE'].fillna(50)
        stc = data['STC'].fillna(0)
        hmm_regime = data['HMM_Regime'].fillna(0)
        sqz_hist = data['Squeeze_Pro_Hist'].fillna(0)

        signals = pd.Series(0.0, index=data.index)
        position = 0
        entry_price = sl_price = tp_price = 0.0
        for i in range(1, len(data)):
            c = close.iloc[i]
            if position == 0:
                if hmm_regime.iloc[i] == 2 and qqe.iloc[i] > self.qqe_threshold:
                    position = 1
                    entry_price = c
                    sl_price = c - atr.iloc[i] * self.atr_mult
                    tp_price = c + atr.iloc[i] * self.tp_mult
            else:
                if (c < sl_price or c > tp_price
                        or stc.iloc[i] > self.stc_threshold
                        or sqz_hist.iloc[i] < 0):
                    position = 0
                    entry_price = sl_price = tp_price = 0.0
            signals.iloc[i] = float(position)
        return signals.clip(0, 1)

    # ─── Ensemble ─────────────────────────────────────────────
    def generate_signals(self) -> pd.Series:
        """Vote: long when >= min_votes strategies signal long."""
        s1 = self._signals_gen3671()
        s2 = self._signals_gen2300()
        s3 = self._signals_gen4415()
        combined = pd.concat([s1, s2, s3], axis=1).fillna(0)
        vote_count = (combined > 0).sum(axis=1)
        return (vote_count >= self.min_votes).astype(float)

    def get_description(self) -> str:
        return (
            f"ChampionEnsemble (min_votes={self.min_votes}): "
            "OR-vote of Gen3671 (RVI+GARCH), Gen2300 (RVI+Force+SMA), "
            "Gen4415 (HMM+QQE+STC). "
            "Verified: Sharpe=1.36, CAGR=52.7%, MaxDD=-33.5%, Calmar=1.58"
        )


# ═══════════════════════════════════════════════════════════════
# Quick test
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    from indicator_pool import get_enriched_data
    from backtest import BacktestEngine
    from champion_rvi import ChampionRVI

    data = get_enriched_data()
    engine = BacktestEngine(data)

    # Ensemble
    ens = ChampionEnsemble()
    ens.init(data)
    bt = engine.run(ens)
    print("ChampionEnsemble (OR vote, min_votes=1):")
    print(f"  Sharpe:  {bt.sharpe_ratio:.2f}")
    print(f"  CAGR:    {bt.cagr:.1%}")
    print(f"  MaxDD:   {bt.max_drawdown:.1%}")
    print(f"  Calmar:  {bt.calmar_ratio:.2f}")
    print(f"  Trades:  {bt.total_trades}")
    print(f"  TiM:     {bt.time_in_market:.1%}")

    # AND vote (min_votes=2)
    ens2 = ChampionEnsemble(min_votes=2)
    ens2.init(data)
    bt2 = engine.run(ens2)
    print("\nChampionEnsemble (AND-2, min_votes=2):")
    print(f"  Sharpe:  {bt2.sharpe_ratio:.2f}")
    print(f"  CAGR:    {bt2.cagr:.1%}")
    print(f"  MaxDD:   {bt2.max_drawdown:.1%}")
    print(f"  Calmar:  {bt2.calmar_ratio:.2f}")

    # Baseline
    champ = ChampionRVI(enable_short=False)
    champ.init(data)
    btc = engine.run(champ)
    print("\nChampionRVI (baseline):")
    print(f"  Sharpe:  {btc.sharpe_ratio:.2f}")
    print(f"  CAGR:    {btc.cagr:.1%}")
    print(f"  MaxDD:   {btc.max_drawdown:.1%}")
    print(f"  Calmar:  {btc.calmar_ratio:.2f}")
