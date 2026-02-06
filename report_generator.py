"""
Report Generator
=================
Generates detailed reports for strategy evolution progress.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd
import matplotlib.pyplot as plt


class ReportGenerator:
    """
    Generates text and visual reports for strategy evolution.
    """

    def __init__(self, history_file: str = "history_of_thoughts.json"):
        self.history_file = Path(history_file)

    def load_history(self) -> Dict:
        """Load evolution history."""
        if self.history_file.exists():
            with open(self.history_file, 'r') as f:
                return json.load(f)
        return {"total_iterations": 0, "strategies": []}

    def generate_text_report(self) -> str:
        """Generate a comprehensive text report."""
        history = self.load_history()
        total = history.get('total_iterations', 0)
        strategies = history.get('strategies', [])
        best_sharpe = history.get('best_sharpe', 0)
        best_strategy = history.get('best_strategy', 'None')

        # Calculate statistics
        successful = [s for s in strategies if s.get('success', False)]
        success_rate = len(successful) / total * 100 if total > 0 else 0

        # Group by success/failure reasons
        failures = [s for s in strategies if not s.get('success', False)]
        failure_reasons = {}
        for f in failures:
            reason = f.get('failure_analysis', 'Unknown')[:50]
            failure_reasons[reason] = failure_reasons.get(reason, 0) + 1

        report = f"""
╔══════════════════════════════════════════════════════════════════╗
║           TQQQ STRATEGY EVOLUTION REPORT                         ║
║           Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}                        ║
╚══════════════════════════════════════════════════════════════════╝

┌─────────────────────────────────────────────────────────────────┐
│ OVERALL STATISTICS                                              │
├─────────────────────────────────────────────────────────────────┤
│ Total Iterations:     {total:>6}                                   │
│ Successful Strategies: {len(successful):>5} ({success_rate:>5.1f}%)                        │
│ Best Sharpe Ratio:    {best_sharpe:>6.2f}                                   │
│ Best Strategy:        {best_strategy:<20}              │
└─────────────────────────────────────────────────────────────────┘

"""

        if successful:
            # Top performers
            top10 = sorted(successful, key=lambda x: x.get('sharpe', 0), reverse=True)[:10]

            report += """┌─────────────────────────────────────────────────────────────────┐
│ TOP 10 STRATEGIES                                               │
├─────────────────────────────────────────────────────────────────┤
"""
            for i, s in enumerate(top10, 1):
                name = s.get('name', 'Unknown')[:20]
                sharpe = s.get('sharpe', 0)
                cagr = s.get('cagr', 0)
                max_dd = s.get('max_dd', 0)
                report += f"│ {i:>2}. {name:<20} Sharpe: {sharpe:>5.2f} CAGR: {cagr:>6.1%} DD: {max_dd:>6.1%} │\n"

            report += """└─────────────────────────────────────────────────────────────────┘

"""

        if failure_reasons:
            report += """┌─────────────────────────────────────────────────────────────────┐
│ COMMON FAILURE REASONS                                          │
├─────────────────────────────────────────────────────────────────┤
"""
            for reason, count in sorted(failure_reasons.items(), key=lambda x: -x[1])[:5]:
                report += f"│ {count:>4}x: {reason:<55} │\n"

            report += """└─────────────────────────────────────────────────────────────────┘
"""

        return report

    def generate_chart(self, output_file: str = "evolution_progress.png"):
        """Generate visual chart of evolution progress."""
        history = self.load_history()
        strategies = history.get('strategies', [])

        if not strategies:
            return

        # Extract data
        ids = []
        sharpes = []
        cagrs = []
        max_dds = []

        for s in strategies:
            if s.get('success', False):
                ids.append(s.get('id', len(ids)))
                sharpes.append(s.get('sharpe', 0))
                cagrs.append(s.get('cagr', 0) * 100)  # Convert to %
                max_dds.append(s.get('max_dd', 0) * 100)  # Convert to %

        if not ids:
            return

        # Create figure
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))

        # Sharpe ratio over time
        ax1 = axes[0]
        ax1.plot(ids, sharpes, 'b-o', markersize=4, alpha=0.7)
        ax1.axhline(y=max(sharpes), color='g', linestyle='--', label=f'Best: {max(sharpes):.2f}')
        ax1.set_ylabel('Sharpe Ratio')
        ax1.set_title('Strategy Evolution Progress')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # CAGR over time
        ax2 = axes[1]
        ax2.bar(ids, cagrs, color='green', alpha=0.6)
        ax2.set_ylabel('CAGR (%)')
        ax2.grid(True, alpha=0.3)

        # Max Drawdown over time
        ax3 = axes[2]
        ax3.bar(ids, max_dds, color='red', alpha=0.6)
        ax3.set_ylabel('Max Drawdown (%)')
        ax3.set_xlabel('Strategy ID')
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_file, dpi=150)
        plt.close()

        print(f"📊 Chart saved to: {output_file}")

    def save_report(self, filename: str = None):
        """Save report to file."""
        if filename is None:
            filename = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

        report = self.generate_text_report()

        with open(filename, 'w', encoding='utf-8') as f:
            f.write(report)

        print(f"📄 Report saved to: {filename}")
        return filename
