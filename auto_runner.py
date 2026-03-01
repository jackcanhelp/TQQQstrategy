#!/usr/bin/env python3
"""
Auto Runner - 自動迭代系統
===========================
持續運行 AI 策略進化，每 N 次迭代發送報告。
"""

import os
import sys
import json
import time
import argparse
import subprocess
from datetime import datetime
from pathlib import Path

import yfinance as yf
import pandas as pd
from dotenv import load_dotenv

from strategy_base import BuyAndHold, SimpleSMA
from backtest import BacktestEngine
from researcher import StrategyGenerator, StrategySandbox
from validator import StrategyValidator
from report_generator import ReportGenerator
from api_manager import get_api_manager

load_dotenv()


class AutoRunner:
    """
    自動迭代引擎，持續進化策略並定期報告。
    """

    def __init__(
        self,
        report_every: int = 50,
        target_sharpe: float = 2.0,
        max_total_iterations: int = None,  # None = 無限迭代
        notification_method: str = 'file',  # 'file', 'telegram', 'email'
        notification_config: dict = None
    ):
        self.report_every = report_every
        self.target_sharpe = target_sharpe
        self.max_total_iterations = max_total_iterations
        self.notification_method = notification_method
        self.notification_config = notification_config or {}

        self.generator = StrategyGenerator()
        self.sandbox = StrategySandbox()
        self.reporter = ReportGenerator()

        # Load data once
        self.data = self._load_data()
        self.engine = BacktestEngine(self.data)

        # Track session stats
        self.session_start = datetime.now()
        self.session_iterations = 0
        self.session_successes = 0

        # API failure consecutive counter (initialized here to avoid AttributeError)
        self._consec_api_fail = 0

        # Track best composite for milestone detection
        self._last_committed_best = 0.0

    def _load_data(self) -> pd.DataFrame:
        """Load or download TQQQ data."""
        cache_file = Path("TQQQ_data.pkl")

        if cache_file.exists():
            data = pd.read_pickle(cache_file)
            print(f"📊 Loaded cached data: {len(data)} rows")
            return data

        print("📥 Downloading TQQQ data...")
        ticker = yf.Ticker("TQQQ")
        data = ticker.history(period="max", auto_adjust=True)
        data = data[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
        data.to_pickle(cache_file)
        print(f"✅ Downloaded {len(data)} rows")
        return data

    def run_single_iteration(self) -> dict:
        """
        Run a single evolution iteration.

        Returns:
            dict with iteration results
        """
        strategy_id = self.generator.get_next_strategy_id()
        result = {
            'id': strategy_id,
            'success': False,
            'sharpe': 0.0,
            'cagr': 0.0,
            'max_dd': 0.0,
            'name': f'Strategy_Gen{strategy_id}',
            'error': None
        }
        idea = "N/A"  # initialize before try so except block can use it

        try:
            # Generate idea
            idea = self.generator.generate_strategy_idea()

            # Generate code
            code, file_path = self.generator.generate_strategy_code(idea, strategy_id)

            # Validate code
            is_valid, warnings = StrategyValidator.validate_code(code)
            if not is_valid:
                result['error'] = 'Look-ahead bias in code'
                self._record_failure(strategy_id, idea, result['error'])
                return result

            # Load and test
            class_name = f"Strategy_Gen{strategy_id}"
            strategy = self.sandbox.load_strategy(file_path, class_name)
            success, error = self.sandbox.test_strategy(strategy, self.data)

            if not success:
                # Try to fix once
                code, file_path = self.generator.fix_strategy_code(code, error, strategy_id)
                strategy = self.sandbox.load_strategy(file_path, class_name)
                success, error = self.sandbox.test_strategy(strategy, self.data)

            if not success:
                result['error'] = error[:100]
                self._record_failure(strategy_id, idea, result['error'])
                return result

            # Run backtest
            bt_result = self.engine.run(strategy)

            # Validate results
            results_valid, _ = StrategyValidator.validate_backtest_results(bt_result)
            if not results_valid:
                result['error'] = 'Unrealistic results'
                self._record_failure(strategy_id, idea, result['error'])
                return result

            # Success!
            result['success'] = True
            result['sharpe'] = bt_result.sharpe_ratio
            result['calmar'] = bt_result.calmar_ratio  # 主要指標
            result['cagr'] = bt_result.cagr
            result['max_dd'] = bt_result.max_drawdown
            result['idea'] = idea[:200]

            self.generator.record_result(
                strategy_id=strategy_id,
                strategy_name=class_name,
                idea=idea,
                sharpe=bt_result.sharpe_ratio,
                calmar=bt_result.calmar_ratio,
                cagr=bt_result.cagr,
                max_dd=bt_result.max_drawdown,
                failure_analysis=bt_result.get_failure_analysis(),
                success=True
            )

            self.session_successes += 1

        except Exception as e:
            result['error'] = str(e)[:100]
            # Always record so total_iterations increments — prevents ID from being stuck
            self._record_failure(strategy_id, idea, result['error'])

        self.session_iterations += 1
        return result

    def _record_failure(self, strategy_id: int, idea: str, error: str):
        """Record a failed strategy."""
        self.generator.record_result(
            strategy_id=strategy_id,
            strategy_name=f"Strategy_Gen{strategy_id}",
            idea=idea if idea else "N/A",
            sharpe=0.0,
            cagr=0.0,
            max_dd=0.0,
            failure_analysis=error,
            success=False
        )

    def generate_report(self) -> str:
        """Generate a progress report."""
        history = self.generator.history
        total = history['total_iterations']
        best_sharpe = history['best_sharpe']
        best_strategy = history['best_strategy']

        # Get recent strategies
        recent = history['strategies'][-10:] if history['strategies'] else []

        # Calculate stats
        successful = [s for s in history['strategies'] if s.get('success', False)]
        success_rate = len(successful) / total * 100 if total > 0 else 0

        # Top 5 strategies (ranked by composite score, filter out "do nothing" strategies)
        rankable = [s for s in successful if s.get('sharpe', 0) > 0 and s.get('cagr', 0) > 0.05]
        top5 = sorted(rankable, key=lambda x: x.get('composite', x.get('calmar', 0)), reverse=True)[:5]

        best_composite = history.get('best_composite', history.get('best_calmar', best_sharpe))
        report = f"""
═══════════════════════════════════════════════════════════════
📊 TQQQ 策略進化報告
   生成時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
═══════════════════════════════════════════════════════════════

📈 總體統計
───────────────────────────────────────────────────────────────
   總迭代次數: {total}
   成功策略數: {len(successful)} ({success_rate:.1f}%)
   最佳 Composite: {best_composite:.4f}
   最佳策略: {best_strategy}

   本次運行: {self.session_iterations} 次迭代
   本次成功: {self.session_successes} 個策略
   運行時長: {datetime.now() - self.session_start}

🏆 Top 5 策略
───────────────────────────────────────────────────────────────"""

        for i, s in enumerate(top5, 1):
            cs = s.get('composite', s.get('calmar', 0))
            report += f"""
   #{i} {s['name']}
       Comp: {cs:.4f} | Sharpe: {s['sharpe']:.2f} | CAGR: {s['cagr']:.1%} | MaxDD: {s['max_dd']:.1%}"""
            if s.get('test_sharpe'):
                report += f" | OOS Sharpe: {s['test_sharpe']:.2f}"

        report += f"""

📝 最近 10 次迭代
───────────────────────────────────────────────────────────────"""

        for s in recent[-10:]:
            status = "✅" if s.get('success') else "❌"
            if s.get('success'):
                info = f"Calmar:{s.get('calmar',0):.2f} Sharpe:{s['sharpe']:.2f} CAGR:{s['cagr']:.1%} MaxDD:{s['max_dd']:.1%}"
            else:
                info = f"Error: {s.get('failure_analysis', 'Unknown')[:30]}"
            report += f"""
   {status} {s['name']}: {info}"""

        # API Key 狀態
        try:
            api_manager = get_api_manager()
            api_status = api_manager.get_status()
            report += f"""

{api_status}"""
        except:
            pass

        report += """

═══════════════════════════════════════════════════════════════
"""
        return report

    def send_notification(self, report: str):
        """Send report via configured method."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        if self.notification_method == 'file':
            # Save to reports directory
            reports_dir = Path('reports')
            reports_dir.mkdir(exist_ok=True)
            report_file = reports_dir / f'report_{timestamp}.txt'
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"📄 Report saved to: {report_file}")

        elif self.notification_method == 'telegram':
            self._send_telegram(report)

        elif self.notification_method == 'email':
            self._send_email(report)

        # Always save latest report
        with open('latest_report.txt', 'w', encoding='utf-8') as f:
            f.write(report)

    def _telegram_post(self, bot_token: str, chat_id: str, text: str) -> bool:
        """Low-level Telegram message POST. Returns True on success."""
        import requests
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        resp = requests.post(url, data={'chat_id': chat_id, 'text': text}, timeout=15)
        return resp.status_code == 200 and resp.json().get('ok', False)

    def _send_telegram(self, report: str):
        """Send report via Telegram, splitting into pages if > 4000 chars."""
        try:
            import requests
            bot_token = self.notification_config.get('telegram_bot_token')
            chat_id = self.notification_config.get('telegram_chat_id')

            if not bot_token or not chat_id:
                print("⚠️ Telegram not configured")
                return

            # Split on line boundaries to avoid breaking tables/code blocks
            PAGE_SIZE = 3800  # leave room for page header
            lines = report.split('\n')
            pages = []
            current = []
            current_len = 0
            for line in lines:
                if current_len + len(line) + 1 > PAGE_SIZE and current:
                    pages.append('\n'.join(current))
                    current = [line]
                    current_len = len(line) + 1
                else:
                    current.append(line)
                    current_len += len(line) + 1
            if current:
                pages.append('\n'.join(current))

            total = len(pages)
            for i, page in enumerate(pages, 1):
                header = f"[{i}/{total}] " if total > 1 else ""
                ok = self._telegram_post(bot_token, chat_id, header + page)
                if not ok:
                    print(f"⚠️ Telegram page {i}/{total} 發送失敗")
                else:
                    print(f"📱 Telegram [{i}/{total}] sent")
                if total > 1 and i < total:
                    time.sleep(1)  # avoid flood limits between pages

        except Exception as e:
            print(f"⚠️ Telegram send failed: {e}")

    def _send_telegram_alert(self, message: str):
        """Send an immediate critical alert via Telegram (short, high-priority)."""
        try:
            import requests
            bot_token = self.notification_config.get('telegram_bot_token')
            chat_id = self.notification_config.get('telegram_chat_id')
            if not bot_token or not chat_id:
                return
            self._telegram_post(bot_token, chat_id, f"🚨 ALERT [{datetime.now().strftime('%H:%M:%S')}]\n{message}")
        except Exception as e:
            print(f"⚠️ Telegram alert failed: {e}")

    def _git_commit(self, message: str, files: list = None):
        """
        Commit specified files (or all tracked changes) to git.
        Silently skips if git is not available or nothing to commit.
        """
        try:
            repo_dir = Path(__file__).parent
            if files:
                for f in files:
                    subprocess.run(['git', 'add', str(f)], cwd=repo_dir,
                                   capture_output=True, timeout=30)
            else:
                subprocess.run(['git', 'add', '-u'], cwd=repo_dir,
                               capture_output=True, timeout=30)

            result = subprocess.run(
                ['git', 'commit', '-m', message],
                cwd=repo_dir, capture_output=True, text=True, timeout=30
            )
            if result.returncode == 0:
                print(f"   📝 Git committed: {message[:60]}")
            # returncode 1 = "nothing to commit" — silently ignore
        except Exception as e:
            print(f"   ⚠️ Git commit failed: {e}")

    def _git_push(self):
        """Push to remote. Requires GITHUB_TOKEN in env."""
        try:
            repo_dir = Path(__file__).parent
            token = os.getenv('GITHUB_TOKEN', '')
            # Read current remote URL and inject token if HTTPS
            url_result = subprocess.run(
                ['git', 'remote', 'get-url', 'origin'],
                cwd=repo_dir, capture_output=True, text=True, timeout=10
            )
            remote_url = url_result.stdout.strip()
            if token and remote_url.startswith('https://') and '@' not in remote_url:
                # GitHub requires "username:token" format, not just "token"
                authed = remote_url.replace('https://', f'https://jackcanhelp:{token}@')
                push_result = subprocess.run(
                    ['git', 'push', authed, 'HEAD'],
                    cwd=repo_dir, capture_output=True, text=True, timeout=60
                )
            else:
                push_result = subprocess.run(
                    ['git', 'push'],
                    cwd=repo_dir, capture_output=True, text=True, timeout=60
                )
            if push_result.returncode == 0:
                print("   ☁️ Git pushed to remote")
            else:
                print(f"   ⚠️ Git push failed: {push_result.stderr[:80]}")
        except Exception as e:
            print(f"   ⚠️ Git push error: {e}")

    def _send_email(self, report: str):
        """Send report via email."""
        try:
            import smtplib
            from email.mime.text import MIMEText

            smtp_server = self.notification_config.get('smtp_server')
            smtp_port = self.notification_config.get('smtp_port', 587)
            email_from = self.notification_config.get('email_from')
            email_to = self.notification_config.get('email_to')
            email_password = self.notification_config.get('email_password')

            if not all([smtp_server, email_from, email_to, email_password]):
                print("⚠️ Email not configured")
                return

            msg = MIMEText(report)
            msg['Subject'] = f'TQQQ Strategy Report - {datetime.now().strftime("%Y-%m-%d %H:%M")}'
            msg['From'] = email_from
            msg['To'] = email_to

            with smtplib.SMTP(smtp_server, smtp_port) as server:
                server.starttls()
                server.login(email_from, email_password)
                server.sendmail(email_from, email_to, msg.as_string())

            print("📧 Report sent via email")
        except Exception as e:
            print(f"⚠️ Email send failed: {e}")

    def run(self):
        """
        Main loop - run until target Sharpe or max iterations.
        """
        print("=" * 60)
        print("🚀 TQQQ Auto Runner 啟動")
        print(f"   報告頻率: 每 {self.report_every} 次迭代")
        print(f"   目標 Sharpe: {self.target_sharpe}")
        print(f"   最大迭代: {'無限' if self.max_total_iterations is None else self.max_total_iterations}")
        print("=" * 60)
        self._send_telegram_alert(
            f"🚀 TQQQ Auto Runner 啟動\n"
            f"目標 Sharpe: {self.target_sharpe}\n"
            f"最大迭代: {self.max_total_iterations}\n"
            f"報告頻率: 每 {self.report_every} 次"
        )

        iteration = 0
        while True:
            iteration += 1
            current_total = self.generator.history['total_iterations']

            if self.max_total_iterations is not None and current_total >= self.max_total_iterations:
                print(f"\n🏁 達到最大迭代次數 ({self.max_total_iterations})")
                break

            # Run iteration
            print(f"\n[{iteration}] Running iteration {current_total + 1}...", end=" ")
            result = self.run_single_iteration()

            if result['success']:
                sharpe = result['sharpe']
                calmar = result.get('calmar', 0.0)
                composite = result.get('composite', calmar)
                print(f"✅ Sharpe: {sharpe:.2f} Calmar: {calmar:.2f}")
                self._consec_api_fail = 0

                # Commit new strategy file to git
                strategy_file = Path('generated_strategies') / f"{result['name']}.py"
                if strategy_file.exists():
                    self._git_commit(
                        f"[auto] {result['name']}: Sharpe={sharpe:.2f} Calmar={calmar:.2f} CAGR={result['cagr']:.1%}",
                        files=[strategy_file, 'history_of_thoughts.json']
                    )

                # Milestone: new best composite → immediate alert + push
                if composite > self._last_committed_best + 0.05:
                    self._last_committed_best = composite
                    alert_msg = (
                        f"🏆 新最佳策略！{result['name']}\n"
                        f"Composite: {composite:.4f}\n"
                        f"Sharpe: {sharpe:.2f} | Calmar: {calmar:.2f}\n"
                        f"CAGR: {result['cagr']:.1%} | MaxDD: {result['max_dd']:.1%}"
                    )
                    self._send_telegram_alert(alert_msg)
                    self._git_push()

                # Check target Sharpe
                if sharpe >= self.target_sharpe:
                    print(f"\n🎯 TARGET ACHIEVED! Sharpe {sharpe:.2f} >= {self.target_sharpe}")
                    report = self.generate_report()
                    print(report)
                    self.send_notification(report)
                    self._git_push()
                    break
            else:
                error_msg = result['error'][:50] if result['error'] else 'Failed'
                print(f"❌ {error_msg}")

                # API 全掛偵測（中英文錯誤訊息都涵蓋）
                api_down_keywords = ['都不可用', 'all apis', 'api failed', 'no api keys']
                err_lower = (result['error'] or '').lower()
                if any(kw in err_lower for kw in api_down_keywords):
                    self._consec_api_fail += 1
                    cooldown = min(60 * self._consec_api_fail, 300)
                    print(f"   ⏳ API 全部限流，冷卻 {cooldown} 秒... (連續 {self._consec_api_fail} 次)")
                    if self._consec_api_fail >= 3:
                        self._send_telegram_alert(
                            f"⚠️ API 已連續失敗 {self._consec_api_fail} 次，系統進入長冷卻。\n"
                            f"請檢查 API Keys 配額狀態。"
                        )
                    time.sleep(cooldown)
                    continue
                else:
                    self._consec_api_fail = 0

            # Periodic report
            if iteration % self.report_every == 0:
                print(f"\n📊 Generating report (iteration {iteration})...")
                report = self.generate_report()
                print(report)
                self.send_notification(report)
                # Push history to remote periodically
                self._git_commit(
                    f"[auto] 週期報告 iteration={iteration} successes={self.session_successes}",
                    files=['history_of_thoughts.json', 'latest_report.txt']
                )
                self._git_push()

            # Delay between iterations to pace API usage
            time.sleep(5)

        # Final report
        print("\n📊 Final Report:")
        report = self.generate_report()
        print(report)
        self.send_notification(report)


def main():
    parser = argparse.ArgumentParser(description='TQQQ Auto Runner')
    parser.add_argument('--report-every', type=int, default=50,
                        help='Report every N iterations (default: 50)')
    parser.add_argument('--target-sharpe', type=float, default=2.0,
                        help='Target Sharpe ratio to stop (default: 2.0)')
    parser.add_argument('--max-iterations', type=int, default=0,
                        help='Maximum total iterations (0 = infinite, default: infinite)')
    parser.add_argument('--notify', type=str, default='file',
                        choices=['file', 'telegram', 'email'],
                        help='Notification method (default: file)')

    args = parser.parse_args()

    # Load notification config from environment
    notification_config = {
        'telegram_bot_token': os.getenv('TELEGRAM_BOT_TOKEN'),
        'telegram_chat_id': os.getenv('TELEGRAM_CHAT_ID'),
        'smtp_server': os.getenv('SMTP_SERVER'),
        'smtp_port': int(os.getenv('SMTP_PORT', 587)),
        'email_from': os.getenv('EMAIL_FROM'),
        'email_to': os.getenv('EMAIL_TO'),
        'email_password': os.getenv('EMAIL_PASSWORD'),
    }

    runner = AutoRunner(
        report_every=args.report_every,
        target_sharpe=args.target_sharpe,
        max_total_iterations=args.max_iterations if args.max_iterations > 0 else None,
        notification_method=args.notify,
        notification_config=notification_config
    )

    try:
        runner.run()
    except KeyboardInterrupt:
        print("\n\n⏹️ Stopped by user")
        report = runner.generate_report()
        print(report)
        runner.send_notification(report)


if __name__ == "__main__":
    main()
