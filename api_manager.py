"""
API Key Manager
================
管理多組 Gemini API Key，自動輪換避免配額限制。
"""

import os
import time
from datetime import datetime, timedelta
from typing import Optional, List
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()


class APIKeyManager:
    """
    管理多組 API Key，遇到 429 錯誤自動切換。
    """

    def __init__(self, keys: List[str] = None):
        """
        初始化 API Key 管理器。

        Args:
            keys: API Key 列表，如果為 None 則從環境變數讀取
        """
        if keys is None:
            keys = self._load_keys_from_env()

        self.keys = [k for k in keys if k]  # 過濾空值
        self.current_index = 0
        self.key_status = {}  # {key: {'blocked_until': datetime, 'fail_count': int}}

        for key in self.keys:
            self.key_status[key] = {
                'blocked_until': None,
                'fail_count': 0,
                'success_count': 0
            }

        print(f"🔑 API Key Manager 初始化: {len(self.keys)} 組 Key 可用")

        # 初始化第一個可用的 Key
        self._configure_current_key()

    def _load_keys_from_env(self) -> List[str]:
        """從環境變數讀取所有 API Key。"""
        keys = []

        # 主 Key
        main_key = os.getenv('GOOGLE_API_KEY')
        if main_key:
            keys.append(main_key)

        # 額外的 Key (GOOGLE_API_KEY_1, GOOGLE_API_KEY_2, ...)
        for i in range(1, 20):
            key = os.getenv(f'GOOGLE_API_KEY_{i}')
            if key:
                keys.append(key)

        return keys

    def _configure_current_key(self):
        """設定當前使用的 API Key。"""
        if not self.keys:
            raise ValueError("沒有可用的 API Key！")

        current_key = self.keys[self.current_index]
        genai.configure(api_key=current_key)
        print(f"   使用 Key #{self.current_index + 1} ({current_key[:10]}...)")

    def get_model(self, model_name: str = "gemini-2.5-flash-lite"):
        """取得已設定好的模型。"""
        return genai.GenerativeModel(model_name)

    def get_available_key(self) -> Optional[str]:
        """
        取得一個可用的 API Key。

        Returns:
            可用的 Key，如果全部都被封鎖則返回 None
        """
        now = datetime.now()

        # 嘗試找到一個未被封鎖的 Key
        for _ in range(len(self.keys)):
            key = self.keys[self.current_index]
            status = self.key_status[key]

            # 檢查是否已解封
            if status['blocked_until'] is None or now >= status['blocked_until']:
                status['blocked_until'] = None  # 清除封鎖狀態
                return key

            # 切換到下一個 Key
            self._rotate_key()

        # 所有 Key 都被封鎖，返回等待時間最短的
        min_wait = None
        min_key = None
        for key, status in self.key_status.items():
            if status['blocked_until']:
                if min_wait is None or status['blocked_until'] < min_wait:
                    min_wait = status['blocked_until']
                    min_key = key

        if min_wait:
            wait_seconds = (min_wait - now).total_seconds()
            if wait_seconds > 0:
                print(f"⏳ 所有 Key 都被限制，等待 {wait_seconds:.0f} 秒...")
                time.sleep(min(wait_seconds + 1, 60))  # 最多等 60 秒

        return min_key

    def _rotate_key(self):
        """切換到下一個 API Key。"""
        self.current_index = (self.current_index + 1) % len(self.keys)
        self._configure_current_key()

    def mark_key_failed(self, key: str = None, wait_seconds: int = 30):
        """
        標記 Key 失敗（遇到 429 錯誤）。

        Args:
            key: 失敗的 Key，如果為 None 則使用當前 Key
            wait_seconds: 等待時間（秒）
        """
        if key is None:
            key = self.keys[self.current_index]

        status = self.key_status[key]
        status['fail_count'] += 1
        status['blocked_until'] = datetime.now() + timedelta(seconds=wait_seconds)

        print(f"⚠️ Key #{self.keys.index(key) + 1} 被限制，{wait_seconds}秒後重試")

        # 自動切換到下一個 Key
        self._rotate_key()

    def mark_key_success(self, key: str = None):
        """標記 Key 成功。"""
        if key is None:
            key = self.keys[self.current_index]

        status = self.key_status[key]
        status['success_count'] += 1
        status['blocked_until'] = None

    def generate_with_retry(
        self,
        prompt: str,
        model_name: str = "gemini-2.5-flash-lite",
        max_retries: int = None
    ) -> Optional[str]:
        """
        使用自動重試和 Key 輪換來生成內容。

        Args:
            prompt: 提示詞
            model_name: 模型名稱
            max_retries: 最大重試次數，None 表示嘗試所有 Key

        Returns:
            生成的文本，失敗返回 None
        """
        if max_retries is None:
            max_retries = len(self.keys) * 3  # 每個 Key 最多嘗試 3 次

        for attempt in range(max_retries):
            key = self.get_available_key()
            if key is None:
                print("❌ 所有 API Key 都不可用")
                return None

            try:
                # 確保使用正確的 Key
                genai.configure(api_key=key)
                model = genai.GenerativeModel(model_name)

                response = model.generate_content(prompt)
                self.mark_key_success(key)
                return response.text

            except Exception as e:
                error_msg = str(e)

                if '429' in error_msg:
                    # 配額超限，標記並切換
                    wait_time = 30
                    if 'retry' in error_msg.lower():
                        # 嘗試從錯誤訊息解析等待時間
                        import re
                        match = re.search(r'(\d+)\.?\d*s', error_msg)
                        if match:
                            wait_time = int(float(match.group(1))) + 1

                    self.mark_key_failed(key, wait_time)

                elif '404' in error_msg:
                    print(f"❌ 模型不存在: {model_name}")
                    return None

                else:
                    print(f"⚠️ API 錯誤: {error_msg[:50]}")
                    self.mark_key_failed(key, 10)

        return None

    def get_status(self) -> str:
        """取得所有 Key 的狀態報告。"""
        lines = ["🔑 API Key 狀態:"]
        now = datetime.now()

        for i, key in enumerate(self.keys):
            status = self.key_status[key]
            blocked = status['blocked_until']

            if blocked and blocked > now:
                remaining = (blocked - now).total_seconds()
                state = f"🔴 封鎖中 ({remaining:.0f}s)"
            else:
                state = "🟢 可用"

            lines.append(
                f"   #{i+1} ({key[:8]}...): {state} | "
                f"成功: {status['success_count']} | 失敗: {status['fail_count']}"
            )

        return "\n".join(lines)


# 全域實例
_api_manager = None

def get_api_manager() -> APIKeyManager:
    """取得全域 API Manager 實例。"""
    global _api_manager
    if _api_manager is None:
        _api_manager = APIKeyManager()
    return _api_manager
