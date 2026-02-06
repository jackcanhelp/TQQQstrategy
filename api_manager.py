"""
API Key & Model Manager
========================
管理多組 Gemini API Key + 多模型自動切換，最大化免費配額使用。

Features:
- 8 組 API Key 輪換
- 多模型 failover (gemini-2.5-flash-lite → gemini-2.0-flash → gemini-1.5-flash)
- 智能等待與重試
"""

import os
import time
import re
from datetime import datetime, timedelta
from typing import Optional, List, Tuple, Dict
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()


# 模型優先順序（從最優先到備用）
MODELS = [
    "gemini-2.5-flash-lite",  # 主要模型
    "gemini-2.0-flash",       # 備用模型 1
    "gemini-2.0-flash-lite",  # 備用模型 2 (1.5-flash 已下架)
]


class APIKeyManager:
    """
    管理多組 API Key + 多模型，遇到配額限制自動切換。

    策略：
    1. 先嘗試當前 Key + 當前 Model
    2. 如果 429 錯誤 → 切換到下一個 Key
    3. 如果所有 Key 對當前 Model 都失敗 → 切換到下一個 Model
    4. 如果所有組合都失敗 → 等待後重試
    """

    def __init__(self, keys: List[str] = None, models: List[str] = None):
        """
        初始化 API Key + Model 管理器。
        """
        if keys is None:
            keys = self._load_keys_from_env()

        self.keys = [k for k in keys if k]
        self.models = models or MODELS.copy()

        self.current_key_index = 0
        self.current_model_index = 0

        # 追蹤每個 (key, model) 組合的狀態
        self.combo_status: Dict[Tuple[str, str], Dict] = {}
        for key in self.keys:
            for model in self.models:
                self.combo_status[(key, model)] = {
                    'blocked_until': None,
                    'fail_count': 0,
                    'success_count': 0
                }

        # 全局統計
        self.total_requests = 0
        self.total_successes = 0
        self.model_switches = 0
        self.key_switches = 0

        print(f"🔑 API Manager 初始化:")
        print(f"   {len(self.keys)} 組 API Key")
        print(f"   {len(self.models)} 個模型: {', '.join(self.models)}")
        print(f"   總共 {len(self.keys) * len(self.models)} 種組合可用")

        self._configure_current()

    def _load_keys_from_env(self) -> List[str]:
        """從環境變數讀取所有 API Key。"""
        keys = []

        main_key = os.getenv('GOOGLE_API_KEY')
        if main_key:
            keys.append(main_key)

        for i in range(1, 20):
            key = os.getenv(f'GOOGLE_API_KEY_{i}')
            if key:
                keys.append(key)

        return keys

    def _configure_current(self):
        """設定當前的 Key。"""
        if not self.keys:
            raise ValueError("沒有可用的 API Key！")

        current_key = self.keys[self.current_key_index]
        genai.configure(api_key=current_key)

    @property
    def current_key(self) -> str:
        return self.keys[self.current_key_index]

    @property
    def current_model(self) -> str:
        return self.models[self.current_model_index]

    def _rotate_key(self) -> bool:
        """
        切換到下一個 Key。

        Returns:
            True 如果成功切換，False 如果已經輪換一圈
        """
        old_index = self.current_key_index
        self.current_key_index = (self.current_key_index + 1) % len(self.keys)
        self.key_switches += 1

        if self.current_key_index == 0:
            # 已經輪換一圈
            return False

        self._configure_current()
        print(f"   🔄 切換 Key: #{old_index + 1} → #{self.current_key_index + 1}")
        return True

    def _rotate_model(self) -> bool:
        """
        切換到下一個 Model。

        Returns:
            True 如果成功切換，False 如果已經嘗試所有模型
        """
        old_model = self.current_model
        self.current_model_index = (self.current_model_index + 1) % len(self.models)
        self.model_switches += 1

        if self.current_model_index == 0:
            # 已經嘗試所有模型
            return False

        print(f"   🔄 切換模型: {old_model} → {self.current_model}")
        return True

    def _get_combo_status(self, key: str = None, model: str = None) -> Dict:
        """取得指定組合的狀態。"""
        key = key or self.current_key
        model = model or self.current_model
        return self.combo_status.get((key, model), {})

    def _mark_combo_failed(self, key: str = None, model: str = None, wait_seconds: int = 30):
        """標記組合失敗。"""
        key = key or self.current_key
        model = model or self.current_model
        combo = (key, model)

        if combo in self.combo_status:
            self.combo_status[combo]['fail_count'] += 1
            self.combo_status[combo]['blocked_until'] = datetime.now() + timedelta(seconds=wait_seconds)

    def _mark_combo_success(self, key: str = None, model: str = None):
        """標記組合成功。"""
        key = key or self.current_key
        model = model or self.current_model
        combo = (key, model)

        if combo in self.combo_status:
            self.combo_status[combo]['success_count'] += 1
            self.combo_status[combo]['blocked_until'] = None

        self.total_successes += 1

    def _is_combo_available(self, key: str, model: str) -> bool:
        """檢查組合是否可用。"""
        status = self.combo_status.get((key, model), {})
        blocked_until = status.get('blocked_until')

        if blocked_until is None:
            return True

        return datetime.now() >= blocked_until

    def _find_available_combo(self) -> Optional[Tuple[str, str]]:
        """
        尋找一個可用的 (key, model) 組合。

        Returns:
            (key, model) tuple 或 None
        """
        # 首先嘗試當前模型的所有 Key
        for i in range(len(self.keys)):
            key_idx = (self.current_key_index + i) % len(self.keys)
            key = self.keys[key_idx]

            if self._is_combo_available(key, self.current_model):
                self.current_key_index = key_idx
                self._configure_current()
                return (key, self.current_model)

        # 當前模型所有 Key 都不可用，嘗試其他模型
        for m in range(1, len(self.models)):
            model_idx = (self.current_model_index + m) % len(self.models)
            model = self.models[model_idx]

            for i in range(len(self.keys)):
                key_idx = (self.current_key_index + i) % len(self.keys)
                key = self.keys[key_idx]

                if self._is_combo_available(key, model):
                    self.current_key_index = key_idx
                    self.current_model_index = model_idx
                    self._configure_current()
                    print(f"   🔄 切換到: Key #{key_idx + 1} + {model}")
                    return (key, model)

        return None

    def _get_min_wait_time(self) -> int:
        """取得最短等待時間（秒）。"""
        now = datetime.now()
        min_wait = float('inf')

        for combo, status in self.combo_status.items():
            blocked = status.get('blocked_until')
            if blocked and blocked > now:
                wait = (blocked - now).total_seconds()
                min_wait = min(min_wait, wait)

        return int(min_wait) if min_wait != float('inf') else 0

    def generate_with_failover(
        self,
        prompt: str,
        preferred_model: str = None,
        max_retries: int = None
    ) -> Optional[str]:
        """
        使用 Key + Model failover 機制生成內容。

        策略：
        1. 嘗試當前 Key + 當前 Model
        2. 429 → 切換 Key
        3. 所有 Key 失敗 → 切換 Model
        4. 所有組合失敗 → 等待後重試

        Args:
            prompt: 提示詞
            preferred_model: 偏好的模型（可選）
            max_retries: 最大重試次數

        Returns:
            生成的文本，失敗返回 None
        """
        if preferred_model and preferred_model in self.models:
            self.current_model_index = self.models.index(preferred_model)

        if max_retries is None:
            max_retries = len(self.keys) * len(self.models) * 2

        self.total_requests += 1
        attempts = 0
        keys_tried_for_model = set()

        while attempts < max_retries:
            attempts += 1

            # 尋找可用組合
            combo = self._find_available_combo()

            if combo is None:
                # 所有組合都被封鎖
                wait_time = self._get_min_wait_time()
                if wait_time > 0:
                    print(f"   ⏳ 所有組合被限制，等待 {wait_time} 秒...")
                    time.sleep(min(wait_time + 1, 60))
                    # 重置追蹤
                    keys_tried_for_model.clear()
                    continue
                else:
                    break

            key, model = combo

            try:
                genai.configure(api_key=key)
                gemini_model = genai.GenerativeModel(model)
                response = gemini_model.generate_content(prompt)

                self._mark_combo_success(key, model)
                return response.text

            except Exception as e:
                error_msg = str(e)

                if '429' in error_msg:
                    # 配額超限
                    wait_time = self._parse_wait_time(error_msg)
                    self._mark_combo_failed(key, model, wait_time)

                    keys_tried_for_model.add(key)

                    # 檢查是否所有 Key 對當前模型都失敗
                    if len(keys_tried_for_model) >= len(self.keys):
                        print(f"   ⚠️ {model} 所有 Key 都被限制，切換模型...")
                        keys_tried_for_model.clear()
                        if not self._rotate_model():
                            # 已嘗試所有模型
                            wait_time = self._get_min_wait_time()
                            if wait_time > 0:
                                print(f"   ⏳ 所有模型被限制，等待 {wait_time} 秒...")
                                time.sleep(min(wait_time + 1, 60))
                    else:
                        self._rotate_key()

                elif '404' in error_msg:
                    # 模型不存在
                    print(f"   ❌ 模型 {model} 不可用，切換...")
                    # 永久封鎖這個模型
                    for k in self.keys:
                        self._mark_combo_failed(k, model, 86400)  # 24 小時
                    self._rotate_model()

                elif '503' in error_msg or 'ServiceUnavailable' in error_msg:
                    # 服務暫時不可用
                    print(f"   ⚠️ 服務暫時不可用，切換...")
                    self._mark_combo_failed(key, model, 10)
                    self._rotate_key()

                else:
                    # 其他錯誤
                    print(f"   ⚠️ API 錯誤: {error_msg[:60]}")
                    self._mark_combo_failed(key, model, 5)
                    self._rotate_key()

        print(f"   ❌ 所有 {max_retries} 次嘗試都失敗")
        return None

    def _parse_wait_time(self, error_msg: str) -> int:
        """從錯誤訊息解析等待時間。"""
        match = re.search(r'(\d+)\.?\d*\s*s', error_msg.lower())
        if match:
            return int(float(match.group(1))) + 1
        return 30

    def get_status(self) -> str:
        """取得詳細狀態報告。"""
        now = datetime.now()
        lines = [
            "🔑 API Manager 狀態:",
            f"   總請求: {self.total_requests} | 成功: {self.total_successes}",
            f"   Key 切換: {self.key_switches} | Model 切換: {self.model_switches}",
            "",
            f"   當前: Key #{self.current_key_index + 1} + {self.current_model}",
            ""
        ]

        # 統計每個模型的可用 Key 數
        for model in self.models:
            available = sum(
                1 for key in self.keys
                if self._is_combo_available(key, model)
            )
            lines.append(f"   {model}: {available}/{len(self.keys)} Keys 可用")

        return "\n".join(lines)

    # 保持向後兼容
    def generate_with_retry(self, prompt: str, model_name: str = None, max_retries: int = None) -> Optional[str]:
        """向後兼容的 API。"""
        return self.generate_with_failover(prompt, model_name, max_retries)


# 全域實例
_api_manager = None


def get_api_manager() -> APIKeyManager:
    """取得全域 API Manager 實例。"""
    global _api_manager
    if _api_manager is None:
        _api_manager = APIKeyManager()
    return _api_manager
