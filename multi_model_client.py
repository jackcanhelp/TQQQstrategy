"""
Multi-Model AI Decision Engine
==============================
使用 GitHub Models API 實現多模型決策引擎。
支援 GPT-4.1 → DeepSeek-V3 → Llama-4-Scout 層級式 failover。
"""

import os
import json
import time
import logging
from typing import Optional, Dict, Any
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

# 設置 logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TradingDecision:
    """標準化的交易決策輸出。"""
    signal: str  # "BUY", "SELL", "HOLD"
    confidence_score: float  # 0.0 - 1.0
    reasoning_summary: str
    model_used: str

    def to_dict(self) -> Dict:
        return {
            "signal": self.signal,
            "confidence_score": self.confidence_score,
            "reasoning_summary": self.reasoning_summary,
            "model_used": self.model_used
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


# ═══════════════════════════════════════════════════════════════
# 模型層級配置
# ═══════════════════════════════════════════════════════════════
MODEL_HIERARCHY = [
    {
        "name": "gpt-4.1",
        "model_id": "openai/gpt-4.1",  # GitHub Models 上最強 OpenAI 模型
        "role": "Lead Strategist",
        "description": "Logic-heavy and multi-step tasks",
        "timeout": 60,
        "max_tokens": 1000,
    },
    {
        "name": "DeepSeek-V3",
        "model_id": "deepseek/DeepSeek-V3-0324",
        "role": "Check & Balance / Failover",
        "description": "High-performance alternative for trend validation",
        "timeout": 45,
        "max_tokens": 1000,
    },
    {
        "name": "Llama-4-Scout-17B",
        "model_id": "meta/Llama-4-Scout-17B-16E-Instruct",
        "role": "Safety Net",
        "description": "Multi-document processing fallback",
        "timeout": 30,
        "max_tokens": 800,
    },
]


class MultiModelClient:
    """
    多模型 AI 決策引擎。

    使用 GitHub Models API，依序嘗試：
    1. GPT-4.1 (主要)
    2. DeepSeek-V3 (備用)
    3. Llama-4-Scout (最終備用)
    """

    GITHUB_MODELS_ENDPOINT = "https://models.github.ai/inference"

    def __init__(self):
        """初始化 Multi-Model Client。"""
        self.token = os.getenv("GITHUB_TOKEN")
        if not self.token:
            raise ValueError("GITHUB_TOKEN not found in environment variables")

        self.client = None
        self._init_client()

        # 統計
        self.stats = {
            "total_requests": 0,
            "model_usage": {m["name"]: 0 for m in MODEL_HIERARCHY},
            "failures": {m["name"]: 0 for m in MODEL_HIERARCHY},
        }

        logger.info("🤖 Multi-Model Client 初始化完成")
        logger.info(f"   模型層級: {' → '.join(m['name'] for m in MODEL_HIERARCHY)}")

    def _init_client(self):
        """初始化 OpenAI 相容客戶端。"""
        try:
            from openai import OpenAI
            self.client = OpenAI(
                base_url=self.GITHUB_MODELS_ENDPOINT,
                api_key=self.token,
            )
            logger.info("✅ OpenAI SDK 客戶端初始化成功")
        except ImportError:
            logger.warning("⚠️ OpenAI SDK 未安裝，嘗試使用 Azure AI Inference SDK")
            try:
                from azure.ai.inference import ChatCompletionsClient
                from azure.core.credentials import AzureKeyCredential
                self.client = ChatCompletionsClient(
                    endpoint=self.GITHUB_MODELS_ENDPOINT,
                    credential=AzureKeyCredential(self.token),
                )
                logger.info("✅ Azure AI Inference SDK 客戶端初始化成功")
            except ImportError:
                raise ImportError(
                    "請安裝 openai 或 azure-ai-inference: "
                    "pip install openai 或 pip install azure-ai-inference"
                )

    def _build_trading_prompt(self, market_data: Dict) -> str:
        """建構交易決策的 prompt。"""
        return f"""You are an expert quantitative trading advisor for TQQQ (3x Leveraged Nasdaq ETF).

MARKET DATA:
{json.dumps(market_data, indent=2)}

ANALYSIS REQUIREMENTS:
1. Evaluate the current market regime (Bull/Bear/Sideways)
2. Assess volatility levels and VIX indicators
3. Consider trend strength and momentum
4. Factor in the 3x leverage decay risk of TQQQ

CRITICAL RULES FOR TQQQ:
- In high volatility (VIX > 25): Prefer HOLD or SELL
- In strong downtrends: SELL to avoid 3x losses
- Only BUY in confirmed uptrends with low volatility
- SURVIVAL is more important than PROFIT

OUTPUT FORMAT (STRICTLY JSON, no other text):
{{
    "signal": "BUY" or "SELL" or "HOLD",
    "confidence_score": 0.0 to 1.0,
    "reasoning_summary": "Concise 1-2 sentence explanation"
}}

Respond ONLY with the JSON object, no additional text."""

    def _parse_response(self, response_text: str, model_name: str) -> Optional[TradingDecision]:
        """解析模型回應為標準化決策。"""
        try:
            # 嘗試直接解析 JSON
            # 移除可能的 markdown 包裝
            text = response_text.strip()
            if text.startswith("```json"):
                text = text[7:]
            if text.startswith("```"):
                text = text[3:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()

            data = json.loads(text)

            # 驗證必要欄位
            signal = data.get("signal", "HOLD").upper()
            if signal not in ["BUY", "SELL", "HOLD"]:
                signal = "HOLD"

            confidence = float(data.get("confidence_score", 0.5))
            confidence = max(0.0, min(1.0, confidence))

            reasoning = data.get("reasoning_summary", "No reasoning provided")

            return TradingDecision(
                signal=signal,
                confidence_score=confidence,
                reasoning_summary=reasoning,
                model_used=model_name
            )

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning(f"⚠️ 無法解析 {model_name} 回應: {e}")
            return None

    def _call_model(self, model_config: Dict, prompt: str) -> Optional[str]:
        """呼叫單一模型。"""
        model_name = model_config["name"]
        model_id = model_config["model_id"]
        timeout = model_config["timeout"]
        max_tokens = model_config["max_tokens"]

        try:
            logger.info(f"🔄 嘗試呼叫 {model_name} ({model_config['role']})...")

            # 使用 OpenAI SDK 格式
            if hasattr(self.client, 'chat'):
                response = self.client.chat.completions.create(
                    model=model_id,
                    messages=[
                        {"role": "system", "content": "You are a quantitative trading expert. Always respond in valid JSON format."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=max_tokens,
                    temperature=0.3,  # 較低溫度以獲得一致性
                    timeout=timeout,
                )
                return response.choices[0].message.content
            else:
                # Azure SDK 格式
                from azure.ai.inference.models import SystemMessage, UserMessage
                response = self.client.complete(
                    model=model_id,
                    messages=[
                        SystemMessage(content="You are a quantitative trading expert. Always respond in valid JSON format."),
                        UserMessage(content=prompt)
                    ],
                    max_tokens=max_tokens,
                    temperature=0.3,
                )
                return response.choices[0].message.content

        except Exception as e:
            error_type = type(e).__name__
            logger.warning(f"❌ {model_name} 失敗 ({error_type}): {str(e)[:100]}")
            self.stats["failures"][model_name] += 1
            # Rate limit 時等待一下再嘗試下一個模型
            if 'RateLimit' in error_type or '429' in str(e):
                time.sleep(5)
            return None

    def get_trading_decision(self, market_data: Dict) -> TradingDecision:
        """
        取得交易決策，依序嘗試各模型。

        Args:
            market_data: 市場數據字典，包含價格、指標等

        Returns:
            TradingDecision 物件
        """
        self.stats["total_requests"] += 1
        prompt = self._build_trading_prompt(market_data)

        # 依序嘗試各模型
        for model_config in MODEL_HIERARCHY:
            model_name = model_config["name"]

            response_text = self._call_model(model_config, prompt)

            if response_text:
                decision = self._parse_response(response_text, model_name)
                if decision:
                    self.stats["model_usage"][model_name] += 1
                    logger.info(f"✅ {model_name} 成功: {decision.signal} (信心度: {decision.confidence_score:.2f})")
                    return decision

        # 所有模型都失敗，返回安全預設值
        logger.error("❌ 所有模型都失敗，返回安全預設值 (HOLD)")
        return TradingDecision(
            signal="HOLD",
            confidence_score=0.0,
            reasoning_summary="All models failed, defaulting to safe HOLD position",
            model_used="fallback"
        )

    def get_strategy_idea(self, context: str, indicators: str) -> Optional[str]:
        """
        使用多模型生成策略想法（整合到現有演化系統）。

        Args:
            context: 歷史策略上下文
            indicators: 必須使用的指標

        Returns:
            策略想法文字
        """
        prompt = f"""You are a Quantitative Research Director designing TQQQ trading strategies.

CONTEXT:
{context}

{indicators}

Generate a NEW trading strategy idea. Focus on:
1. Regime Filter (when to stay in cash)
2. Entry Signal (when to buy)
3. Exit Rules (when to sell)

Use ONLY backward-looking indicators. NO look-ahead bias.
Keep the response concise and actionable."""

        for model_config in MODEL_HIERARCHY:
            response = self._call_model(model_config, prompt)
            if response:
                logger.info(f"✅ 策略想法由 {model_config['name']} 生成")
                return response

        return None

    def generate(self, prompt: str) -> Optional[str]:
        """
        Public interface: 依序嘗試各模型，返回第一個成功的原始回應文字。
        供 researcher.py 等外部模組使用。
        """
        for model_config in MODEL_HIERARCHY:
            response = self._call_model(model_config, prompt)
            if response:
                logger.info(f"✅ GitHub Models ({model_config['name']}) 回應成功")
                return response
        return None

    def _call_model_chain(self, prompt: str) -> Optional[str]:
        """Deprecated alias for generate(). Use generate() instead."""
        return self.generate(prompt)

    def get_stats(self) -> str:
        """取得使用統計。"""
        lines = [
            "🤖 Multi-Model Client 統計:",
            f"   總請求數: {self.stats['total_requests']}",
            "",
            "   模型使用:",
        ]
        for name, count in self.stats["model_usage"].items():
            failures = self.stats["failures"][name]
            lines.append(f"      {name}: {count} 成功, {failures} 失敗")

        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
# 便捷函數
# ═══════════════════════════════════════════════════════════════

_client = None

def get_multi_model_client() -> MultiModelClient:
    """取得全域 MultiModelClient 實例。"""
    global _client
    if _client is None:
        _client = MultiModelClient()
    return _client


def get_trading_decision(market_data: Dict) -> TradingDecision:
    """
    便捷函數：取得交易決策。

    Args:
        market_data: 市場數據

    Returns:
        TradingDecision
    """
    client = get_multi_model_client()
    return client.get_trading_decision(market_data)


# ═══════════════════════════════════════════════════════════════
# 測試
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # 測試用市場數據
    test_market_data = {
        "date": "2024-01-15",
        "tqqq_price": 45.23,
        "tqqq_change_pct": -2.1,
        "qqq_price": 410.50,
        "vix": 18.5,
        "sma_50": 44.00,
        "sma_200": 42.50,
        "rsi_14": 45.2,
        "atr_14": 1.85,
        "trend": "neutral",
        "volume_ratio": 1.2
    }

    print("=" * 60)
    print("🧪 測試 Multi-Model Client")
    print("=" * 60)

    try:
        decision = get_trading_decision(test_market_data)
        print("\n📊 交易決策:")
        print(decision.to_json())

        client = get_multi_model_client()
        print("\n" + client.get_stats())

    except Exception as e:
        print(f"❌ 測試失敗: {e}")
