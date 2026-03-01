"""
Groq Unified Client — Primary LLM Engine (5-Key Pool Edition)
===============================================================
5 Key pool-based allocation + task-based model selection.
Pool design: idea=K1,K2 | code=K3,K4 | fix=K5 + cross-pool overflow.
Failover: pool keys → overflow keys → all models × all keys.

Rate Limits (Free Tier, per key):
  llama-3.1-8b-instant       : 30 RPM, 14.4K RPD, 6K TPM, 500K TPD
  llama-3.3-70b-versatile    : 30 RPM, 1K RPD, 12K TPM, 100K TPD
  llama-4-scout-17b-16e      : 30 RPM, 1K RPD, 30K TPM, 500K TPD
  llama-4-maverick-17b-128e  : 30 RPM, 1K RPD, 6K TPM, 500K TPD
  kimi-k2-instruct           : 60 RPM, 1K RPD, 10K TPM, 300K TPD
  qwen3-32b                  : 60 RPM, 1K RPD, 6K TPM, 500K TPD

Usage:
    from groq_client import GroqClient
    client = GroqClient()
    result = client.generate(prompt, task="idea")  # or "code" / "fix"
"""

import os
import time
from typing import Optional, List, Dict

from dotenv import load_dotenv

load_dotenv()


class GroqClient:
    """Unified Groq client with 5-key pool allocation and task-based model routing."""

    # ── Models grouped by task type — ordered by preference ──
    MODELS_BY_TASK = {
        "idea": [  # Creative generation, large context, good reasoning
            "moonshotai/kimi-k2-instruct",                      # 60 RPM, best reasoning
            "meta-llama/llama-4-scout-17b-16e-instruct",        # 30K TPM, large context
            "qwen/qwen3-32b",                                   # 60 RPM, versatile
        ],
        "code": [  # Code generation, strong logic, structured output
            "llama-3.3-70b-versatile",                          # Best logic, 12K TPM
            "meta-llama/llama-4-maverick-17b-128e-instruct",    # Good code gen
            "qwen/qwen3-32b",                                   # Versatile fallback
        ],
        "fix": [   # Quick bug fixes, fast turnaround
            "llama-3.1-8b-instant",                             # 14.4K RPD, fastest
            "meta-llama/llama-4-scout-17b-16e-instruct",        # Good understanding
        ],
    }

    # ── Key pool indices (0-based) per task type ──
    # idea: K1,K2 (index 0,1) — creative tasks get dedicated keys
    # code: K3,K4 (index 2,3) — code gen gets dedicated keys
    # fix:  K5    (index 4)   — fix is fast & light, 1 key suffices
    KEY_POOLS = {
        "idea": [0, 1],
        "code": [2, 3],
        "fix":  [4],
    }

    # Rate limit cooldown tracking (seconds to wait per key)
    RATE_LIMIT_COOLDOWN = 65  # Wait slightly over 1 minute for RPM reset

    def __init__(self):
        # Collect all available keys (up to 5)
        self.keys: List[str] = []
        for i in range(1, 6):
            env_name = "GROQ_API_KEY" if i == 1 else f"GROQ_API_KEY_{i}"
            key = os.getenv(env_name, "").strip().strip('"')
            if key:
                self.keys.append(key)

        # Per-pool round-robin index
        self._pool_index: Dict[str, int] = {"idea": 0, "code": 0, "fix": 0}

        # Rate limit cooldown tracker: key_index → timestamp when usable again
        self._key_cooldown: Dict[int, float] = {}

        # Stats
        self.calls = 0
        self.successes = 0
        self.model_stats: Dict[str, int] = {}
        self.pool_stats: Dict[str, int] = {"idea": 0, "code": 0, "fix": 0, "overflow": 0}

        # Lazy-init clients per key
        self._clients: dict = {}

        if self.keys:
            n = len(self.keys)
            print(f"   🔑 Groq: {n} keys loaded — pool: idea=K1-K{min(2,n)}, code=K{min(3,n)}-K{min(4,n)}, fix=K{min(5,n)}")

    def _get_client(self, key: str):
        """Get or create an OpenAI-compatible client for a Groq key."""
        if key not in self._clients:
            from openai import OpenAI
            self._clients[key] = OpenAI(
                base_url="https://api.groq.com/openai/v1",
                api_key=key,
            )
        return self._clients[key]

    def _is_key_available(self, key_index: int) -> bool:
        """Check if a key is available (not in cooldown)."""
        if key_index not in self._key_cooldown:
            return True
        return time.time() >= self._key_cooldown[key_index]

    def _mark_key_limited(self, key_index: int):
        """Mark a key as rate-limited with cooldown."""
        self._key_cooldown[key_index] = time.time() + self.RATE_LIMIT_COOLDOWN

    def _get_pool_keys(self, task: str) -> List[int]:
        """Get available key indices for a task's pool, respecting cooldowns."""
        pool = self.KEY_POOLS.get(task, self.KEY_POOLS["idea"])
        # Filter to keys that actually exist and are available
        available = [i for i in pool if i < len(self.keys) and self._is_key_available(i)]
        return available

    def _get_overflow_keys(self, task: str) -> List[int]:
        """Get overflow keys from OTHER pools (not the task's own pool)."""
        own_pool = set(self.KEY_POOLS.get(task, []))
        all_indices = set(range(len(self.keys)))
        overflow = all_indices - own_pool
        # Sort: prefer fix pool key (K5) as overflow first, then others
        return sorted(
            [i for i in overflow if self._is_key_available(i)],
            key=lambda x: (0 if x == 4 else 1, x)
        )

    def _rotate_pool_keys(self, pool_keys: List[int], task: str) -> List[int]:
        """Reorder pool_keys starting from current round-robin index for even distribution."""
        if not pool_keys:
            return []
        n = len(pool_keys)
        start = self._pool_index.get(task, 0) % n
        self._pool_index[task] = (start + 1) % n
        return pool_keys[start:] + pool_keys[:start]

    def _try_call(self, key_index: int, model: str, prompt: str) -> Optional[str]:
        """Attempt a single API call. Returns result or None."""
        key = self.keys[key_index]
        key_label = f"K{key_index + 1}"
        try:
            print(f"   🔄 Groq [{key_label}]: {model.split('/')[-1]}...")
            client = self._get_client(key)
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": (
                        "You are an expert Python Quantitative Developer. "
                        "Output ONLY valid Python code or strategy ideas as requested. "
                        "No markdown, no explanations."
                    )},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=4096,
                temperature=0.7,
            )
            result = response.choices[0].message.content
            if result:
                return result
        except Exception as e:
            error_msg = str(e)
            if '429' in error_msg or 'rate_limit' in error_msg.lower():
                print(f"   ⚠️ Groq {model.split('/')[-1]} {key_label} rate limited, cooldown {self.RATE_LIMIT_COOLDOWN}s")
                self._mark_key_limited(key_index)
                return None
            print(f"   ⚠️ Groq {model.split('/')[-1]} {key_label} failed: {error_msg[:80]}")
        return None

    def generate(self, prompt: str, task: str = "idea") -> Optional[str]:
        """
        Generate text via Groq with pool-based key allocation + model failover.

        Strategy:
          1. Try models × dedicated pool keys (task-specific)
          2. Try models × overflow keys (from other pools)
          3. If all exhausted, return None → caller falls back to GitHub/Gemini

        Args:
            prompt: The prompt to send
            task: "idea", "code", or "fix" — selects model list & key pool

        Returns:
            Generated text or None if all attempts fail
        """
        if not self.keys:
            print("   ⚠️ Groq: no API keys configured")
            return None

        self.calls += 1
        models = self.MODELS_BY_TASK.get(task, self.MODELS_BY_TASK["idea"])

        # Phase 1: Try dedicated pool keys (round-robin start for even distribution)
        pool_keys = self._get_pool_keys(task)
        if pool_keys:
            rotated_keys = self._rotate_pool_keys(pool_keys, task)
            for model in models:
                for key_idx in rotated_keys:
                    result = self._try_call(key_idx, model, prompt)
                    if result:
                        self.successes += 1
                        self.model_stats[model] = self.model_stats.get(model, 0) + 1
                        self.pool_stats[task] = self.pool_stats.get(task, 0) + 1
                        key_label = f"K{key_idx + 1}"
                        print(f"   ✅ Groq ({model.split('/')[-1]} {key_label} pool:{task}) success")
                        return result

        # Phase 2: Try overflow keys from other pools
        overflow_keys = self._get_overflow_keys(task)
        if overflow_keys:
            print(f"   🔀 Groq: {task} pool exhausted, trying overflow keys...")
            for model in models:
                for key_idx in overflow_keys:
                    result = self._try_call(key_idx, model, prompt)
                    if result:
                        self.successes += 1
                        self.model_stats[model] = self.model_stats.get(model, 0) + 1
                        self.pool_stats["overflow"] = self.pool_stats.get("overflow", 0) + 1
                        key_label = f"K{key_idx + 1}"
                        print(f"   ✅ Groq ({model.split('/')[-1]} {key_label} overflow) success")
                        return result

        print(f"   ❌ Groq: all {len(self.keys)} keys × {len(models)} models exhausted for '{task}'")
        return None

    def get_stats(self) -> str:
        """Return a human-readable stats string."""
        parts = [f"Groq: {self.successes}/{self.calls} ok, {len(self.keys)} keys"]

        # Pool usage stats
        pool_info = []
        for pool_name in ["idea", "code", "fix", "overflow"]:
            count = self.pool_stats.get(pool_name, 0)
            if count > 0:
                pool_info.append(f"{pool_name}={count}")
        if pool_info:
            parts.append(f"pools: {', '.join(pool_info)}")

        # Top models
        if self.model_stats:
            top = sorted(self.model_stats.items(), key=lambda x: x[1], reverse=True)
            model_info = ", ".join(f"{m.split('/')[-1]}={c}" for m, c in top[:3])
            parts.append(f"models: {model_info}")

        # Active cooldowns
        active_cooldowns = sum(1 for k, t in self._key_cooldown.items() if time.time() < t)
        if active_cooldowns > 0:
            parts.append(f"cooling: {active_cooldowns} keys")

        return " | ".join(parts)
