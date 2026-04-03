"""
evaluation/api_client.py
========================
Unified API client for multimodal geo-localization evaluation.

Supports multiple provider backends via a common OpenAI-compatible interface:
  - Local vLLM (self-hosted, no API key needed)
  - School relay station (LCPU)
  - SiliconFlow (open-source models via cloud)
  - OpenRouter (multi-provider gateway, closed-source models)
  - OpenAI (direct)
  - Anthropic (via OpenAI-compatible endpoint)
  - Google Gemini (via OpenAI-compatible endpoint)

Usage:
    from evaluation.api_client import build_client, PROVIDER_CONFIGS

    client = build_client("gpt-4o", provider="openrouter", api_key="sk-...")
    result = client.predict_location(base64_image)
"""

import requests
import json
import re
import time
import base64
import io
from typing import Optional, Tuple


# ===========================================================================
#  Provider Configuration Registry
# ===========================================================================

PROVIDER_CONFIGS = {
    # Local vLLM (self-hosted, no API key required)
    "local": {
        "api_base": "http://0.0.0.0:8001/v1",
        "requires_api_key": False,
        "default_api_key": "local-key",
        "supports_frequency_penalty": True,
        "max_image_size_mb": None,  # No limit
        "notes": "Self-hosted vLLM. Start with: vllm serve <model> --port 8001",
    },
    # School relay station (LCPU)
    "relay": {
        "api_base": "https://llmapi.lcpu.dev/v1",
        "requires_api_key": True,
        "default_api_key": None,
        "supports_frequency_penalty": False,
        "max_image_size_mb": 20,
        "notes": "LCPU school relay. Supports GPT, Claude, Gemini, Qwen, etc.",
    },
    # SiliconFlow (open-source models)
    "siliconflow": {
        "api_base": "https://api.siliconflow.cn/v1",
        "requires_api_key": True,
        "default_api_key": None,
        "supports_frequency_penalty": False,
        "max_image_size_mb": 10,
        "notes": "SiliconFlow cloud. Good for large open-source models (Qwen2.5-VL-72B, InternVL, etc.)",
    },
    # OpenRouter (multi-provider gateway)
    "openrouter": {
        "api_base": "https://openrouter.ai/api/v1",
        "requires_api_key": True,
        "default_api_key": None,
        "supports_frequency_penalty": False,
        "max_image_size_mb": 20,
        "notes": "OpenRouter gateway. Supports GPT-4o, Claude, Gemini, Llama, etc.",
        "extra_headers": {
            "HTTP-Referer": "https://github.com/inorganicwriter/SIGNPOST-Bench",
            "X-Title": "SIGNPOST-Bench",
        },
    },
    # OpenAI direct
    "openai": {
        "api_base": "https://api.openai.com/v1",
        "requires_api_key": True,
        "default_api_key": None,
        "supports_frequency_penalty": True,
        "max_image_size_mb": 20,
        "notes": "OpenAI direct API.",
    },
}

# ===========================================================================
#  Model Registry: short_name -> {model_id, provider}
# ===========================================================================

MODEL_REGISTRY = {
    # =========================================================
    # Local vLLM Models (self-hosted)
    # =========================================================
    "qwen3-30b": {
        "model": "/home/nas/lsr/huggingface/Qwen/Qwen3-VL-30B-A3B-Thinking",
        "provider": "local",
        "thinking": True,
    },
    "qwen3-8b": {
        "model": "/home/nas/lsr/huggingface/Qwen/Qwen3-VL-8B-Thinking",
        "provider": "local",
        "thinking": True,
    },

    # =========================================================
    # SiliconFlow (Open-Source Cloud)
    # Model list verified: 2026-03-18
    # https://api.siliconflow.cn/v1/models
    # =========================================================
    # Qwen3-VL series (latest)
    "qwen3-vl-235b-thinking-sf": {
        "model": "Qwen/Qwen3-VL-235B-A22B-Thinking",
        "provider": "siliconflow",
        "thinking": True,
    },
    "qwen3-vl-235b-sf": {
        "model": "Qwen/Qwen3-VL-235B-A22B-Instruct",
        "provider": "siliconflow",
        "thinking": False,
    },
    "qwen3-vl-32b-thinking-sf": {
        "model": "Qwen/Qwen3-VL-32B-Thinking",
        "provider": "siliconflow",
        "thinking": True,
    },
    "qwen3-vl-32b-sf": {
        "model": "Qwen/Qwen3-VL-32B-Instruct",
        "provider": "siliconflow",
        "thinking": False,
    },
    "qwen3-vl-30b-thinking-sf": {
        "model": "Qwen/Qwen3-VL-30B-A3B-Thinking",
        "provider": "siliconflow",
        "thinking": True,
    },
    "qwen3-vl-30b-sf": {
        "model": "Qwen/Qwen3-VL-30B-A3B-Instruct",
        "provider": "siliconflow",
        "thinking": False,
    },
    "qwen3-vl-8b-thinking-sf": {
        "model": "Qwen/Qwen3-VL-8B-Thinking",
        "provider": "siliconflow",
        "thinking": True,
    },
    "qwen3-vl-8b-sf": {
        "model": "Qwen/Qwen3-VL-8B-Instruct",
        "provider": "siliconflow",
        "thinking": False,
    },
    # Qwen2.5-VL series
    "qwen2.5-vl-72b-sf": {
        "model": "Qwen/Qwen2.5-VL-72B-Instruct",
        "provider": "siliconflow",
        "thinking": False,
    },
    "qwen2.5-vl-32b-sf": {
        "model": "Qwen/Qwen2.5-VL-32B-Instruct",
        "provider": "siliconflow",
        "thinking": False,
    },
    # GLM-4.6V (新增，硅基流动支持)
    "glm-4.6v-sf": {
        "model": "zai-org/GLM-4.6V",
        "provider": "siliconflow",
        "thinking": False,
    },
    # GLM-4.5V (新增，硅基流动支持)
    "glm-4.5v-sf": {
        "model": "zai-org/GLM-4.5V",
        "provider": "siliconflow",
        "thinking": False,
    },

    # =========================================================
    # OpenRouter (Multi-Provider Gateway)
    # Model list verified: 2026-03-18
    # https://openrouter.ai/api/v1/models
    # =========================================================

    # --- OpenAI ---
    "gpt-5": {
        "model": "openai/gpt-5",
        "provider": "openrouter",
        "thinking": False,
    },
    "gpt-5.4": {
        "model": "openai/gpt-5.4",
        "provider": "openrouter",
        "thinking": False,
    },
    "gpt-5.4-mini": {
        "model": "openai/gpt-5.4-mini",
        "provider": "openrouter",
        "thinking": False,
    },
    "gpt-4.1": {
        "model": "openai/gpt-4.1",
        "provider": "openrouter",
        "thinking": False,
    },
    "gpt-4.1-mini": {
        "model": "openai/gpt-4.1-mini",
        "provider": "openrouter",
        "thinking": False,
    },
    "gpt-4o": {
        "model": "openai/gpt-4o",
        "provider": "openrouter",
        "thinking": False,
    },
    "o3": {
        "model": "openai/o3",
        "provider": "openrouter",
        "thinking": True,
    },
    "o4-mini": {
        "model": "openai/o4-mini",
        "provider": "openrouter",
        "thinking": True,
    },

    # --- Anthropic Claude ---
    "claude-opus-4.6": {
        "model": "anthropic/claude-opus-4.6",
        "provider": "openrouter",
        "thinking": False,
    },
    "claude-sonnet-4.6": {
        "model": "anthropic/claude-sonnet-4.6",
        "provider": "openrouter",
        "thinking": False,
    },
    "claude-haiku-4.5": {
        "model": "anthropic/claude-haiku-4.5",
        "provider": "openrouter",
        "thinking": False,
    },
    "claude-3.7-sonnet": {
        "model": "anthropic/claude-3.7-sonnet",
        "provider": "openrouter",
        "thinking": False,
    },
    "claude-3.7-sonnet-thinking": {
        "model": "anthropic/claude-3.7-sonnet:thinking",
        "provider": "openrouter",
        "thinking": True,
    },

    # --- Google Gemini ---
    "gemini-3.1-pro": {
        "model": "google/gemini-3.1-pro-preview",
        "provider": "openrouter",
        "thinking": False,
    },
    "gemini-3.1-flash": {
        "model": "google/gemini-3.1-flash-image-preview",
        "provider": "openrouter",
        "thinking": False,
    },
    "gemini-3-pro": {
        "model": "google/gemini-3-pro-preview",
        "provider": "openrouter",
        "thinking": False,
    },
    "gemini-2.5-pro": {
        "model": "google/gemini-2.5-pro",
        "provider": "openrouter",
        "thinking": False,
    },
    "gemini-2.5-flash": {
        "model": "google/gemini-2.5-flash",
        "provider": "openrouter",
        "thinking": False,
    },

    # --- xAI Grok ---
    "grok-4": {
        "model": "x-ai/grok-4",
        "provider": "openrouter",
        "thinking": False,
    },
    "grok-4.20": {
        "model": "x-ai/grok-4.20-beta",
        "provider": "openrouter",
        "thinking": False,
    },

    # --- Qwen via OpenRouter ---
    "qwen3-vl-235b-or": {
        "model": "qwen/qwen3-vl-235b-a22b-instruct",
        "provider": "openrouter",
        "thinking": False,
    },
    "qwen3-vl-235b-thinking-or": {
        "model": "qwen/qwen3-vl-235b-a22b-thinking",
        "provider": "openrouter",
        "thinking": True,
    },
    "qwen3-vl-32b-or": {
        "model": "qwen/qwen3-vl-32b-instruct",
        "provider": "openrouter",
        "thinking": False,
    },
    "qwen3-vl-30b-or": {
        "model": "qwen/qwen3-vl-30b-a3b-instruct",
        "provider": "openrouter",
        "thinking": False,
    },
    "qwen3-vl-30b-thinking-or": {
        "model": "qwen/qwen3-vl-30b-a3b-thinking",
        "provider": "openrouter",
        "thinking": True,
    },
    "qwen3-vl-8b-or": {
        "model": "qwen/qwen3-vl-8b-instruct",
        "provider": "openrouter",
        "thinking": False,
    },
    "qwen3-vl-8b-thinking-or": {
        "model": "qwen/qwen3-vl-8b-thinking",
        "provider": "openrouter",
        "thinking": True,
    },
    "qwen3.5-397b-or": {
        "model": "qwen/qwen3.5-397b-a17b",
        "provider": "openrouter",
        "thinking": False,
    },
    "qwen3.5-122b-or": {
        "model": "qwen/qwen3.5-122b-a10b",
        "provider": "openrouter",
        "thinking": False,
    },
    "qwen3.5-27b-or": {
        "model": "qwen/qwen3.5-27b",
        "provider": "openrouter",
        "thinking": False,
    },

    # --- Meta Llama ---
    "llama-4-maverick": {
        "model": "meta-llama/llama-4-maverick",
        "provider": "openrouter",
        "thinking": False,
    },
    "llama-4-scout": {
        "model": "meta-llama/llama-4-scout",
        "provider": "openrouter",
        "thinking": False,
    },

    # --- Mistral ---
    "mistral-medium-3.1": {
        "model": "mistralai/mistral-medium-3.1",
        "provider": "openrouter",
        "thinking": False,
    },

    # --- GLM ---
    "glm-4.6v": {
        "model": "z-ai/glm-4.6v",
        "provider": "openrouter",
        "thinking": False,
    },
    "glm-4.5v": {
        "model": "z-ai/glm-4.5v",
        "provider": "openrouter",
        "thinking": False,
    },

    # --- Amazon Nova ---
    "nova-pro": {
        "model": "amazon/nova-pro-v1",
        "provider": "openrouter",
        "thinking": False,
    },
    "nova-lite": {
        "model": "amazon/nova-lite-v1",
        "provider": "openrouter",
        "thinking": False,
    },

    # =========================================================
    # OpenRouter FREE Models (for testing / pipeline validation)
    # These models have ":free" suffix and cost $0.
    # Use these to verify the pipeline works before spending money.
    #
    # Fetched from: https://openrouter.ai/api/v1/models
    # Last updated: 2026-03-28
    #
    # ⚠️  Only VISION models (text+image->text) are useful for SIGNPOST-Bench.
    #     Text-only free models are listed but commented out.
    # =========================================================

    # --- Vision-capable FREE models (can process images) ---
    "gemma3-27b-free": {
        "model": "google/gemma-3-27b-it:free",
        "provider": "openrouter",
        "thinking": False,
    },
    "gemma3-12b-free": {
        "model": "google/gemma-3-12b-it:free",
        "provider": "openrouter",
        "thinking": False,
    },
    "gemma3-4b-free": {
        "model": "google/gemma-3-4b-it:free",
        "provider": "openrouter",
        "thinking": False,
    },
    "mistral-small-3.1-free": {
        "model": "mistralai/mistral-small-3.1-24b-instruct:free",
        "provider": "openrouter",
        "thinking": False,
    },
    "nemotron-nano-12b-vl-free": {
        "model": "nvidia/nemotron-nano-12b-v2-vl:free",
        "provider": "openrouter",
        "thinking": False,
    },

    # --- Text-only FREE models (cannot process images, NOT for evaluation) ---
    # Kept here for reference; uncomment if needed for text-only tasks.
    # "trinity-large-free": {
    #     "model": "arcee-ai/trinity-large-preview:free",
    #     "provider": "openrouter",
    #     "thinking": False,
    # },
    # "trinity-mini-free": {
    #     "model": "arcee-ai/trinity-mini:free",
    #     "provider": "openrouter",
    #     "thinking": False,
    # },
    # "dolphin-mistral-24b-free": {
    #     "model": "cognitivecomputations/dolphin-mistral-24b-venice-edition:free",
    #     "provider": "openrouter",
    #     "thinking": False,
    # },
    # "gemma3n-e2b-free": {
    #     "model": "google/gemma-3n-e2b-it:free",
    #     "provider": "openrouter",
    #     "thinking": False,
    # },
    # "gemma3n-e4b-free": {
    #     "model": "google/gemma-3n-e4b-it:free",
    #     "provider": "openrouter",
    #     "thinking": False,
    # },
    # "lfm-2.5-1.2b-free": {
    #     "model": "liquid/lfm-2.5-1.2b-instruct:free",
    #     "provider": "openrouter",
    #     "thinking": False,
    # },
    # "lfm-2.5-1.2b-thinking-free": {
    #     "model": "liquid/lfm-2.5-1.2b-thinking:free",
    #     "provider": "openrouter",
    #     "thinking": True,
    # },
    # "llama-3.2-3b-free": {
    #     "model": "meta-llama/llama-3.2-3b-instruct:free",
    #     "provider": "openrouter",
    #     "thinking": False,
    # },
    # "llama-3.3-70b-free": {
    #     "model": "meta-llama/llama-3.3-70b-instruct:free",
    #     "provider": "openrouter",
    #     "thinking": False,
    # },
    # "minimax-m2.5-free": {
    #     "model": "minimax/minimax-m2.5:free",
    #     "provider": "openrouter",
    #     "thinking": False,
    # },
    # "hermes-3-405b-free": {
    #     "model": "nousresearch/hermes-3-llama-3.1-405b:free",
    #     "provider": "openrouter",
    #     "thinking": False,
    # },
    # "nemotron-3-nano-30b-free": {
    #     "model": "nvidia/nemotron-3-nano-30b-a3b:free",
    #     "provider": "openrouter",
    #     "thinking": False,
    # },
    # "nemotron-3-super-120b-free": {
    #     "model": "nvidia/nemotron-3-super-120b-a12b:free",
    #     "provider": "openrouter",
    #     "thinking": False,
    # },
    # "nemotron-nano-9b-free": {
    #     "model": "nvidia/nemotron-nano-9b-v2:free",
    #     "provider": "openrouter",
    #     "thinking": False,
    # },
    # "gpt-oss-120b-free": {
    #     "model": "openai/gpt-oss-120b:free",
    #     "provider": "openrouter",
    #     "thinking": False,
    # },
    # "gpt-oss-20b-free": {
    #     "model": "openai/gpt-oss-20b:free",
    #     "provider": "openrouter",
    #     "thinking": False,
    # },
    # "qwen3-4b-free": {
    #     "model": "qwen/qwen3-4b:free",
    #     "provider": "openrouter",
    #     "thinking": False,
    # },
    # "qwen3-coder-free": {
    #     "model": "qwen/qwen3-coder:free",
    #     "provider": "openrouter",
    #     "thinking": False,
    # },
    # "qwen3-next-80b-free": {
    #     "model": "qwen/qwen3-next-80b-a3b-instruct:free",
    #     "provider": "openrouter",
    #     "thinking": False,
    # },
    # "step-3.5-flash-free": {
    #     "model": "stepfun/step-3.5-flash:free",
    #     "provider": "openrouter",
    #     "thinking": False,
    # },
    # "glm-4.5-air-free": {
    #     "model": "z-ai/glm-4.5-air:free",
    #     "provider": "openrouter",
    #     "thinking": False,
    # },
}


# ===========================================================================
#  Unified API Client
# ===========================================================================

class GeoLocalizationClient:
    """
    Unified client for geo-localization inference across multiple API providers.

    Handles:
    - Provider-specific headers and parameters
    - Thinking model tag stripping (<think>...</think>)
    - Automatic retry with temperature escalation
    - Image compression for providers with size limits
    - Coordinate parsing with multiple fallback strategies
    """

    PROMPT = (
        "Analyze this photo and determine where it was taken.\n"
        "You MUST provide your best estimate of GPS coordinates even if uncertain.\n"
        "Do NOT refuse. Always give a coordinate guess.\n"
        "Output ONLY: (Latitude, Longitude)\n"
        "Example: (48.8584, 2.2945)"
    )

    def __init__(
        self,
        model_name: str,
        api_base: str,
        api_key: str,
        provider: str = "local",
        is_thinking_model: bool = False,
        max_tokens: int = 2048,
        max_retries: int = 3,
        timeout: int = 90,
    ):
        self.model_name = model_name
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.provider = provider
        self.is_thinking_model = is_thinking_model
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.timeout = timeout

        provider_cfg = PROVIDER_CONFIGS.get(provider, {})
        self.supports_frequency_penalty = provider_cfg.get("supports_frequency_penalty", False)
        self.max_image_size_mb = provider_cfg.get("max_image_size_mb", None)
        self.extra_headers = provider_cfg.get("extra_headers", {})

    def _build_headers(self) -> dict:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        headers.update(self.extra_headers)
        return headers

    def _compress_image_if_needed(self, base64_image: str) -> str:
        """Compress image if it exceeds provider's size limit."""
        if self.max_image_size_mb is None:
            return base64_image

        size_mb = len(base64_image) * 3 / 4 / (1024 * 1024)
        if size_mb <= self.max_image_size_mb:
            return base64_image

        try:
            from PIL import Image
            img_bytes = base64.b64decode(base64_image)
            img = Image.open(io.BytesIO(img_bytes))

            # Resize to fit within limit
            max_pixels = int(self.max_image_size_mb * 1024 * 1024 * 0.75)
            w, h = img.size
            scale = (max_pixels / (w * h * 3)) ** 0.5
            if scale < 1.0:
                new_w, new_h = int(w * scale), int(h * scale)
                img = img.resize((new_w, new_h), Image.LANCZOS)

            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=85)
            compressed = base64.b64encode(buf.getvalue()).decode("utf-8")
            print(f"  [IMG] Compressed {size_mb:.1f}MB -> {len(compressed)*3/4/1024/1024:.1f}MB")
            return compressed
        except ImportError:
            print("  [WARN] PIL not installed, skipping image compression")
            return base64_image
        except Exception as e:
            print(f"  [WARN] Image compression failed: {e}")
            return base64_image

    def _clean_thinking_tags(self, text: str) -> str:
        """Strip <think>...</think> tags from Thinking model output."""
        if not text:
            return ""
        if "</think>" in text:
            text = text.split("</think>")[-1]
        elif "<think>" in text:
            text = re.sub(r"<think>.*", "", text, flags=re.DOTALL)
        # Strip markdown code blocks
        text = re.sub(r"```[a-z]*\n?", "", text)
        return text.strip()

    def _build_payload(self, base64_image: str, temperature: float) -> dict:
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self.PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                        },
                    ],
                }
            ],
            "temperature": temperature,
            "max_tokens": self.max_tokens,
        }
        if self.supports_frequency_penalty:
            payload["frequency_penalty"] = 0.1
        return payload

    def predict_location(self, base64_image: str) -> Optional[str]:
        """
        Run geo-localization inference on a base64-encoded image.
        Returns the raw text response (coordinates), or None on failure.
        """
        base64_image = self._compress_image_if_needed(base64_image)
        url = f"{self.api_base}/chat/completions"
        headers = self._build_headers()
        current_temp = 0.0

        for attempt in range(self.max_retries + 1):
            payload = self._build_payload(base64_image, current_temp)
            try:
                response = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
                response.raise_for_status()
                result = response.json()

                if "choices" not in result or not result["choices"]:
                    print(f"  [WARN] No choices in response (attempt {attempt+1})")
                    current_temp = min(current_temp + 0.1, 1.0)
                    continue

                choice = result["choices"][0]
                finish_reason = choice.get("finish_reason", "")

                # Thinking model runaway detection
                if finish_reason == "length" and self.is_thinking_model:
                    if attempt < self.max_retries:
                        print(f"  ⚠️  Thinking runaway (truncated), retrying {attempt+1}/{self.max_retries}...")
                        current_temp = min(current_temp + 0.1, 1.0)
                        continue
                    else:
                        print("  ❌ Max retries for thinking runaway.")
                        return None

                content = choice["message"].get("content", "")
                if self.is_thinking_model:
                    content = self._clean_thinking_tags(content)

                if content:
                    lat, lon = self.parse_coordinates(content)
                    if lat is not None:
                        return content
                    # Coordinates not found, retry
                    if attempt < self.max_retries:
                        print(f"  ⚠️  No coordinates found, retrying {attempt+1}/{self.max_retries}...")
                        current_temp = min(current_temp + 0.15, 1.0)
                        continue
                    return content  # Return as-is on last attempt

                if attempt < self.max_retries:
                    print(f"  ⚠️  Empty response, retrying...")
                    time.sleep(1)
                    continue

            except requests.exceptions.HTTPError as e:
                status = e.response.status_code if e.response else "?"
                print(f"  HTTP {status} Error (attempt {attempt+1}): {e}")
                if status == 429:  # Rate limit
                    wait = 2 ** attempt
                    print(f"  Rate limited, waiting {wait}s...")
                    time.sleep(wait)
                elif status in (500, 502, 503):  # Server error
                    time.sleep(2)
                else:
                    return None  # Client error, don't retry
            except requests.exceptions.Timeout:
                print(f"  Timeout (attempt {attempt+1}), retrying...")
                time.sleep(2)
            except Exception as e:
                print(f"  API Error (attempt {attempt+1}): {e}")
                if attempt < self.max_retries:
                    time.sleep(1)

        return None

    @staticmethod
    def parse_coordinates(text: str) -> Tuple[Optional[float], Optional[float]]:
        """Parse GPS coordinates from model output using multiple strategies."""
        if not text:
            return None, None

        # 1. JSON format: {"latitude": ..., "longitude": ...}
        try:
            json_match = re.search(r"\{.*\}", text, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(0))
                if "latitude" in data and "longitude" in data:
                    return float(data["latitude"]), float(data["longitude"])
        except Exception:
            pass

        # 2. Tuple format: (lat, lon) or [lat, lon]
        match = re.search(r"[\(\[]\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*[\)\]]", text)
        if match:
            return float(match.group(1)), float(match.group(2))

        # 3. Labeled format: "Latitude: 48.85, Longitude: 2.29"
        lat = re.search(r"[Ll]at(?:itude)?[:\s]+(-?\d+\.?\d*)", text)
        lon = re.search(r"[Ll]on(?:gitude)?[:\s]+(-?\d+\.?\d*)", text)
        if lat and lon:
            return float(lat.group(1)), float(lon.group(1))

        # 4. Plain two-number format: "48.8584 2.2945"
        nums = re.findall(r"-?\d{1,3}\.\d{2,}", text)
        if len(nums) >= 2:
            try:
                lat_v, lon_v = float(nums[0]), float(nums[1])
                if -90 <= lat_v <= 90 and -180 <= lon_v <= 180:
                    return lat_v, lon_v
            except Exception:
                pass

        return None, None


# ===========================================================================
#  Factory Function
# ===========================================================================

def build_client(
    model_short_name: str,
    provider: Optional[str] = None,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    **kwargs,
) -> GeoLocalizationClient:
    """
    Build a GeoLocalizationClient from a model short name.

    Args:
        model_short_name: Key in MODEL_REGISTRY (e.g., "gpt-4o", "qwen3-30b")
                          or a full model path for local vLLM.
        provider: Override the provider (e.g., "openrouter", "siliconflow").
        api_key: API key. If None, uses environment variable or default.
        api_base: Override the API base URL.
        **kwargs: Additional arguments passed to GeoLocalizationClient.

    Returns:
        Configured GeoLocalizationClient instance.

    Examples:
        # Local vLLM
        client = build_client("qwen3-30b")

        # SiliconFlow
        client = build_client("qwen2.5-vl-72b", api_key="sk-sf-xxx")

        # OpenRouter
        client = build_client("gpt-4o", api_key="sk-or-xxx")

        # Custom model on OpenRouter
        client = build_client(
            "meta-llama/llama-3.2-90b-vision-instruct",
            provider="openrouter",
            api_key="sk-or-xxx"
        )
    """
    import os

    # Look up registry
    if model_short_name in MODEL_REGISTRY:
        entry = MODEL_REGISTRY[model_short_name]
        resolved_model = entry["model"]
        resolved_provider = provider or entry["provider"]
        is_thinking = entry.get("thinking", False)
    else:
        # Treat as full model path/ID (local vLLM or custom)
        resolved_model = model_short_name
        resolved_provider = provider or "local"
        is_thinking = kwargs.pop("is_thinking_model", False)

    # Get provider config
    provider_cfg = PROVIDER_CONFIGS.get(resolved_provider, PROVIDER_CONFIGS["local"])
    resolved_api_base = api_base or provider_cfg["api_base"]

    # Resolve API key
    if api_key:
        resolved_api_key = api_key
    elif not provider_cfg.get("requires_api_key", True):
        resolved_api_key = provider_cfg.get("default_api_key", "local-key")
    else:
        # Try environment variables
        env_map = {
            "openrouter": "OPENROUTER_API_KEY",
            "siliconflow": "SILICONFLOW_API_KEY",
            "openai": "OPENAI_API_KEY",
            "relay": "RELAY_API_KEY",
        }
        env_var = env_map.get(resolved_provider, f"{resolved_provider.upper()}_API_KEY")
        resolved_api_key = os.environ.get(env_var, "")
        if not resolved_api_key:
            raise ValueError(
                f"API key required for provider '{resolved_provider}'. "
                f"Pass api_key= or set environment variable {env_var}."
            )

    return GeoLocalizationClient(
        model_name=resolved_model,
        api_base=resolved_api_base,
        api_key=resolved_api_key,
        provider=resolved_provider,
        is_thinking_model=is_thinking,
        **kwargs,
    )


def list_models(provider: Optional[str] = None) -> None:
    """Print available models, optionally filtered by provider."""
    print(f"{'Short Name':<25} {'Provider':<15} {'Model ID':<50} {'Thinking'}")
    print("-" * 100)
    for name, entry in MODEL_REGISTRY.items():
        if provider and entry["provider"] != provider:
            continue
        thinking = "Y" if entry.get("thinking") else ""
        print(f"{name:<25} {entry['provider']:<15} {entry['model']:<50} {thinking}")


if __name__ == "__main__":
    list_models()
