# -*- coding: utf-8 -*-
"""
OpenAICompatibleProvider - OpenAI 兼容接口提供者
用于调用本地部署的开源模型（vLLM, Ollama, text-generation-webui 等）
"""

from __future__ import annotations

import os
import base64
import logging
import json
import asyncio
from pathlib import Path
from typing import Optional, Any, List, Dict, Union
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# 尝试导入 OpenAI SDK
try:
    from openai import OpenAI, AsyncOpenAI
    _OPENAI_AVAILABLE = True
except ImportError:
    _OPENAI_AVAILABLE = False
    OpenAI = None
    AsyncOpenAI = None

# ==========================================
# Base Classes (Inlined for standalone usage)
# ==========================================

@dataclass
class AnalysisResult:
    """分析结果封装"""
    success: bool
    content: Optional[str] = None
    error: Optional[str] = None
    raw_response: Any = None

    @classmethod
    def ok(cls, content: str, raw_response: Any = None) -> AnalysisResult:
        return cls(success=True, content=content, raw_response=raw_response)

    @classmethod
    def fail(cls, error: str) -> AnalysisResult:
        return cls(success=False, error=error)

class ModelProvider:
    """模型提供者基类"""
    name: str = "base"
    supports_json_mode: bool = False

    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_tokens: int = 2048,
        temperature: float = 0.0,
        **kwargs
    ):
        self.model_name = model_name
        self.api_key = api_key
        self.base_url = base_url
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.kwargs = kwargs

    def is_available(self) -> bool:
        return True

# ==========================================
# User Provided Implementation
# ==========================================

class OpenAICompatibleProvider(ModelProvider):
    """
    OpenAI 兼容接口提供者
    
    支持任何提供 OpenAI 兼容 API 的服务，包括：
    - vLLM (本地部署)
    - Ollama
    - text-generation-webui
    - LM Studio
    - 以及官方 OpenAI API
    """
    
    name = "openai_compatible"
    supports_json_mode = True
    
    def __init__(
        self,
        model_name: str = "gpt-4o",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_tokens: int = 2048,
        temperature: float = 0.0,
        use_base64: bool = True,  # 是否使用 base64 编码图像
        **kwargs
    ):
        super().__init__(
            model_name=model_name,
            api_key=api_key,
            base_url=base_url,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs
        )
        self.use_base64 = use_base64
        self._async_client: Optional[AsyncOpenAI] = None
        self._initialize_client()
    
    def _initialize_client(self) -> bool:
        """初始化 OpenAI 兼容客户端"""
        if not _OPENAI_AVAILABLE:
            logger.error("❌ OpenAI SDK 未安装，请运行: pip install openai")
            return False
        
        try:
            # 获取 API key（本地服务可以用占位符）
            key = self.api_key or os.getenv("OPENAI_API_KEY") or "local-key"
            url = self.base_url or os.getenv("OPENAI_BASE_URL")
            
            # 同步客户端
            self._client = OpenAI(
                api_key=key,
                base_url=url,
            )
            
            # 异步客户端
            self._async_client = AsyncOpenAI(
                api_key=key,
                base_url=url,
            )
            
            # logger.info("✅ OpenAI 兼容客户端初始化成功 (model=%s, base_url=%s)", self.model_name, url or "default")
            return True
        except Exception as e:
            logger.exception("❌ OpenAI 兼容客户端初始化失败: %s", e)
            self._client = None
            self._async_client = None
            return False
    
    def _encode_image_base64(self, image_path: Path) -> str:
        """将图像编码为 base64"""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    
    def _get_image_mime_type(self, image_path: Path) -> str:
        """根据文件扩展名获取 MIME 类型"""
        suffix = image_path.suffix.lower()
        mime_types = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".webp": "image/webp",
            ".bmp": "image/bmp",
        }
        return mime_types.get(suffix, "image/jpeg")
    
    async def analyze_image_async(
        self,
        image_path: Path,
        prompt: str,
        json_mode: bool = False,
    ) -> AnalysisResult:
        """
        异步分析图像
        """
        if not self._async_client:
            return AnalysisResult.fail("OpenAI 兼容客户端未初始化")
        
        if not image_path.exists():
            return AnalysisResult.fail(f"图片文件不存在: {image_path}")
        
        try:
            # 初始化 extra_kwargs
            extra_kwargs: Dict[str, Any] = {}
            if json_mode:
                extra_kwargs["response_format"] = {"type": "json_object"}
            
            # 检测是否为 Thinking 模型
            is_thinking_model = "thinking" in self.model_name.lower()
            
            # 如果是 Thinking 模型，强制关闭 JSON 模式，以免截断思考过程
            if is_thinking_model and json_mode:
                # logger.warning("🧠 检测到 Thinking 模型，自动禁用 JSON 模式以保留思考过程")
                json_mode = False
                if "response_format" in extra_kwargs:
                    del extra_kwargs["response_format"]
            
            # 构建消息
            if self.use_base64:
                image_data = self._encode_image_base64(image_path)
                mime_type = self._get_image_mime_type(image_path)
                user_content = [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{image_data}"}}
                ]
            else:
                user_content = [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_path.resolve().as_uri()}}
                ]
                
            messages = [
                {"role": "user", "content": user_content}
            ]
            
            # 添加防复读参数
            if "frequency_penalty" not in extra_kwargs:
                extra_kwargs["frequency_penalty"] = 0.1
            if "presence_penalty" not in extra_kwargs:
                extra_kwargs["presence_penalty"] = 0.1
            
            # 自动重试逻辑（使用局部变量 current_temperature，避免修改 self.temperature）
            max_runaway_retries = 3
            current_temperature = self.temperature  # 局部副本，不修改对象状态

            for attempt in range(max_runaway_retries + 1):
                try:
                    # 异步调用 API
                    response = await self._async_client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        max_tokens=self.max_tokens,
                        temperature=current_temperature,
                        **extra_kwargs,
                    )
                    
                    # 检查是否截断 (Runaway Check)
                    finish_reason = response.choices[0].finish_reason
                    if finish_reason == "length":
                        if attempt < max_runaway_retries:
                            logger.warning(f"⚠️ 检测到思考暴走 (Length Truncated), 正在重试 ({attempt+1}/{max_runaway_retries})...")
                            continue
                        else:
                            logger.error("❌ 思考暴走重试失败，放弃该样本。")

                    # 提取响应文本
                    message = response.choices[0].message
                    text = getattr(message, "content", None)
                    
                    if isinstance(text, str) and text.strip():
                        # 清理逻辑 (去除 <think>)
                        cleaned_text = text
                        if is_thinking_model: 
                            if "</think>" in text:
                                cleaned_text = text.split("</think>")[-1].strip()
                            elif "<think>" in text:
                                import re
                                cleaned_text = re.sub(r"<think>.*", "", text, flags=re.DOTALL).strip()
                        
                        # 清理 Markdown 代码块
                        if "```json" in cleaned_text:
                            cleaned_text = cleaned_text.replace("```json", "").replace("```", "")
                        
                        # 增强型 JSON 提取与验证
                        json_obj = None
                        validation_errors = []

                        try:
                            start_idx = cleaned_text.find("{")
                            end_idx = cleaned_text.rfind("}")
                            
                            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                                potential_json = cleaned_text[start_idx : end_idx + 1]
                                json_obj = json.loads(potential_json)
                                cleaned_text = json.dumps(json_obj, ensure_ascii=False)
                            else:
                                validation_errors.append("No JSON brackets '{}' found")
                        except json.JSONDecodeError as e:
                            validation_errors.append(f"JSON Parse Failed: {str(e)[:50]}...")
                        except Exception as e:
                            validation_errors.append(f"JSON Extraction Error: {str(e)}")

                        if not json_obj:
                            pass
                        else:
                            # 简单的验证，根据实际需求调整
                            pass

                        if validation_errors:
                            if attempt < max_runaway_retries:
                                logger.warning(f"⚠️ 输出校验失败 ({', '.join(validation_errors)}), 触发重试 ({attempt+1}/{max_runaway_retries})...")
                                current_temperature = min(current_temperature + 0.1, 1.0)
                                continue
                            else:
                                logger.error(f"❌ 最终校验失败: {', '.join(validation_errors)}")
                                return AnalysisResult.fail(f"Validation Failed: {', '.join(validation_errors)}")
                        
                        return AnalysisResult.ok(cleaned_text.strip(), raw_response=response)
                    
                    continue

                except Exception as e:
                    if attempt == max_runaway_retries:
                        raise e
                    logger.warning(f"API Error during attempt {attempt}: {e}")
            
            return AnalysisResult.fail("Max retries exceeded with invalid output")

        except Exception as e:
            logger.exception("❌ OpenAI 兼容 API 分析失败: %s", e)
            return AnalysisResult.fail(str(e))
