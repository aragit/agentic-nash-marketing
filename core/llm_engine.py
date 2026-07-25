"""LLM Inference Engine for Nash Marketing Agents.

Supports multiple backends:
1. MockLLM — deterministic responses, instant, no download (default for local)
2. Transformers — real LLM via HuggingFace (CPU, downloads on first use)
3. OllamaEngine — local Ollama server (async native)
4. VLLMEngine — vLLM OpenAI-compatible server (async native)
"""

import os
import json
import time
import random
import asyncio
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any
from dataclasses import dataclass

from core.telemetry import tracer

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    content: str
    tokens_in: int
    tokens_out: int
    latency_ms: float
    model: str


class BaseLLMEngine(ABC):
    """Abstract base for LLM inference backends."""

    @abstractmethod
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 512,
    ) -> LLMResponse:
        pass

    async def async_chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 512,
    ) -> LLMResponse:
        """Async version — delegates to sync method via thread pool.

        Override in subclasses that have native async I/O (e.g. aiohttp).
        """
        with tracer.start_as_current_span("llm_inference") as span:
            span.set_attribute("llm.model", getattr(self, "model_name", "unknown"))
            span.set_attribute("llm.provider", type(self).__name__)
            span.set_attribute("llm.temperature", temperature)
            span.set_attribute("llm.max_tokens", max_tokens)
            span.set_attribute("llm.input_tokens", len(str(messages)))

            result = await asyncio.to_thread(
                self.chat_completion, messages, temperature, max_tokens
            )

            span.set_attribute("llm.output_tokens", result.tokens_out)
            span.set_attribute("llm.latency_ms", result.latency_ms)
            return result

    @abstractmethod
    def shutdown(self):
        pass


class MockLLMEngine(BaseLLMEngine):
    """
    Deterministic mock LLM for local development and testing.
    Generates realistic bidding strategy responses without any model download.
    """

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self.model_name = "mock-llm-nash"
        logger.info("[LLM] Using MockLLM — deterministic, instant, no download")

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 512,
    ) -> LLMResponse:
        start = time.time()

        system_prompt = messages[0]["content"] if messages else ""

        # Detect planner prompts (expect single-word strategy response)
        if "strategic planner" in system_prompt.lower():
            content = self._generate_planner_response(system_prompt)
        else:
            role = self._extract_role(system_prompt)

            # Extract context values from prompt
            budget = self._extract_float(system_prompt, "budget: $")
            remaining = self._extract_float(system_prompt, "remaining budget: $")
            market_price = self._extract_float(system_prompt, "market clearing price: $")
            win_rate = self._extract_float(system_prompt, "win rate:")
            target_cpa = self._extract_float(system_prompt, "target CPA (cost per acquisition): $")

            content = self._generate_strategy(role, budget, remaining, market_price, win_rate, target_cpa)

        latency_ms = (time.time() - start) * 1000

        return LLMResponse(
            content=content,
            tokens_in=len(str(messages)),
            tokens_out=len(content.split()),
            latency_ms=latency_ms,
            model=self.model_name,
        )

    def _extract_role(self, text: str) -> str:
        idx = text.lower().find("your role: ")
        if idx >= 0:
            role_text = text[idx:idx+60].lower()
            if "aggressive" in role_text:
                return "aggressive"
            if "conservative" in role_text:
                return "conservative"
            if "balanced" in role_text:
                return "balanced"
        return "balanced"

    def _generate_planner_response(self, prompt: str) -> str:
        """Generate a deterministic planner response based on budget and win rate."""
        budget_pct = self._extract_budget_pct(prompt)
        win_rate = self._extract_win_rate(prompt)

        # Deterministic heuristic matching the original planner logic
        if budget_pct < 0.30:
            return "conserve"
        if win_rate < 0.20:
            return "aggressive"
        return "balanced"

    def _extract_budget_pct(self, text: str) -> float:
        """Extract budget percentage from planner prompt."""
        try:
            idx = text.lower().find("budget remaining:")
            if idx == -1:
                return 0.5
            # Skip past "budget remaining:" and any whitespace
            start = idx + len("budget remaining:")
            while start < len(text) and text[start] == " ":
                start += 1
            end = start
            while end < len(text) and (text[end].isdigit() or text[end] in ".%"):
                end += 1
            val = text[start:end].strip().rstrip("%")
            return float(val) / 100.0
        except (ValueError, IndexError):
            return 0.5

    def _extract_win_rate(self, text: str) -> float:
        """Extract win rate from planner prompt."""
        try:
            idx = text.lower().find("win rate:")
            if idx == -1:
                return 0.5
            # Skip past "win rate:" and any whitespace
            start = idx + len("win rate:")
            while start < len(text) and text[start] == " ":
                start += 1
            end = start
            while end < len(text) and (text[end].isdigit() or text[end] in ".%"):
                end += 1
            val = text[start:end].strip().rstrip("%")
            return float(val) / 100.0
        except (ValueError, IndexError):
            return 0.5

    def _extract_float(self, text: str, key: str) -> float:
        try:
            idx = text.lower().find(key.lower())
            if idx == -1:
                return 0.0
            start = idx + len(key)
            end = start
            while end < len(text) and (text[end].isdigit() or text[end] in ".,"):
                end += 1
            return float(text[start:end].replace(",", ""))
        except (ValueError, IndexError):
            return 0.0

    def _generate_strategy(
        self, role: str, budget: float, remaining: float, market_price: float, win_rate: float, target_cpa: float
    ) -> str:
        # Bid = target_cpa × role_pct  (role modulates how aggressively CPA is spent)
        if role == "aggressive":
            bid_pct = self.rng.uniform(0.70, 0.95)
            justification = "Aggressive: bidding near valuation to maximize win rate"
        elif role == "conservative":
            bid_pct = self.rng.uniform(0.05, 0.25)
            justification = "Conservative: bidding well below valuation to protect margin"
        else:
            bid_pct = self.rng.uniform(0.35, 0.60)
            justification = "Balanced: competitive bid with moderate budget risk"

        bid = round(max(target_cpa, 1.0) * bid_pct, 2) if target_cpa > 0 else round(market_price, 2)

        return json.dumps({
            "bid": bid,
            "max_daily_spend": round(remaining * 0.15, 2) if remaining > 0 else 100.0,
            "target_cpa": round(target_cpa * self.rng.uniform(0.9, 1.1), 2),
            "strategy": role,
            "justification": justification,
        })

    def shutdown(self):
        pass


class TransformersEngine(BaseLLMEngine):
    """Real LLM engine using HuggingFace Transformers. Downloads model on first use."""

    def __init__(self, model_name: str = "microsoft/Phi-3-mini-4k-instruct", device: str = "cpu"):
        logger.info(f"[LLM] Loading {model_name} via Transformers (CPU)...")
        start = time.time()

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map=device,
            trust_remote_code=True,
        )
        self.model.eval()

        load_time = time.time() - start
        logger.info(f"[LLM] Model loaded in {load_time:.1f}s")

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 512,
    ) -> LLMResponse:
        start = time.time()
        import torch

        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        input_ids = model_inputs.input_ids
        tokens_in = input_ids.shape[1]

        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        tokens_out = generated_ids.shape[1] - tokens_in
        response = self.tokenizer.decode(generated_ids[0][tokens_in:], skip_special_tokens=True)

        latency_ms = (time.time() - start) * 1000

        return LLMResponse(
            content=response.strip(),
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_ms=latency_ms,
            model=self.model_name,
        )

    def shutdown(self):
        del self.model
        import torch
        torch.cuda.empty_cache() if torch.cuda.is_available() else None


class LLMEngineFactory:
    """Legacy factory to select the best available LLM backend."""

    @staticmethod
    def create(
        model_name: str = "microsoft/Phi-3-mini-4k-instruct",
        use_mock: bool = True,
    ) -> BaseLLMEngine:
        if use_mock:
            return MockLLMEngine()

        try:
            return TransformersEngine(model_name)
        except Exception as e:
            logger.warning(f"[LLM] Transformers failed ({e}), falling back to MockLLM")
            return MockLLMEngine()


# ---------------------------------------------------------------------------
# High-throughput serving backends
# ---------------------------------------------------------------------------

class OllamaEngine(BaseLLMEngine):
    """Native async engine for local Ollama server.

    Requires the ``ollama`` Python package: ``pip install ollama``.
    """

    def __init__(self, model: str, host: str = "http://localhost:11434"):
        from ollama import AsyncClient as OllamaAsyncClient
        self.model = model
        self.client = OllamaAsyncClient(host=host)
        logger.info(f"[LLM] Ollama engine initialised — model={model}, host={host}")

    async def async_chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 512,
    ) -> LLMResponse:
        with tracer.start_as_current_span("llm_inference") as span:
            span.set_attribute("llm.model", self.model)
            span.set_attribute("llm.provider", "ollama")
            span.set_attribute("llm.temperature", temperature)
            span.set_attribute("llm.max_tokens", max_tokens)
            span.set_attribute("llm.input_tokens", len(str(messages)))

            start = time.time()
            response = await self.client.chat(
                model=self.model,
                messages=messages,
                options={"temperature": temperature, "num_predict": max_tokens},
                format="json",
            )
            latency_ms = (time.time() - start) * 1000

            content = response["message"]["content"]
            result = LLMResponse(
                content=content,
                tokens_in=0,  # Ollama does not expose token counts in chat API
                tokens_out=len(content.split()),
                latency_ms=latency_ms,
                model=self.model,
            )

            span.set_attribute("llm.output_tokens", result.tokens_out)
            span.set_attribute("llm.latency_ms", latency_ms)
            return result

    def chat_completion(self, messages, temperature=0.7, max_tokens=512):
        raise NotImplementedError("OllamaEngine is async-only; use async_chat_completion")

    def shutdown(self):
        pass


class VLLMEngine(BaseLLMEngine):
    """Native async engine for vLLM OpenAI-compatible server.

    Requires the ``openai`` Python package: ``pip install openai``.
    Works with any OpenAI-compatible endpoint (vLLM, TGI, LiteLLM, etc.).
    """

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:8000/v1",
        api_key: str = "not-needed",
    ):
        from openai import AsyncOpenAI
        self.model = model
        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        logger.info(f"[LLM] vLLM engine initialised — model={model}, base_url={base_url}")

    async def async_chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 512,
    ) -> LLMResponse:
        with tracer.start_as_current_span("llm_inference") as span:
            span.set_attribute("llm.model", self.model)
            span.set_attribute("llm.provider", "vllm")
            span.set_attribute("llm.temperature", temperature)
            span.set_attribute("llm.max_tokens", max_tokens)
            span.set_attribute("llm.input_tokens", len(str(messages)))

            start = time.time()
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            latency_ms = (time.time() - start) * 1000

            choice = response.choices[0]
            content = choice.message.content or ""
            usage = response.usage
            result = LLMResponse(
                content=content,
                tokens_in=usage.prompt_tokens if usage else 0,
                tokens_out=usage.completion_tokens if usage else 0,
                latency_ms=latency_ms,
                model=self.model,
            )

            span.set_attribute("llm.output_tokens", result.tokens_out)
            span.set_attribute("llm.latency_ms", latency_ms)
            return result

    def chat_completion(self, messages, temperature=0.7, max_tokens=512):
        raise NotImplementedError("VLLMEngine is async-only; use async_chat_completion")

    def shutdown(self):
        pass


# ---------------------------------------------------------------------------
# Factory for dependency injection
# ---------------------------------------------------------------------------

def create_llm_engine(
    provider: str = "mock",
    model: str = "microsoft/Phi-3-mini-4k-instruct",
    **kwargs,
) -> BaseLLMEngine:
    """Create an LLM engine by provider name.

    Args:
        provider: One of "mock", "transformers", "ollama", "vllm".
        model:    Model identifier (meaning depends on provider).
        **kwargs: Provider-specific options forwarded to the engine constructor.

    Returns:
        A ``BaseLLMEngine`` instance ready for injection.
    """
    provider = provider.lower()

    if provider == "mock":
        return MockLLMEngine(seed=kwargs.get("seed", 42))

    if provider == "transformers":
        return TransformersEngine(
            model_name=model,
            device=kwargs.get("device", "cpu"),
        )

    if provider == "ollama":
        return OllamaEngine(
            model=model,
            host=kwargs.get("host", "http://localhost:11434"),
        )

    if provider in ("vllm", "openai"):
        return VLLMEngine(
            model=model,
            base_url=kwargs.get("base_url", "http://localhost:8000/v1"),
            api_key=kwargs.get("api_key", "not-needed"),
        )

    raise ValueError(f"Unsupported LLM provider: {provider!r}")