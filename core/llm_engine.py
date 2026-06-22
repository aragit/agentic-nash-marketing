"""LLM Inference Engine for Nash Marketing Agents.

Supports two backends:
1. MockLLM — deterministic responses, instant, no download (default for local)
2. Transformers — real LLM via HuggingFace (CPU, downloads on first use)
"""

import os
import json
import time
import random
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any
from dataclasses import dataclass

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
        role = self._extract_role(system_prompt)

        # Extract context values from prompt
        budget = self._extract_float(system_prompt, "budget: $")
        remaining = self._extract_float(system_prompt, "remaining: $")
        market_price = self._extract_float(system_prompt, "market price: $")
        win_rate = self._extract_float(system_prompt, "win rate:")

        content = self._generate_strategy(role, budget, remaining, market_price, win_rate)

        latency_ms = (time.time() - start) * 1000

        return LLMResponse(
            content=content,
            tokens_in=len(str(messages)),
            tokens_out=len(content.split()),
            latency_ms=latency_ms,
            model=self.model_name,
        )

    def _extract_role(self, text: str) -> str:
        text_lower = text.lower()
        if "aggressive" in text_lower:
            return "aggressive"
        if "conservative" in text_lower:
            return "conservative"
        if "balanced" in text_lower:
            return "balanced"
        return "balanced"

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
        self, role: str, budget: float, remaining: float, market_price: float, win_rate: float
    ) -> str:
        if role == "aggressive":
            bid_multiplier = self.rng.uniform(1.05, 1.20)
            justification = "Aggressive market capture: outbidding competitors to secure impressions"
        elif role == "conservative":
            bid_multiplier = self.rng.uniform(0.85, 0.95)
            justification = "Conservative ROI focus: minimizing CPA while maintaining presence"
        else:
            bid_multiplier = self.rng.uniform(0.95, 1.05)
            justification = "Balanced approach: competitive bidding with budget awareness"

        bid = round(market_price * bid_multiplier, 2) if market_price > 0 else round(self.rng.uniform(1.0, 5.0), 2)

        # Budget guardrail in mock logic
        if remaining > 0 and bid > remaining * 0.1:
            bid = round(remaining * 0.05, 2)
            justification += " [GUARDRAIL: Bid capped at 5% of remaining budget]"

        return json.dumps({
            "bid": bid,
            "max_daily_spend": round(remaining * 0.15, 2) if remaining > 0 else 100.0,
            "target_cpa": round(bid * self.rng.uniform(0.8, 1.2), 2),
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
    """Factory to select the best available LLM backend."""

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