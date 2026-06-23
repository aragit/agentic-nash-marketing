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