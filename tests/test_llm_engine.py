"""Tests for LLM engines — Ollama, vLLM, and factory."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from core.llm_engine import (
    BaseLLMEngine,
    MockLLMEngine,
    OllamaEngine,
    VLLMEngine,
    LLMResponse,
    create_llm_engine,
)


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------

def _make_ollama_response(content: str = "test response"):
    """Build a dict mimicking ollama's chat response."""
    return {"message": {"role": "assistant", "content": content}}


def _make_openai_response(content: str = "test response", prompt_tokens=10, completion_tokens=5):
    """Build objects mimicking openai's chat completion response."""
    choice = MagicMock()
    choice.message.content = content

    usage = MagicMock()
    usage.prompt_tokens = prompt_tokens
    usage.completion_tokens = completion_tokens

    response = MagicMock()
    response.choices = [choice]
    response.usage = usage
    return response


# ---------------------------------------------------------------------------
# OllamaEngine tests
# ---------------------------------------------------------------------------

class TestOllamaEngine:
    """Tests for OllamaEngine with mocked ollama client."""

    @pytest.mark.asyncio
    async def test_async_chat_completion_returns_response(self):
        """Should return LLMResponse with content from Ollama."""
        with patch("core.llm_engine.OllamaEngine.__init__", return_value=None):
            engine = OllamaEngine.__new__(OllamaEngine)
            engine.model = "llama3"
            engine.client = AsyncMock()
            engine.client.chat.return_value = _make_ollama_response("hello world")

        result = await engine.async_chat_completion(
            messages=[{"role": "user", "content": "hi"}],
            temperature=0.5,
            max_tokens=100,
        )
        assert isinstance(result, LLMResponse)
        assert result.content == "hello world"
        assert result.model == "llama3"

    @pytest.mark.asyncio
    async def test_client_chat_called_with_correct_args(self):
        """Should forward messages, model, and options to ollama client."""
        with patch("core.llm_engine.OllamaEngine.__init__", return_value=None):
            engine = OllamaEngine.__new__(OllamaEngine)
            engine.model = "mistral"
            engine.client = AsyncMock()
            engine.client.chat.return_value = _make_ollama_response("ok")

        msgs = [{"role": "system", "content": "You are a bid agent."}]
        await engine.async_chat_completion(messages=msgs, temperature=0.3, max_tokens=256)

        engine.client.chat.assert_awaited_once_with(
            model="mistral",
            messages=msgs,
            options={"temperature": 0.3, "num_predict": 256},
        )

    @pytest.mark.asyncio
    async def test_latency_is_positive(self):
        """Latency should be a non-negative float."""
        with patch("core.llm_engine.OllamaEngine.__init__", return_value=None):
            engine = OllamaEngine.__new__(OllamaEngine)
            engine.model = "test"
            engine.client = AsyncMock()
            engine.client.chat.return_value = _make_ollama_response("x")

        result = await engine.async_chat_completion([{"role": "user", "content": "y"}])
        assert result.latency_ms >= 0

    def test_sync_chat_completion_raises(self):
        """Sync path should raise NotImplementedError."""
        with patch("core.llm_engine.OllamaEngine.__init__", return_value=None):
            engine = OllamaEngine.__new__(OllamaEngine)
            engine.model = "test"
            engine.client = AsyncMock()

        with pytest.raises(NotImplementedError, match="async-only"):
            engine.chat_completion([{"role": "user", "content": "hi"}])


# ---------------------------------------------------------------------------
# VLLMEngine tests
# ---------------------------------------------------------------------------

class TestVLLMEngine:
    """Tests for VLLMEngine with mocked openai client."""

    @pytest.mark.asyncio
    async def test_async_chat_completion_returns_response(self):
        """Should return LLMResponse with content from vLLM."""
        with patch("core.llm_engine.VLLMEngine.__init__", return_value=None):
            engine = VLLMEngine.__new__(VLLMEngine)
            engine.model = "meta-llama/Llama-3-8B"
            engine.client = AsyncMock()
            engine.client.chat.completions.create = AsyncMock(
                return_value=_make_openai_response("bid result")
            )

        result = await engine.async_chat_completion(
            messages=[{"role": "user", "content": "decide bid"}],
            temperature=0.7,
            max_tokens=512,
        )
        assert isinstance(result, LLMResponse)
        assert result.content == "bid result"
        assert result.tokens_in == 10
        assert result.tokens_out == 5

    @pytest.mark.asyncio
    async def test_client_create_called_with_correct_args(self):
        """Should forward model, messages, temperature, max_tokens."""
        with patch("core.llm_engine.VLLMEngine.__init__", return_value=None):
            engine = VLLMEngine.__new__(VLLMEngine)
            engine.model = "my-model"
            engine.client = AsyncMock()
            engine.client.chat.completions.create = AsyncMock(
                return_value=_make_openai_response("ok")
            )

        msgs = [{"role": "system", "content": "plan"}]
        await engine.async_chat_completion(messages=msgs, temperature=0.1, max_tokens=64)

        engine.client.chat.completions.create.assert_awaited_once_with(
            model="my-model",
            messages=msgs,
            temperature=0.1,
            max_tokens=64,
        )

    @pytest.mark.asyncio
    async def test_handles_missing_usage_gracefully(self):
        """Should handle response with usage=None."""
        with patch("core.llm_engine.VLLMEngine.__init__", return_value=None):
            engine = VLLMEngine.__new__(VLLMEngine)
            engine.model = "test"
            engine.client = AsyncMock()

            choice = MagicMock()
            choice.message.content = "reply"
            response = MagicMock()
            response.choices = [choice]
            response.usage = None
            engine.client.chat.completions.create = AsyncMock(return_value=response)

        result = await engine.async_chat_completion([{"role": "user", "content": "hi"}])
        assert result.tokens_in == 0
        assert result.tokens_out == 0

    @pytest.mark.asyncio
    async def test_handles_none_content_gracefully(self):
        """Should handle response where message.content is None."""
        with patch("core.llm_engine.VLLMEngine.__init__", return_value=None):
            engine = VLLMEngine.__new__(VLLMEngine)
            engine.model = "test"
            engine.client = AsyncMock()

            choice = MagicMock()
            choice.message.content = None
            usage = MagicMock()
            usage.prompt_tokens = 0
            usage.completion_tokens = 0
            response = MagicMock()
            response.choices = [choice]
            response.usage = usage
            engine.client.chat.completions.create = AsyncMock(return_value=response)

        result = await engine.async_chat_completion([{"role": "user", "content": "hi"}])
        assert result.content == ""

    def test_sync_chat_completion_raises(self):
        """Sync path should raise NotImplementedError."""
        with patch("core.llm_engine.VLLMEngine.__init__", return_value=None):
            engine = VLLMEngine.__new__(VLLMEngine)
            engine.model = "test"
            engine.client = AsyncMock()

        with pytest.raises(NotImplementedError, match="async-only"):
            engine.chat_completion([{"role": "user", "content": "hi"}])


# ---------------------------------------------------------------------------
# Factory tests
# ---------------------------------------------------------------------------

class TestCreateLLMEngine:
    """Tests for the create_llm_engine factory."""

    def test_mock_provider(self):
        engine = create_llm_engine(provider="mock")
        assert isinstance(engine, MockLLMEngine)

    def test_mock_provider_with_seed(self):
        engine = create_llm_engine(provider="mock", seed=99)
        assert isinstance(engine, MockLLMEngine)
        # Two engines with same seed produce identical first random value
        engine2 = create_llm_engine(provider="mock", seed=99)
        assert engine.rng.random() == engine2.rng.random()

    @patch("core.llm_engine.OllamaEngine.__init__", return_value=None)
    def test_ollama_provider(self, mock_init):
        engine = create_llm_engine(provider="ollama", model="llama3")
        assert isinstance(engine, OllamaEngine)
        mock_init.assert_called_once_with(model="llama3", host="http://localhost:11434")

    @patch("core.llm_engine.OllamaEngine.__init__", return_value=None)
    def test_ollama_provider_custom_host(self, mock_init):
        engine = create_llm_engine(provider="ollama", model="mistral", host="http://gpu-server:11434")
        mock_init.assert_called_once_with(model="mistral", host="http://gpu-server:11434")

    @patch("core.llm_engine.VLLMEngine.__init__", return_value=None)
    def test_vllm_provider(self, mock_init):
        engine = create_llm_engine(provider="vllm", model="meta-llama/Llama-3-8B")
        assert isinstance(engine, VLLMEngine)
        mock_init.assert_called_once_with(
            model="meta-llama/Llama-3-8B",
            base_url="http://localhost:8000/v1",
            api_key="not-needed",
        )

    @patch("core.llm_engine.VLLMEngine.__init__", return_value=None)
    def test_vllm_provider_custom_url(self, mock_init):
        engine = create_llm_engine(
            provider="vllm",
            model="my-model",
            base_url="http://10.0.0.5:8000/v1",
            api_key="sk-test",
        )
        mock_init.assert_called_once_with(
            model="my-model",
            base_url="http://10.0.0.5:8000/v1",
            api_key="sk-test",
        )

    @patch("core.llm_engine.VLLMEngine.__init__", return_value=None)
    def test_openai_alias_maps_to_vllm(self, mock_init):
        engine = create_llm_engine(provider="openai", model="gpt-4")
        assert isinstance(engine, VLLMEngine)

    def test_unsupported_provider_raises(self):
        with pytest.raises(ValueError, match="Unsupported"):
            create_llm_engine(provider="llama.cpp")

    def test_case_insensitive_provider(self):
        engine = create_llm_engine(provider="Mock")
        assert isinstance(engine, MockLLMEngine)

    def test_default_provider_is_mock(self):
        engine = create_llm_engine()
        assert isinstance(engine, MockLLMEngine)


# ---------------------------------------------------------------------------
# BaseLLMEngine contract tests
# ---------------------------------------------------------------------------

class TestBaseLLMEngineContract:
    """Verify all engines satisfy the BaseLLMEngine interface."""

    def test_mock_is_subclass(self):
        assert issubclass(MockLLMEngine, BaseLLMEngine)

    def test_ollama_is_subclass(self):
        assert issubclass(OllamaEngine, BaseLLMEngine)

    def test_vllm_is_subclass(self):
        assert issubclass(VLLMEngine, BaseLLMEngine)

    @pytest.mark.asyncio
    async def test_mock_async_chat_completion_works(self):
        """MockLLMEngine should support both sync and async paths."""
        engine = MockLLMEngine(seed=42)
        result = await engine.async_chat_completion(
            messages=[{"role": "user", "content": "bid"}],
        )
        assert isinstance(result, LLMResponse)
        assert result.content
