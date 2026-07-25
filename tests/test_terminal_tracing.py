import pytest
from unittest.mock import patch
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult
from core.simulation_runner import SimulationRunner
from core.llm_engine import create_llm_engine
from core.auction import AuctionEngine
from core.market import MarketSimulator
from core.agents import BrandAgent


class InMemorySpanExporter(SpanExporter):
    def __init__(self):
        self.spans = []

    def export(self, spans):
        self.spans.extend(spans)
        return SpanExportResult.SUCCESS

    def shutdown(self):
        pass

    def force_flush(self, timeout_millis=None):
        pass

    def get_finished_spans(self):
        return self.spans

    def clear(self):
        self.spans.clear()


_exporter = InMemorySpanExporter()
_provider = TracerProvider()
_provider.add_span_processor(SimpleSpanProcessor(_exporter))
trace.set_tracer_provider(_provider)
_test_tracer = trace.get_tracer("test-service")


@pytest.fixture(autouse=True)
def _setup_telemetry():
    _exporter.clear()
    with patch("core.telemetry.tracer", _test_tracer):
        yield


def _make_simulation(num_agents=3):
    mock_engine = create_llm_engine(provider="mock", model="test-model")
    market = MarketSimulator(seed=42)
    auction_engine = AuctionEngine(market=market)
    agent_defs = [
        ("Alpha Corp", "aggressive", 1000.0, 50.0),
        ("Beta Inc", "conservative", 800.0, 60.0),
        ("Gamma LLC", "balanced", 900.0, 55.0),
    ][:num_agents]
    agents = [
        BrandAgent(name=n, role=r, budget=b, target_cpa=c, llm=mock_engine)
        for n, r, b, c in agent_defs
    ]
    return SimulationRunner(engine=auction_engine, agents=agents, num_rounds=1)


@pytest.mark.asyncio
async def test_auction_round_span():
    await _make_simulation().run()
    spans = _exporter.get_finished_spans()
    span_names = [s.name for s in spans]
    assert "auction_round" in span_names, f"Expected auction_round span, got {span_names}"
    auction_span = next(s for s in spans if s.name == "auction_round")
    assert auction_span.attributes.get("auction.agent_count") == 3


@pytest.mark.asyncio
async def test_agent_decide_bid_spans():
    await _make_simulation().run()
    spans = _exporter.get_finished_spans()
    agent_spans = [s for s in spans if s.name == "agent_decide_bid"]
    assert len(agent_spans) == 3, f"Expected 3 agent_decide_bid spans, got {len(agent_spans)}"
    agent_names = {s.attributes.get("agent.name") for s in agent_spans}
    assert agent_names == {"Alpha Corp", "Beta Inc", "Gamma LLC"}


@pytest.mark.asyncio
async def test_llm_inference_attributes():
    await _make_simulation(num_agents=1).run()
    spans = _exporter.get_finished_spans()
    llm_spans = [s for s in spans if s.name == "llm_inference"]
    assert len(llm_spans) >= 1, f"Expected at least 1 llm_inference span, got {len(llm_spans)}"
    llm_span = llm_spans[0]
    assert llm_span.attributes.get("llm.provider") == "MockLLMEngine"
    assert llm_span.attributes.get("llm.model") == "mock-llm-nash"
