import asyncio
from opentelemetry import trace
from core.telemetry import setup_telemetry
from core.simulation_runner import SimulationRunner
from core.llm_engine import create_llm_engine
from core.auction import AuctionEngine
from core.market import MarketSimulator
from core.agents import BrandAgent


async def main():
    print("Initializing Neuro-Symbolic Engine with Console Telemetry...")
    setup_telemetry(service_name="terminal-demo", export_to_console=True)
    
    mock_engine = create_llm_engine(provider="mock", model="test-model")
    
    market = MarketSimulator(seed=42)
    auction_engine = AuctionEngine(market=market)
    
    agents = [
        BrandAgent(name="Alpha Corp", role="aggressive", budget=1000.0, target_cpa=50.0, llm=mock_engine),
        BrandAgent(name="Beta Inc", role="conservative", budget=800.0, target_cpa=60.0, llm=mock_engine),
        BrandAgent(name="Gamma LLC", role="balanced", budget=900.0, target_cpa=55.0, llm=mock_engine),
    ]
    
    runner = SimulationRunner(engine=auction_engine, agents=agents, num_rounds=1)
    
    print("Executing Parallel Multi-Agent Round...\n")
    await runner.run()
    
    print("\nSimulation Complete. Flushing Traces to Terminal:")
    trace.get_tracer_provider().force_flush()

if __name__ == "__main__":
    asyncio.run(main())