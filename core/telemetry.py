"""OpenTelemetry tracing setup for the neuro-symbolic auction engine.

Call ``setup_telemetry()`` once at application startup to initialise the
global tracer provider.  All core modules import ``tracer`` from this
module so spans are automatically connected in a single trace tree.

Usage::

    from core.telemetry import setup_telemetry, tracer

    setup_telemetry("my-service")

    with tracer.start_as_current_span("my-operation"):
        ...
"""

import logging
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.sdk.resources import Resource

logger = logging.getLogger(__name__)

# Module-level tracer — safe to import before setup_telemetry() is called.
# Before setup, it returns a no-op tracer that costs nothing.
tracer: trace.Tracer = trace.get_tracer(__name__)


def setup_telemetry(
    service_name: str = "multi-agent-engine",
    export_to_console: bool = True,
) -> trace.Tracer:
    """Initialise the global TracerProvider and return the application tracer.

    Args:
        service_name:      Identifier for this service in trace backends.
        export_to_console: If True, print spans to stderr (dev mode).

    Returns:
        The configured ``Tracer`` instance.
    """
    global tracer

    resource = Resource.create({"service.name": service_name})
    provider = TracerProvider(resource=resource)

    if export_to_console:
        processor = BatchSpanProcessor(ConsoleSpanExporter())
        provider.add_span_processor(processor)

    trace.set_tracer_provider(provider)
    tracer = trace.get_tracer(service_name)

    logger.info(f"Telemetry initialised — service={service_name}, console={export_to_console}")
    return tracer
