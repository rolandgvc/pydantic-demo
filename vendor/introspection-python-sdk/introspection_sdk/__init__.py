"""Introspection Python SDK."""

from introspection_sdk.client import IntrospectionClient
from introspection_sdk.span_processor import IntrospectionSpanProcessor
from introspection_sdk.types import (
    Attr,
    Baggage,
    EventName,
    FeedbackProperties,
)


# Lazy import for optional dependency (openai-agents)
def __getattr__(name: str):
    if name == "IntrospectionTracingProcessor":
        from introspection_sdk.tracing_processor import (
            IntrospectionTracingProcessor,
        )

        return IntrospectionTracingProcessor
    if name == "AdvancedOptions":
        from introspection_sdk.config import AdvancedOptions

        return AdvancedOptions
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "AdvancedOptions",
    "Attr",
    "Baggage",
    "EventName",
    "FeedbackProperties",
    "IntrospectionClient",
    "IntrospectionSpanProcessor",
    "IntrospectionTracingProcessor",
]
