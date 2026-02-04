"""
Langfuse Integration Example

Demonstrates dual export: traces sent to both Introspection and Langfuse.

Uses explicit OTEL setup pattern per Langfuse cookbook:
https://langfuse.com/guides/cookbook/otel_integration_python_sdk

Run with:
    uv run -m introspection_examples.openinference.langfuse
"""

import base64
import os

try:
    import openai
    from langfuse import get_client
    from openinference.instrumentation.openai import OpenAIInstrumentor
except ImportError as e:
    raise ImportError(
        "Missing dependencies. Install with: uv sync --extra langfuse"
    ) from e

from introspection_sdk import IntrospectionSpanProcessor
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
    OTLPSpanExporter,
)
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

LANGFUSE_AUTH = base64.b64encode(
    f"{os.environ.get('LANGFUSE_PUBLIC_KEY')}:{os.environ.get('LANGFUSE_SECRET_KEY')}".encode()
).decode()

os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = (
    os.environ.get("LANGFUSE_BASE_URL", "https://cloud.langfuse.com")
    + "/api/public/otel"
)
os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = (
    f"Authorization=Basic {LANGFUSE_AUTH}"
)


def main():
    provider = TracerProvider()
    get_client()

    langfuse_processor = BatchSpanProcessor(OTLPSpanExporter())
    provider.add_span_processor(langfuse_processor)

    introspection_processor = IntrospectionSpanProcessor()
    provider.add_span_processor(introspection_processor)

    trace.set_tracer_provider(provider)

    OpenAIInstrumentor().instrument(tracer_provider=provider)

    client = openai.OpenAI()
    response = client.chat.completions.create(
        model="gpt-5-nano",
        messages=[{"role": "user", "content": "Say hello in one word."}],
    )
    print(f"Response: {response.choices[0].message.content}")

    langfuse_processor.force_flush()
    introspection_processor.force_flush()
    OpenAIInstrumentor().uninstrument()


if __name__ == "__main__":
    main()
