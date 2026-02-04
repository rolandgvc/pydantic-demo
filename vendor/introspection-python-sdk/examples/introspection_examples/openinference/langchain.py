"""
LangChain Integration Example

Demonstrates dual export: LangChain agent traces sent to both Introspection
and LangSmith.

Run with:
    uv run -m introspection_examples.openinference.langchain
"""

import os

try:
    from langchain.agents import create_agent
    from openinference.instrumentation.langchain import LangChainInstrumentor
except ImportError as e:
    raise ImportError(
        "Missing dependencies. Install with: uv sync --extra langchain"
    ) from e

from introspection_sdk import IntrospectionSpanProcessor
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
    OTLPSpanExporter,
)
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor


def main():
    provider = TracerProvider()

    langsmith_exporter = OTLPSpanExporter(
        endpoint="https://api.smith.langchain.com/otel/v1/traces",
        headers={
            "x-api-key": os.environ["LANGSMITH_API_KEY"],
            "Langsmith-Project": os.environ.get(
                "LANGSMITH_PROJECT", "default"
            ),
        },
    )
    langsmith_processor = BatchSpanProcessor(langsmith_exporter)
    provider.add_span_processor(langsmith_processor)

    introspection_processor = IntrospectionSpanProcessor()
    provider.add_span_processor(introspection_processor)

    trace.set_tracer_provider(provider)

    LangChainInstrumentor().instrument(tracer_provider=provider)

    def get_weather(city: str) -> str:
        """Get weather for a given city."""
        return f"It's always sunny in {city}!"

    agent = create_agent(
        model="openai:gpt-5-nano",
        tools=[get_weather],
        system_prompt="You are a helpful assistant",
    )

    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "What is the weather in San Francisco?",
                }
            ]
        }
    )
    print(f"Agent Response: {result}")

    langsmith_processor.force_flush()
    introspection_processor.force_flush()
    LangChainInstrumentor().uninstrument()


if __name__ == "__main__":
    main()
