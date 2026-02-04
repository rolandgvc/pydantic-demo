"""
Braintrust Integration Example

Demonstrates dual export: traces sent to both Introspection and Braintrust.

Run with:
    uv run -m introspection_examples.openinference.braintrust
"""

import os

try:
    import openai
    from openinference.instrumentation.openai import OpenAIInstrumentor
except ImportError as e:
    raise ImportError(
        "Missing dependencies. Install with: uv sync --extra braintrust"
    ) from e

from introspection_sdk import IntrospectionSpanProcessor
from introspection_sdk.config import AdvancedOptions
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider


def main():
    provider = TracerProvider()

    braintrust_processor = IntrospectionSpanProcessor(
        token=os.environ["BRAINTRUST_API_KEY"],
        advanced=AdvancedOptions(
            base_url="https://api.braintrust.dev/otel/v1/traces",
            additional_headers={
                "x-bt-parent": "project_name:dual-export-example",
            },
        ),
    )
    provider.add_span_processor(braintrust_processor)

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

    braintrust_processor.force_flush()
    introspection_processor.force_flush()
    OpenAIInstrumentor().uninstrument()


if __name__ == "__main__":
    main()
