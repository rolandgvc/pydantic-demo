"""
Arize Phoenix Integration Example

Demonstrates dual export: traces sent to both Introspection and Arize Phoenix.

Run with:
    uv run -m introspection_examples.openinference.arize
"""

import os

try:
    import openai
    from openinference.instrumentation.openai import OpenAIInstrumentor
    from phoenix.otel import register
except ImportError as e:
    raise ImportError(
        "Missing dependencies. Install with: uv sync --extra arize"
    ) from e

from introspection_sdk import IntrospectionSpanProcessor


def main():
    tracer_provider = register(
        project_name="dual-export-example",
        endpoint="https://otlp.arize.com/v1/traces",
        headers={
            "space_id": os.environ["ARIZE_SPACE_KEY"],
            "api_key": os.environ["ARIZE_API_KEY"],
        },
        batch=False,
    )

    introspection_processor = IntrospectionSpanProcessor()
    tracer_provider.add_span_processor(
        introspection_processor, replace_default_processor=False
    )

    OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)

    client = openai.OpenAI()
    response = client.chat.completions.create(
        model="gpt-5-nano",
        messages=[{"role": "user", "content": "Say hello in one word."}],
    )
    print(f"Response: {response.choices[0].message.content}")

    introspection_processor.force_flush()
    OpenAIInstrumentor().uninstrument()


if __name__ == "__main__":
    main()
