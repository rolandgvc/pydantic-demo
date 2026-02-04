"""
Logfire + Introspection Integration Example

Demonstrates dual export: traces sent to both Introspection and Logfire.

Run with:
    uv run -m introspection_examples.logfire.logfire_introspection
"""

try:
    import logfire
    import openai
except ImportError as e:
    raise ImportError(
        "Missing dependencies. Install with: uv sync --extra logfire"
    ) from e

from introspection_sdk import IntrospectionSpanProcessor


def main():
    introspection_processor = IntrospectionSpanProcessor()

    logfire.configure(
        additional_span_processors=[introspection_processor],
    )

    logfire.instrument_openai()

    client = openai.OpenAI()
    response = client.chat.completions.create(
        model="gpt-5-nano",
        messages=[{"role": "user", "content": "Say hello in one word."}],
    )
    print(f"Response: {response.choices[0].message.content}")

    introspection_processor.force_flush()
    logfire.shutdown()


if __name__ == "__main__":
    main()
