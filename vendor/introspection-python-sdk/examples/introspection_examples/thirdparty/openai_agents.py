"""
OpenAI Agents SDK Integration Example

Demonstrates using OpenAI Agents SDK's native tracing with Introspection.

Run with:
    uv run -m introspection_examples.thirdparty.openai_agents
"""

try:
    from agents import Agent, Runner, set_trace_processors
except ImportError as e:
    raise ImportError(
        "Missing dependencies. Install with: uv sync --extra openai-agents"
    ) from e

from introspection_sdk import IntrospectionTracingProcessor


def main():
    processor = IntrospectionTracingProcessor()
    set_trace_processors([processor])

    agent = Agent(
        name="Assistant",
        instructions="You are a helpful assistant. Be concise.",
    )

    result = Runner.run_sync(agent, "Say hello in one word.")
    print(f"Agent Response: {result.final_output}")

    processor.force_flush()
    processor.shutdown()


if __name__ == "__main__":
    main()
