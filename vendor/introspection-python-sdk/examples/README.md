# Introspection SDK Examples

Examples demonstrating how to integrate Introspection SDK with popular observability platforms.

## Installation

From the examples directory:

```bash
cd examples

# Install all example dependencies
uv sync --extra all
```

## Required Environment Variables

Set these environment variables before running examples:

```bash
# Always required
export INTROSPECTION_TOKEN=your-token
```

## OpenInference Instrumentors

The OpenInference examples use these instrumentors for auto-instrumentation:

| Instrumentor | Import | Install |
|--------------|--------|---------|
| OpenAI | `from openinference.instrumentation.openai import OpenAIInstrumentor` | `uv add openinference-instrumentation-openai` |
| LangChain | `from openinference.instrumentation.langchain import LangChainInstrumentor` | `uv add openinference-instrumentation-langchain` |
| Anthropic | `from openinference.instrumentation.anthropic import AnthropicInstrumentor` | `uv add openinference-instrumentation-anthropic` |
| Google GenAI | `from openinference.instrumentation.google_genai import GoogleGenAIInstrumentor` | `uv add openinference-instrumentation-google-genai` |
