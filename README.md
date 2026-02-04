# Deep Research

AI-powered deep research agent built with [Pydantic AI](https://ai.pydantic.dev/).

## Features

- Multi-agent architecture with supervisor, researcher, and report writer agents
- Parallel research across multiple subtopics
- DuckDuckGo search integration
- Structured outputs using Pydantic models
- Optional [Logfire](https://pydantic.dev/logfire) observability

## Installation

```bash
uv sync
```

## Setup

Create a `.env` file with your OpenAI API key:

```
OPENAI_API_KEY=your-key-here
```

## Usage

### CLI

```bash
# Single query
uv run research "What are the latest developments in quantum computing?"

# Save to file
uv run research -o report.md "History of artificial intelligence"

# Interactive mode
uv run research -i

# Custom model and parallelism
uv run research -m openai:gpt-4o -p 5 "Your query"
```

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `-m, --model` | Model to use | `openai:gpt-4o` |
| `-p, --parallel` | Max parallel researchers | `3` |
| `-o, --output` | Output file path | None |
| `-i, --interactive` | Interactive mode | False |

## Observability

This project uses Logfire spans for each pipeline stage. If you set an Introspection token, the CLI will also export those spans to Introspection.

Add to your `.env`:

```
INTROSPECTION_TOKEN=your-token-here
```

## How It Works

1. **Query Analysis** - Analyzes the input query
2. **Research Brief** - Creates a detailed research brief
3. **Planning** - Supervisor breaks query into parallel subtopics
4. **Research** - Multiple researcher agents search in parallel
5. **Compression** - Findings are organized and deduplicated
6. **Report** - Final comprehensive report is generated
