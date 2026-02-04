# Introspection Python SDK

Python SDK for tracking events and collecting feedback using OpenTelemetry OTLP Logs.

## Install

```shell
uv add introspection-sdk
# or
pip install introspection-sdk
```

## Environment Variables

```shell
export INTROSPECTION_TOKEN="intro_xxx"
export INTROSPECTION_BASE_URL="https://api.nuraline.ai"  # optional
```

## OpenTelemetry Integration

### Span Processor

For automatic tracing with logfire:

```python
from introspection_sdk import IntrospectionSpanProcessor
import logfire

introspection_span_processor = IntrospectionSpanProcessor()

logfire.configure(
    additional_span_processors=[introspection_span_processor],
)

logfire.instrument_openai()
```

### OpenAI Agents SDK

For tracing OpenAI Agents SDK applications:

```shell
pip install 'introspection-sdk[openai-agents]'
```

```python
from agents import set_trace_processors
from introspection_sdk import IntrospectionTracingProcessor

set_trace_processors([IntrospectionTracingProcessor()])
```

## Client Usage

### Quick Start

```python
from introspection_sdk import IntrospectionClient

client = IntrospectionClient()

# Track feedback - main use case
with client.set_user_id("user_123"):
    with client.set_conversation("conv_456", previous_response_id="msg_123"):
        client.feedback(
            "thumbs_up",
            comments="Great response!",
        )

# Shutdown flushes pending events
client.shutdown()
```

### Context Managers

Use context managers for automatic context propagation via OpenTelemetry baggage:

```python
from introspection_sdk import IntrospectionClient

client = IntrospectionClient()

# Combined contexts
with client.set_user_id("user_123"):
    with client.set_agent("support-bot"):
        with client.set_conversation("conv_456", previous_response_id="msg_123"):
            client.feedback("thumbs_up")  # inherits all context

client.shutdown()
```

Available context managers:
- `set_conversation(conversation_id?, previous_response_id?)` - Set conversation context
- `set_agent(agent_name, agent_id?)` - Set agent context
- `set_user_id(user_id)` - Set user context
- `set_anonymous_id(anonymous_id)` - Set anonymous ID context
- `set_baggage(**values)` - Set arbitrary baggage values

### API Reference

#### feedback

Track feedback on AI responses. Context is automatically extracted from baggage if not provided.

```python
# Simple feedback
client.feedback("thumbs_up")

# With comments and context
client.feedback(
    "thumbs_down",
    comments="Answer was off topic",
    conversation_id="conv_123",
    previous_response_id="msg_456",
)

# With extra properties
client.feedback("rating", score=4, category="helpfulness")
```

#### identify

Context manager that identifies a user and emits an identify event.

```python
with client.identify("user_123", traits={"email": "user@example.com", "plan": "pro"}):
    client.feedback("thumbs_up")  # inherits identity
```

#### track

Track any user action.

```python
client.track("Button Clicked", {"button_id": "submit"})
```

### Lifecycle Methods

- `flush(timeout_ms=30000)` - Flush pending events to server
- `shutdown()` - Shutdown client and flush events
- `reset()` - Clear client state

### Configuration Options

- `token` - API authentication token (default: `INTROSPECTION_TOKEN` env)
- `base_url` - API base URL (default: `https://api.nuraline.ai`)
- `service_name` - Service name for telemetry (default: `introspection-client`)
- `flush_interval_ms` - Flush interval in milliseconds (default: `5000`)
- `max_batch_size` - Max events per batch (default: `100`)

## Development

### Setup

```bash
uv pip install -e ".[dev]"  # or: pip install -e ".[dev]"
uv run pre-commit install   # or: pre-commit install
```

### Pre-commit Hooks

Hooks run automatically on `git commit`. To run manually:

```bash
uv run pre-commit run --all-files  # or: pre-commit run --all-files
```

### Manual Commands

```bash
uv run ruff format .    # Format code
uv run ruff check .     # Lint
uvx ty check            # Type check (or: uv run ty check)
uv run pytest           # Run tests
```
