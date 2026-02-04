"""OTel Gen AI Semantic Convention Pydantic models.

Based on the OpenTelemetry Gen AI semantic conventions:
- https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-input-messages.json
- https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-output-messages.json
"""

from typing import Annotated, Any, Literal

try:
    from pydantic import BaseModel, Field
except ImportError as e:
    raise ImportError(
        "pydantic is required to use the schemas module. "
        "Install it with: pip install 'introspection-sdk[test]'"
    ) from e


class TextPart(BaseModel):
    """Text content part."""

    type: Literal["text"]
    content: str


class ToolCallRequestPart(BaseModel):
    """Tool/function call request part."""

    type: Literal["tool_call"]
    name: str
    id: str | None = None
    arguments: Any = None


class ToolCallResponsePart(BaseModel):
    """Tool/function call response part."""

    type: Literal["tool_call_response"]
    response: Any
    id: str | None = None


# Union type for message parts - uses discriminated union on 'type' field
MessagePart = Annotated[
    TextPart | ToolCallRequestPart | ToolCallResponsePart,
    Field(discriminator="type"),
]


class InputMessage(BaseModel):
    """Input message in OTel Gen AI semantic convention format."""

    role: Literal["system", "user", "assistant", "tool"]
    parts: list[MessagePart]
    name: str | None = None


class OutputMessage(BaseModel):
    """Output message in OTel Gen AI semantic convention format."""

    role: Literal["system", "user", "assistant", "tool"]
    parts: list[MessagePart]
    finish_reason: str | None = None
    name: str | None = None


# Type aliases for validating lists of messages
InputMessages = list[InputMessage]
OutputMessages = list[OutputMessage]
