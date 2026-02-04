"""OTel Gen AI Semantic Convention schemas."""

from .genai import (
    InputMessage,
    InputMessages,
    MessagePart,
    OutputMessage,
    OutputMessages,
    TextPart,
    ToolCallRequestPart,
    ToolCallResponsePart,
)

__all__ = [
    "TextPart",
    "ToolCallRequestPart",
    "ToolCallResponsePart",
    "MessagePart",
    "InputMessage",
    "OutputMessage",
    "InputMessages",
    "OutputMessages",
]
