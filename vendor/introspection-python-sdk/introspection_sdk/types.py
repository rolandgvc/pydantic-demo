"""Type definitions for the Introspection SDK."""

import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class GenAiAttributes(BaseModel):
    """GenAI semantic convention attributes."""

    model_config = ConfigDict(populate_by_name=True)

    request_model: str | None = Field(
        default=None, alias="gen_ai.request.model"
    )
    system: str | None = Field(default=None, alias="gen_ai.system")
    tool_definitions: str | None = Field(
        default=None, alias="gen_ai.tool.definitions"
    )
    input_messages: str | None = Field(
        default=None, alias="gen_ai.input.messages"
    )
    output_messages: str | None = Field(
        default=None, alias="gen_ai.output.messages"
    )
    system_instructions: str | None = Field(
        default=None, alias="gen_ai.system_instructions"
    )
    response_id: str | None = Field(default=None, alias="gen_ai.response.id")
    input_tokens: int | None = Field(
        default=None, alias="gen_ai.usage.input_tokens"
    )
    output_tokens: int | None = Field(
        default=None, alias="gen_ai.usage.output_tokens"
    )

    def to_attributes(self) -> dict[str, str | int]:
        """Convert to dictionary with OTEL semconv keys (dots), excluding None values."""
        return {
            k: v
            for k, v in self.model_dump(by_alias=True).items()
            if v is not None
        }


@dataclass
class FeedbackProperties:
    """Feedback event properties.

    Note: trace_id, span_id, identity, gen_ai.response.id, and gen_ai.conversation.id
    are automatically extracted from the current OpenTelemetry span/baggage.
    """

    name: str
    """Feedback name/action (e.g., "thumbs_up", "thumbs_down", "flag")"""

    comments: str | None = None
    """User's comments (e.g., "Answer was off topic")"""

    extra: dict[str, Any] = field(default_factory=dict)
    """Additional custom data"""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary, excluding None values."""
        result: dict[str, Any] = {"name": self.name}
        if self.comments is not None:
            result["comments"] = self.comments
        result.update(self.extra)
        return result


def _generate_message_id() -> str:
    """Generate a unique event ID.

    Format: intro_event_<timestamp>-<random8>
    """
    timestamp = hex(int(time.time() * 1000))[2:]
    random_part = uuid.uuid4().hex[:8]
    return f"intro_event_{timestamp}-{random_part}"


class EventName:
    """Standard event names used by the Introspection SDK."""

    IDENTIFY = "identify"
    FEEDBACK = "introspection.feedback"


class Defaults:
    """Default configuration values."""

    SERVICE_NAME = "introspection-client"
    BASE_URL = "https://api.nuraline.ai"
    FLUSH_INTERVAL_MS = 5000
    MAX_BATCH_SIZE = 100


class Severity:
    """Log severity text constants."""

    INFO = "INFO"


class LoggerName:
    """Logger names for OpenTelemetry instrumentation scope."""

    PYTHON_SDK = "introspection-sdk"


class ApiPath:
    """API endpoint paths."""

    LOGS = "/v1/logs"


class Attr:
    """Standard log attribute keys used by the Introspection SDK.

    These follow OpenTelemetry semantic conventions where applicable.
    """

    # Core event fields
    EVENT_NAME = "event.name"
    EVENT_ID = "event.id"

    # Identity
    USER_ID = "identity.user.id"
    ANONYMOUS_ID = "identity.anonymous.id"

    # Gen AI (OTel semantic conventions)
    CONVERSATION_ID = "gen_ai.conversation.id"
    PREVIOUS_RESPONSE_ID = "gen_ai.request.previous_response_id"
    AGENT_NAME = "gen_ai.agent.name"
    AGENT_ID = "gen_ai.agent.id"

    # Prefixes for dynamic keys
    PROPERTIES_PREFIX = "properties."
    TRAITS_PREFIX = "context.traits."


class Baggage:
    """Baggage keys used for context propagation.

    Note: Identity keys use underscores instead of dots for baggage compatibility.
    """

    USER_ID = "identity.user_id"
    ANONYMOUS_ID = "identity.anonymous_id"
    CONVERSATION_ID = "gen_ai.conversation.id"
    PREVIOUS_RESPONSE_ID = "gen_ai.request.previous_response_id"
    AGENT_NAME = "gen_ai.agent.name"
    AGENT_ID = "gen_ai.agent.id"
