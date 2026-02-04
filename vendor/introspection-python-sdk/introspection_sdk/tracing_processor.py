"""OpenAI Agents TracingProcessor for Introspection SDK.

Forwards OpenAI agent traces to the backend via OTLP with OTel Gen AI semantic
convention attributes.
"""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, Any, cast
from urllib.parse import urljoin

from opentelemetry import trace as otel_trace
from opentelemetry.exporter.otlp.proto.http import Compression
from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
    OTLPSpanExporter as OTLPHTTPSpanExporter,
)
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from .config import AdvancedOptions
from .converters.openai import (
    convert_responses_inputs_to_semconv,
    convert_responses_outputs_to_semconv,
)
from .utils import logger, platform_is_emscripten
from .version import VERSION

if TYPE_CHECKING:
    from openai.types.responses import FunctionTool
else:
    try:
        from openai.types.responses import FunctionTool
    except ImportError:
        FunctionTool = None

# Import OpenAI agents tracing types (optional dependency)
try:
    from agents.tracing import Span as AgentSpan
    from agents.tracing import Trace, TracingProcessor
    from agents.tracing.span_data import (
        AgentSpanData,
        FunctionSpanData,
        GenerationSpanData,
        HandoffSpanData,
        ResponseSpanData,
        SpanData,
    )
except ImportError as err:
    raise RuntimeError(
        "IntrospectionTracingProcessor requires the `openai-agents` package.\n"
        "You can install this with:\n"
        "    pip install 'introspection-sdk[openai-agents]'"
    ) from err


class IntrospectionTracingProcessor(TracingProcessor):
    """Forwards OpenAI agent traces to Introspection backend via OTLP.

    Extracts OTel Gen AI semantic convention attributes from span data:
    - Agent spans: gen_ai.agent.name, gen_ai.agent.id
    - Function spans: gen_ai.tool.name
    - Response spans: gen_ai.system_instructions, gen_ai.input/output.messages,
                      gen_ai.usage.input/output_tokens, gen_ai.tool.definitions
    """

    def __init__(
        self,
        *,
        token: str | None = None,
        advanced: AdvancedOptions | None = None,
    ):
        # Use defaults if not provided
        self._advanced = advanced or AdvancedOptions()

        if self._advanced.span_exporter:
            # Use provided exporter (for testing)
            exporter = self._advanced.span_exporter
        else:
            # Create default OTLP exporter
            base_url = self._advanced.base_url or os.getenv(
                "INTROSPECTION_BASE_URL", "https://api.nuraline.ai"
            )
            token = token or os.getenv("INTROSPECTION_TOKEN")
            if not token:
                raise ValueError("INTROSPECTION_TOKEN is not set")

            headers = {
                "User-Agent": f"introspection-sdk/{VERSION}",
                "Authorization": f"Bearer {token}",
                **(self._advanced.additional_headers or {}),
            }

            endpoint = (
                base_url
                if base_url.endswith("/v1/traces")
                else urljoin(base_url, "/v1/traces")
            )
            logger.info(f"IntrospectionTracingProcessor endpoint: {endpoint}")

            exporter = OTLPHTTPSpanExporter(
                endpoint=endpoint,
                compression=Compression.NoCompression,
                headers=headers,
            )

        self._tracer_provider = TracerProvider(
            id_generator=self._advanced.id_generator
        )
        if platform_is_emscripten():
            from opentelemetry.sdk.trace.export import SimpleSpanProcessor

            self._tracer_provider.add_span_processor(
                SimpleSpanProcessor(exporter)
            )
        else:
            self._tracer_provider.add_span_processor(
                BatchSpanProcessor(exporter, schedule_delay_millis=1000)
            )

        self._tracer = self._tracer_provider.get_tracer(
            "openai-agents", VERSION
        )
        self._spans: dict[str, otel_trace.Span] = {}

    def on_trace_start(self, trace: Trace | None = None) -> None:
        if trace:
            start_time = self._advanced.ns_timestamp_generator()
            otel_span = self._tracer.start_span(
                trace.name, start_time=start_time
            )
            self._spans[trace.trace_id] = otel_span

    def on_trace_end(self, trace: Trace | None = None) -> None:
        if trace:
            otel_span = self._spans.pop(trace.trace_id, None)
            if otel_span:
                end_time = self._advanced.ns_timestamp_generator()
                otel_span.end(end_time=end_time)

    def on_span_start(self, span: AgentSpan[SpanData] | None = None) -> None:
        if span:
            parent_id = span.parent_id or span.trace_id
            parent = self._spans.get(parent_id)
            context = (
                otel_trace.set_span_in_context(parent) if parent else None
            )

            span_data = span.span_data
            if isinstance(span_data, AgentSpanData | FunctionSpanData):
                name = span_data.name
            else:
                name = span_data.type
            start_time = self._advanced.ns_timestamp_generator()
            otel_span = self._tracer.start_span(
                name, context=context, start_time=start_time
            )
            self._spans[span.span_id] = otel_span

    def on_span_end(self, span: AgentSpan[SpanData] | None = None) -> None:
        if span:
            otel_span = self._spans.pop(span.span_id, None)
            if otel_span:
                span_data = span.span_data
                span_type = span_data.type

                # Extract gen_ai.* attributes based on span type
                # Use type casts to help type checker narrow union types
                if span_type == "agent":
                    self._process_agent_span(
                        otel_span,
                        cast(AgentSpan[AgentSpanData], span),
                        cast(AgentSpanData, span_data),
                    )
                elif span_type == "function":
                    self._process_function_span(
                        otel_span,
                        cast(AgentSpan[FunctionSpanData], span),
                        cast(FunctionSpanData, span_data),
                    )
                elif span_type == "response":
                    self._process_response_span(
                        otel_span,
                        cast(AgentSpan[ResponseSpanData], span),
                        cast(ResponseSpanData, span_data),
                    )
                elif span_type == "generation":
                    self._process_generation_span(
                        otel_span,
                        cast(AgentSpan[GenerationSpanData], span),
                        cast(GenerationSpanData, span_data),
                    )
                elif span_type == "handoff":
                    self._process_handoff_span(
                        otel_span,
                        cast(AgentSpan[HandoffSpanData], span),
                        cast(HandoffSpanData, span_data),
                    )

                # Keep raw span data for debugging
                otel_span.set_attribute(
                    "openai_agents.span_data", json.dumps(span_data.export())
                )
                end_time = self._advanced.ns_timestamp_generator()
                otel_span.end(end_time=end_time)

    def _process_agent_span(
        self,
        otel_span: otel_trace.Span,
        span: AgentSpan[AgentSpanData],
        span_data: AgentSpanData,
    ) -> None:
        """Extract attributes from agent spans."""
        otel_span.set_attribute("gen_ai.agent.name", span_data.name)

        if span_data.tools:
            # tools is a list of tool names
            otel_span.set_attribute(
                "gen_ai.tool.definitions", json.dumps(span_data.tools)
            )

        if span_data.handoffs:
            otel_span.set_attribute(
                "gen_ai.agent.handoffs", json.dumps(span_data.handoffs)
            )

        if span_data.output_type:
            otel_span.set_attribute(
                "gen_ai.agent.output_type", span_data.output_type
            )

    def _process_function_span(
        self,
        otel_span: otel_trace.Span,
        span: AgentSpan[FunctionSpanData],
        span_data: FunctionSpanData,
    ) -> None:
        """Extract attributes from function/tool spans."""
        otel_span.set_attribute("gen_ai.tool.name", span_data.name)

        if span_data.input:
            otel_span.set_attribute("gen_ai.tool.input", span_data.input)

        if span_data.output:
            otel_span.set_attribute(
                "gen_ai.tool.output", str(span_data.output)
            )

    def _process_response_span(
        self,
        otel_span: otel_trace.Span,
        span: AgentSpan[ResponseSpanData],
        span_data: ResponseSpanData,
    ) -> None:
        """Extract attributes from response spans."""
        response = span_data.response
        if not response:
            return

        # System instructions
        if response.instructions:
            system_instructions = [
                {"type": "text", "content": response.instructions}
            ]
            otel_span.set_attribute(
                "gen_ai.system_instructions", json.dumps(system_instructions)
            )

        # Tool definitions (with full details from Response object)
        if response.tools:
            tool_defs = []
            for tool in response.tools:
                if isinstance(tool, FunctionTool):
                    tool_def = {"name": tool.name}
                    if tool.description:
                        tool_def["description"] = tool.description
                    if tool.parameters:
                        tool_def["parameters"] = tool.parameters
                    tool_defs.append(tool_def)
                else:
                    # For non-function tools (web_search, file_search, etc.), use type as name
                    tool_defs.append({"name": tool.type})
            otel_span.set_attribute(
                "gen_ai.tool.definitions", json.dumps(tool_defs)
            )

        # Token usage
        if response.usage:
            if response.usage.input_tokens:
                otel_span.set_attribute(
                    "gen_ai.usage.input_tokens", response.usage.input_tokens
                )
            if response.usage.output_tokens:
                otel_span.set_attribute(
                    "gen_ai.usage.output_tokens", response.usage.output_tokens
                )

        # Model info
        if response.model:
            otel_span.set_attribute("gen_ai.request.model", response.model)

        # Response ID
        if response.id:
            otel_span.set_attribute("gen_ai.response.id", response.id)

        # Input messages (from span_data.input)
        if span_data.input:
            # Type cast to help type checker - ResponseSpanData.input is compatible
            input_messages, _ = convert_responses_inputs_to_semconv(
                cast("list[dict[str, Any]]", span_data.input), None
            )
            if input_messages:
                otel_span.set_attribute(
                    "gen_ai.input.messages", json.dumps(input_messages)
                )

        # Output messages (from response.output)
        if response.output:
            # Convert response.output to list of dicts for the converter
            output_items = [item.model_dump() for item in response.output]
            output_messages = convert_responses_outputs_to_semconv(
                output_items
            )
            if output_messages:
                otel_span.set_attribute(
                    "gen_ai.output.messages", json.dumps(output_messages)
                )

    def _process_generation_span(
        self,
        otel_span: otel_trace.Span,
        span: AgentSpan[GenerationSpanData],
        span_data: GenerationSpanData,
    ) -> None:
        """Extract attributes from generation spans."""
        if span_data.model:
            otel_span.set_attribute("gen_ai.request.model", span_data.model)

        if span_data.usage:
            usage = span_data.usage
            if isinstance(usage, dict):
                if "input_tokens" in usage:
                    otel_span.set_attribute(
                        "gen_ai.usage.input_tokens", usage["input_tokens"]
                    )
                if "output_tokens" in usage:
                    otel_span.set_attribute(
                        "gen_ai.usage.output_tokens", usage["output_tokens"]
                    )

        if span_data.input:
            otel_span.set_attribute(
                "gen_ai.input.messages", json.dumps(list(span_data.input))
            )

        if span_data.output:
            otel_span.set_attribute(
                "gen_ai.output.messages", json.dumps(list(span_data.output))
            )

    def _process_handoff_span(
        self,
        otel_span: otel_trace.Span,
        span: AgentSpan[HandoffSpanData],
        span_data: HandoffSpanData,
    ) -> None:
        """Extract attributes from handoff spans."""
        if span_data.from_agent:
            otel_span.set_attribute(
                "gen_ai.handoff.from_agent", span_data.from_agent
            )
        if span_data.to_agent:
            otel_span.set_attribute(
                "gen_ai.handoff.to_agent", span_data.to_agent
            )

    def shutdown(self) -> None:
        self._tracer_provider.shutdown()

    def force_flush(self) -> None:
        self._tracer_provider.force_flush()
