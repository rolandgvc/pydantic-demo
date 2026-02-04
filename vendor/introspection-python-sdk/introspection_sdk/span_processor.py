import os
from urllib.parse import urljoin

from opentelemetry.context import Context
from opentelemetry.exporter.otlp.proto.http import Compression
from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
    OTLPSpanExporter as OTLPHTTPSpanExporter,
)
from opentelemetry.sdk.trace import ReadableSpan, Span, SpanProcessor
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    SimpleSpanProcessor,
)

from .config import AdvancedOptions
from .converters.openinference import (
    ConvertedReadableSpan,
    convert_openinference_to_genai,
    is_openinference_span,
)
from .utils import logger, platform_is_emscripten
from .version import VERSION


class IntrospectionSpanProcessor(SpanProcessor):
    """Span processor that sends traces to the introspection API."""

    def __init__(
        self,
        *,
        token: str | None = None,
        system_name: (str | None) = None,
        advanced: AdvancedOptions | None = None,
    ):
        # Use defaults if not provided
        self._advanced = advanced or AdvancedOptions()

        emscripten = platform_is_emscripten()

        if self._advanced.span_exporter:
            # Use provided exporter (for testing)
            span_exporter = self._advanced.span_exporter
        else:
            # Create default OTLP exporter
            base_url = self._advanced.base_url or os.getenv(
                "INTROSPECTION_BASE_URL", "https://api.nuraline.ai"
            )
            if not base_url:
                raise ValueError("INTROSPECTION_BASE_URL is not set")
            token = token or os.getenv("INTROSPECTION_TOKEN")
            if not token:
                raise ValueError("INTROSPECTION_TOKEN is not set")
            headers = {
                "User-Agent": f"introspection-sdk/{VERSION}",
                "Authorization": f"Bearer {token}",
                **(
                    self._advanced.additional_headers or {}
                ),  # TODO: Add validation for headers
            }
            if base_url.endswith("/v1/traces"):
                endpoint = base_url
            else:
                endpoint = urljoin(base_url, "/v1/traces")
            logger.info(
                "Initializing introspection with endpoint: %s", endpoint
            )
            span_exporter = OTLPHTTPSpanExporter(
                endpoint=endpoint,
                compression=Compression.NoCompression,
                headers=headers,
            )

        # Store exporter for debugging
        self._span_exporter = span_exporter
        if emscripten:  # pragma: no cover
            self._span_processor = SimpleSpanProcessor(span_exporter)
        else:
            # Configure BatchSpanProcessor with shorter timeout for faster sending
            self._span_processor = BatchSpanProcessor(
                span_exporter,
                max_queue_size=2048,
                export_timeout_millis=30000,
                schedule_delay_millis=1000,  # Send batches every 1 second
            )

    def on_start(
        self, span: Span, parent_context: Context | None = None
    ) -> None:
        logger.debug(
            f"Starting introspection span: {span.name} (trace_id={span.context.trace_id:x})"
        )
        self._span_processor.on_start(span, parent_context)

    def on_end(self, span: ReadableSpan) -> None:
        logger.debug(
            f"Ending introspection span: {span.name} (trace_id={span.context.trace_id:x})"
        )
        if not span.context.trace_flags.sampled:
            return

        scope = span.instrumentation_scope
        scope_name = scope.name if scope else None

        if is_openinference_span(scope_name):
            converted_attrs = convert_openinference_to_genai(span.attributes)
            span = ConvertedReadableSpan(span, converted_attrs)

        self._span_processor.on_end(span)

    def shutdown(self) -> None:
        logger.info("Shutting down introspection span processor")
        try:
            self._span_processor.shutdown()
        except Exception as e:
            logger.warning(f"Error during span processor shutdown: {e}")

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        logger.info("Flushing introspection span processor")
        return self._span_processor.force_flush(timeout_millis)
