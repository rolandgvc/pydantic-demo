"""Tests for IntrospectionSpanProcessor."""

import os

import pytest
from inline_snapshot import snapshot
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

from introspection_sdk import AdvancedOptions, IntrospectionSpanProcessor

from .test_utils import IncrementalIdGenerator, TimeGenerator, spans_to_dict


class TestIntrospectionSpanProcessor:
    """Test suite for IntrospectionSpanProcessor."""

    def test_span_processor_creation_with_token(self):
        """Test basic creation with token."""
        processor = IntrospectionSpanProcessor(token="test-token")
        assert processor is not None
        assert processor.force_flush(1000) is True

    def test_span_processor_creation_with_advanced_options(self):
        """Test creation with advanced options."""
        custom_headers = {"X-Custom-Header": "custom-value"}

        processor = IntrospectionSpanProcessor(
            token="test-token",
            advanced=AdvancedOptions(
                base_url="http://localhost:5418/v1/traces",
                additional_headers=custom_headers,
            ),
        )

        assert processor is not None
        assert processor.force_flush(1000) is True

    def test_span_processor_with_in_memory_exporter(self):
        """Test processor with in-memory exporter to validate spans."""
        exporter = InMemorySpanExporter()

        processor = IntrospectionSpanProcessor(
            token="test-token",
            advanced=AdvancedOptions(
                span_exporter=exporter,
                id_generator=IncrementalIdGenerator(),
                ns_timestamp_generator=TimeGenerator(),
            ),
        )

        # Create a tracer provider with our processor
        provider = TracerProvider(id_generator=IncrementalIdGenerator())
        provider.add_span_processor(processor)
        tracer = provider.get_tracer("test-tracer")

        # Create and end a span
        with tracer.start_as_current_span("test-span") as span:
            span.set_attribute("test.key", "test.value")
            span.set_attribute("test.number", 42)

        # Force flush to ensure spans are exported
        processor.force_flush(1000)

        # Convert to dict and compare with snapshot
        # Normalize timestamps for deterministic snapshots
        spans = spans_to_dict(
            exporter.get_finished_spans(),
            parse_json_attributes=False,
            normalize_timestamps=True,
        )
        spans = sorted(spans, key=lambda s: s["start_time"])
        assert spans == snapshot(
            [
                {
                    "name": "test-span",
                    "context": {
                        "trace_id": 1,
                        "span_id": 1,
                        "is_remote": False,
                    },
                    "parent": None,
                    "start_time": 1000000000,
                    "end_time": 2000000000,
                    "attributes": {
                        "test.key": "test.value",
                        "test.number": 42,
                    },
                },
            ]
        )

        provider.shutdown()

    def test_span_processor_preserves_span_attributes(self):
        """Test that span attributes are preserved through processing."""
        exporter = InMemorySpanExporter()

        processor = IntrospectionSpanProcessor(
            token="test-token",
            advanced=AdvancedOptions(
                span_exporter=exporter,
                id_generator=IncrementalIdGenerator(),
                ns_timestamp_generator=TimeGenerator(),
            ),
        )

        provider = TracerProvider(id_generator=IncrementalIdGenerator())
        provider.add_span_processor(processor)
        tracer = provider.get_tracer("test-tracer")

        # Create a span with multiple attributes
        with tracer.start_as_current_span("test-span-with-attributes") as span:
            span.set_attribute("service.name", "test-service")
            span.set_attribute("http.method", "GET")
            span.set_attribute("http.status_code", 200)
            span.set_status(trace.Status(trace.StatusCode.OK))

        processor.force_flush(1000)

        # Convert to dict and compare with snapshot
        # Normalize timestamps for deterministic snapshots
        spans = spans_to_dict(
            exporter.get_finished_spans(),
            parse_json_attributes=False,
            normalize_timestamps=True,
        )
        spans = sorted(spans, key=lambda s: s["start_time"])
        assert spans == snapshot(
            [
                {
                    "name": "test-span-with-attributes",
                    "context": {
                        "trace_id": 1,
                        "span_id": 1,
                        "is_remote": False,
                    },
                    "parent": None,
                    "start_time": 1000000000,
                    "end_time": 2000000000,
                    "attributes": {
                        "service.name": "test-service",
                        "http.method": "GET",
                        "http.status_code": 200,
                    },
                },
            ]
        )

        provider.shutdown()

    def test_span_processor_with_custom_exporter(self):
        """Test processor accepts custom exporter via AdvancedOptions."""
        exporter = InMemorySpanExporter()

        processor = IntrospectionSpanProcessor(
            token="test-token",
            advanced=AdvancedOptions(span_exporter=exporter),
        )

        # Verify processor was created successfully with custom exporter
        assert processor.force_flush(1000) is True

    def test_span_processor_shutdown(self):
        """Test processor shutdown."""
        processor = IntrospectionSpanProcessor(token="test-token")
        processor.shutdown()
        # Shutdown should complete without error
        assert True

    def test_span_processor_requires_token(self):
        """Test that token is required when not using custom exporter."""
        # Clear any env var that might be set
        old_token = os.environ.pop("INTROSPECTION_TOKEN", None)

        try:
            with pytest.raises(ValueError, match="INTROSPECTION_TOKEN"):
                IntrospectionSpanProcessor()
        finally:
            # Restore the env var if it was set
            if old_token:
                os.environ["INTROSPECTION_TOKEN"] = old_token

    def test_span_processor_uses_env_token(self):
        """Test processor uses INTROSPECTION_TOKEN env var."""
        # Save any existing token
        old_token = os.environ.get("INTROSPECTION_TOKEN")

        try:
            os.environ["INTROSPECTION_TOKEN"] = "env-token"

            processor = IntrospectionSpanProcessor()
            assert processor is not None
        finally:
            # Restore the original token or remove if it didn't exist
            if old_token:
                os.environ["INTROSPECTION_TOKEN"] = old_token
            else:
                os.environ.pop("INTROSPECTION_TOKEN", None)

    def test_span_processor_processes_multiple_spans(self):
        """Test processor handles multiple spans correctly."""
        exporter = InMemorySpanExporter()

        processor = IntrospectionSpanProcessor(
            token="test-token",
            advanced=AdvancedOptions(
                span_exporter=exporter,
                id_generator=IncrementalIdGenerator(),
                ns_timestamp_generator=TimeGenerator(),
            ),
        )

        provider = TracerProvider(id_generator=IncrementalIdGenerator())
        provider.add_span_processor(processor)
        tracer = provider.get_tracer("test-tracer")

        # Create multiple spans
        with tracer.start_as_current_span("span-1"):
            pass

        with tracer.start_as_current_span("span-2") as span:
            span.set_attribute("span.id", 2)

        with tracer.start_as_current_span("span-3") as span:
            span.set_attribute("span.id", 3)

        processor.force_flush(1000)

        # Convert to dict and compare with snapshot
        # Normalize timestamps for deterministic snapshots
        spans = spans_to_dict(
            exporter.get_finished_spans(),
            parse_json_attributes=False,
            normalize_timestamps=True,
        )
        spans = sorted(spans, key=lambda s: s["start_time"])
        assert spans == snapshot(
            [
                {
                    "name": "span-1",
                    "context": {
                        "trace_id": 1,
                        "span_id": 1,
                        "is_remote": False,
                    },
                    "parent": None,
                    "start_time": 1000000000,
                    "end_time": 2000000000,
                    "attributes": {},
                },
                {
                    "name": "span-2",
                    "context": {
                        "trace_id": 2,
                        "span_id": 2,
                        "is_remote": False,
                    },
                    "parent": None,
                    "start_time": 2000000000,
                    "end_time": 3000000000,
                    "attributes": {"span.id": 2},
                },
                {
                    "name": "span-3",
                    "context": {
                        "trace_id": 3,
                        "span_id": 3,
                        "is_remote": False,
                    },
                    "parent": None,
                    "start_time": 3000000000,
                    "end_time": 4000000000,
                    "attributes": {"span.id": 3},
                },
            ]
        )

        provider.shutdown()

    def test_span_processor_with_nested_spans(self):
        """Test processor handles nested spans correctly."""
        exporter = InMemorySpanExporter()

        processor = IntrospectionSpanProcessor(
            token="test-token",
            advanced=AdvancedOptions(
                span_exporter=exporter,
                id_generator=IncrementalIdGenerator(),
                ns_timestamp_generator=TimeGenerator(),
            ),
        )

        provider = TracerProvider(id_generator=IncrementalIdGenerator())
        provider.add_span_processor(processor)
        tracer = provider.get_tracer("test-tracer")

        # Create nested spans
        with tracer.start_as_current_span("parent-span") as parent:
            parent.set_attribute("level", "parent")
            with tracer.start_as_current_span("child-span") as child:
                child.set_attribute("level", "child")
                with tracer.start_as_current_span(
                    "grandchild-span"
                ) as grandchild:
                    grandchild.set_attribute("level", "grandchild")

        processor.force_flush(1000)

        # Convert to dict and compare with snapshot
        # Normalize timestamps for deterministic snapshots
        spans = spans_to_dict(
            exporter.get_finished_spans(),
            parse_json_attributes=False,
            normalize_timestamps=True,
        )
        spans = sorted(spans, key=lambda s: s["start_time"])
        assert spans == snapshot(
            [
                {
                    "name": "grandchild-span",
                    "context": {
                        "trace_id": 1,
                        "span_id": 3,
                        "is_remote": False,
                    },
                    "parent": {
                        "trace_id": 1,
                        "span_id": 2,
                        "is_remote": False,
                    },
                    "start_time": 1000000000,
                    "end_time": 2000000000,
                    "attributes": {"level": "grandchild"},
                },
                {
                    "name": "child-span",
                    "context": {
                        "trace_id": 1,
                        "span_id": 2,
                        "is_remote": False,
                    },
                    "parent": {
                        "trace_id": 1,
                        "span_id": 1,
                        "is_remote": False,
                    },
                    "start_time": 2000000000,
                    "end_time": 3000000000,
                    "attributes": {"level": "child"},
                },
                {
                    "name": "parent-span",
                    "context": {
                        "trace_id": 1,
                        "span_id": 1,
                        "is_remote": False,
                    },
                    "parent": None,
                    "start_time": 3000000000,
                    "end_time": 4000000000,
                    "attributes": {"level": "parent"},
                },
            ]
        )

        provider.shutdown()


class TestOTLPHttpCalls:
    """Test that OTLP HTTP calls are made correctly."""

    def test_otlp_call_made_with_correct_url(self):
        """Verify the exporter calls the correct OTLP endpoint."""
        import responses

        with responses.RequestsMock() as rsps:
            rsps.add(
                responses.POST,
                "http://test-endpoint.com/v1/traces",
                status=200,
            )

            processor = IntrospectionSpanProcessor(
                token="test-token",
                advanced=AdvancedOptions(
                    base_url="http://test-endpoint.com",
                ),
            )

            provider = TracerProvider()
            provider.add_span_processor(processor)
            tracer = provider.get_tracer("test")

            with tracer.start_as_current_span("test-span"):
                pass

            processor.force_flush(5000)

            assert len(rsps.calls) == 1
            assert (
                rsps.calls[0].request.url
                == "http://test-endpoint.com/v1/traces"
            )

            provider.shutdown()

    def test_otlp_call_made_with_correct_headers(self):
        """Verify Authorization and User-Agent headers are sent."""
        import responses

        with responses.RequestsMock() as rsps:
            rsps.add(
                responses.POST,
                "http://test-endpoint.com/v1/traces",
                status=200,
            )

            processor = IntrospectionSpanProcessor(
                token="my-secret-token",
                advanced=AdvancedOptions(
                    base_url="http://test-endpoint.com",
                    additional_headers={"X-Custom": "custom-value"},
                ),
            )

            provider = TracerProvider()
            provider.add_span_processor(processor)
            tracer = provider.get_tracer("test")

            with tracer.start_as_current_span("test-span"):
                pass

            processor.force_flush(5000)

            request = rsps.calls[0].request
            assert request.headers["Authorization"] == "Bearer my-secret-token"  # type: ignore[index]
            assert "introspection-sdk" in request.headers["User-Agent"]  # type: ignore[index,operator]
            assert request.headers["X-Custom"] == "custom-value"  # type: ignore[index]

            provider.shutdown()
