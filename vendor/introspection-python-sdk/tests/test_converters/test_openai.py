"""Tests for OpenAI format conversion to OTel Gen AI Semantic Conventions.

These tests validate that the converter functions produce output that conforms
to the OTel Gen AI semantic convention schemas.
"""

from pydantic import TypeAdapter

from introspection_sdk.converters.openai import (
    convert_responses_inputs_to_semconv,
    convert_responses_outputs_to_semconv,
)
from introspection_sdk.schemas.genai import InputMessages, OutputMessages

# Validators for schema compliance
input_messages_validator = TypeAdapter(InputMessages)
output_messages_validator = TypeAdapter(OutputMessages)


class TestInputMessagesSchemaCompliance:
    """Test that input conversions produce schema-compliant output."""

    def test_with_instructions(self):
        """Instructions are returned separately (not part of messages schema)."""
        inputs = [{"role": "user", "content": "Hello"}]
        messages, system_instructions = convert_responses_inputs_to_semconv(
            inputs, "Be helpful"
        )
        input_messages_validator.validate_python(messages)
        assert len(messages) == 1
        assert len(system_instructions) == 1
        assert system_instructions[0]["type"] == "text"

    def test_message_list_input(self):
        """List of messages converts to valid schema."""
        inputs = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        messages, _ = convert_responses_inputs_to_semconv(inputs, None)
        input_messages_validator.validate_python(messages)
        assert len(messages) == 2

    def test_function_call_input(self):
        """Function call input converts to valid schema with tool_call part."""
        inputs = [
            {
                "type": "function_call",
                "call_id": "call_123",
                "name": "get_weather",
                "arguments": '{"city": "Tokyo"}',
            }
        ]
        messages, _ = convert_responses_inputs_to_semconv(inputs, None)
        input_messages_validator.validate_python(messages)
        assert messages[0]["role"] == "assistant"
        assert messages[0]["parts"][0]["type"] == "tool_call"

    def test_function_call_output_input(self):
        """Function call output input converts to valid schema with tool_call_response."""
        inputs = [
            {
                "type": "function_call_output",
                "call_id": "call_123",
                "output": "Sunny, 25C",
            }
        ]
        messages, _ = convert_responses_inputs_to_semconv(inputs, None)
        input_messages_validator.validate_python(messages)
        assert messages[0]["role"] == "tool"
        assert messages[0]["parts"][0]["type"] == "tool_call_response"

    def test_empty_input(self):
        """Empty input produces valid empty list."""
        messages, system_instructions = convert_responses_inputs_to_semconv(
            None, None
        )
        input_messages_validator.validate_python(messages)
        assert messages == []
        assert system_instructions == []


class TestOutputMessagesSchemaCompliance:
    """Test that output conversions produce schema-compliant output."""

    def test_text_output(self):
        """Text output converts to valid schema."""
        outputs = [{"type": "message", "content": "Hello!"}]
        messages = convert_responses_outputs_to_semconv(outputs)
        output_messages_validator.validate_python(messages)
        assert len(messages) == 1
        assert messages[0]["role"] == "assistant"

    def test_function_call_output(self):
        """Function call output converts to valid schema with tool_call part."""
        outputs = [
            {
                "type": "function_call",
                "call_id": "call_456",
                "name": "search",
                "arguments": '{"query": "test"}',
            }
        ]
        messages = convert_responses_outputs_to_semconv(outputs)
        output_messages_validator.validate_python(messages)
        assert messages[0]["parts"][0]["type"] == "tool_call"

    def test_multiple_outputs(self):
        """Multiple outputs all convert to valid schema."""
        outputs = [
            {"content": "First response"},
            {"content": "Second response"},
        ]
        messages = convert_responses_outputs_to_semconv(outputs)
        output_messages_validator.validate_python(messages)
        assert len(messages) == 2

    def test_mixed_output_types(self):
        """Mix of message and function_call outputs converts to valid schema."""
        outputs = [
            {"content": "I'll search for that."},
            {
                "type": "function_call",
                "call_id": "call_789",
                "name": "search",
                "arguments": "{}",
            },
        ]
        messages = convert_responses_outputs_to_semconv(outputs)
        output_messages_validator.validate_python(messages)
        assert messages[0]["parts"][0]["type"] == "text"
        assert messages[1]["parts"][0]["type"] == "tool_call"

    def test_empty_output(self):
        """Empty output produces valid empty list."""
        messages = convert_responses_outputs_to_semconv([])
        output_messages_validator.validate_python(messages)
        assert messages == []
