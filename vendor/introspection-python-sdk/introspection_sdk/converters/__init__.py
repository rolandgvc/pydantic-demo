"""Conversion functions for transforming provider formats to OTel Gen AI Semantic Conventions."""

from .openai import (
    convert_responses_inputs_to_semconv,
    convert_responses_outputs_to_semconv,
)
from .openinference import (
    ConvertedReadableSpan,
    convert_openinference_to_genai,
    is_openinference_span,
)

__all__ = [
    "convert_responses_inputs_to_semconv",
    "convert_responses_outputs_to_semconv",
    "ConvertedReadableSpan",
    "convert_openinference_to_genai",
    "is_openinference_span",
]
