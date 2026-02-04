"""OpenAI format conversion functions for OTel Gen AI Semantic Conventions.

These functions convert OpenAI API formats (Responses API, Agents SDK) to the
standardized OTel Gen AI Semantic Convention format for gen_ai.input.messages
and gen_ai.output.messages attributes.
"""

from typing import Any


def convert_responses_inputs_to_semconv(
    inputs: list[dict[str, Any]] | None,
    instructions: str | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Convert Responses API inputs to OTel Gen AI Semantic Convention format.

    Args:
        inputs: List of input items from the request.
        instructions: System instructions/prompt.

    Returns:
        Tuple of (input_messages, system_instructions) in semconv format.
    """
    input_messages: list[dict[str, Any]] = []
    system_instructions: list[dict[str, Any]] = []

    if instructions:
        system_instructions.append({"type": "text", "content": instructions})

    if inputs:
        for inp in inputs:
            role = inp.get("role", "user")
            typ = inp.get("type")
            content = inp.get("content")

            if typ in (None, "message") and content:
                parts: list[dict[str, Any]] = []
                if isinstance(content, str):
                    parts.append({"type": "text", "content": content})
                elif isinstance(content, list):
                    for item in content:
                        if (
                            isinstance(item, dict)
                            and item.get("type") == "output_text"
                        ):
                            parts.append(
                                {
                                    "type": "text",
                                    "content": item.get("text", ""),
                                }
                            )
                        else:
                            parts.append(
                                item
                                if isinstance(item, dict)
                                else {"type": "text", "content": str(item)}
                            )
                input_messages.append({"role": role, "parts": parts})

            elif typ == "function_call":
                input_messages.append(
                    {
                        "role": "assistant",
                        "parts": [
                            {
                                "type": "tool_call",
                                "id": inp.get("call_id"),
                                "name": inp.get("name"),
                                "arguments": inp.get("arguments"),
                            }
                        ],
                    }
                )

            elif typ == "function_call_output":
                msg: dict[str, Any] = {
                    "role": "tool",
                    "parts": [
                        {
                            "type": "tool_call_response",
                            "id": inp.get("call_id"),
                            "response": inp.get("output"),
                        }
                    ],
                }
                if "name" in inp:
                    msg["name"] = inp["name"]
                input_messages.append(msg)

    return input_messages, system_instructions


def convert_responses_outputs_to_semconv(
    outputs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Convert Responses API outputs to OTel Gen AI Semantic Convention format.

    Args:
        outputs: List of output items from the response.

    Returns:
        List of output messages in semconv format.
    """
    output_messages: list[dict[str, Any]] = []

    for out in outputs:
        typ = out.get("type")
        content = out.get("content")

        if typ in (None, "message") and content:
            parts: list[dict[str, Any]] = []
            if isinstance(content, str):
                parts.append({"type": "text", "content": content})
            elif isinstance(content, list):
                for item in content:
                    if (
                        isinstance(item, dict)
                        and item.get("type") == "output_text"
                    ):
                        parts.append(
                            {"type": "text", "content": item.get("text", "")}
                        )
                    else:
                        parts.append(
                            item
                            if isinstance(item, dict)
                            else {"type": "text", "content": str(item)}
                        )
            output_messages.append({"role": "assistant", "parts": parts})

        elif typ == "function_call":
            output_messages.append(
                {
                    "role": "assistant",
                    "parts": [
                        {
                            "type": "tool_call",
                            "id": out.get("call_id"),
                            "name": out.get("name"),
                            "arguments": out.get("arguments"),
                        }
                    ],
                }
            )

    return output_messages
