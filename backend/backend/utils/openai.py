"""
OpenAI helper utilities.
"""
from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List


def sanitize_messages(messages: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Ensure every message payload sent to the OpenAI Chat API has a string content.

    Args:
        messages: Iterable of message dicts.

    Returns:
        List of sanitized message dicts safe to send to OpenAI.
    """
    sanitized: List[Dict[str, Any]] = []

    for msg in messages or []:
        if msg is None:
            continue

        if hasattr(msg, "model_dump"):
            msg_copy = msg.model_dump()
        elif hasattr(msg, "dict") and callable(getattr(msg, "dict")):
            try:
                msg_copy = msg.dict()
            except TypeError:
                msg_copy = dict(msg)
        else:
            msg_copy = dict(msg)
        content = msg_copy.get("content")

        # Don't skip assistant messages with tool_calls even if content is empty
        # OpenAI requires these messages to be present for tool responses
        has_tool_calls = msg_copy.get("role") == "assistant" and msg_copy.get("tool_calls")

        # Also don't skip tool messages - they must follow tool_calls
        is_tool_message = msg_copy.get("role") == "tool"

        if (content is None or content == "") and not has_tool_calls and not is_tool_message:
            # Skip messages with no content - OpenAI rejects empty strings
            # EXCEPT for assistant messages with tool_calls and tool messages
            continue

        # For messages with tool_calls, ensure content is at least empty string (not null)
        if has_tool_calls and (content is None or content == ""):
            msg_copy["content"] = None  # OpenAI allows null content if tool_calls present
        elif isinstance(content, (dict, list)):
            try:
                msg_copy["content"] = json.dumps(content)
            except (TypeError, ValueError):
                msg_copy["content"] = str(content)
        elif not isinstance(content, str):
            msg_copy["content"] = str(content)

        sanitized.append(msg_copy)

    return sanitized
