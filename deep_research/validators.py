"""Deterministic validators/gates for workflow step outputs.

These are intentionally lightweight and fast: they should catch obviously malformed
outputs early, before expensive downstream stages run.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class ValidationError(Exception):
    stage: str
    message: str

    def __str__(self) -> str:  # pragma: no cover
        return f"[{self.stage}] {self.message}"


_SOURCES_HEADER_RE = re.compile(r"^##?\s+Sources\s*$", re.IGNORECASE | re.MULTILINE)
_MARKDOWN_LINK_RE = re.compile(r"\[[^\]]+\]\((https?://[^)\s]+)\)")


def require_non_empty(stage: str, text: str) -> None:
    if not text or not text.strip():
        raise ValidationError(stage, "Output was empty")


def require_sources_section(stage: str, markdown: str) -> None:
    """Require a 'Sources' section header.

    We don't attempt full markdown parsing; this gate just prevents silent omission.
    """
    if not _SOURCES_HEADER_RE.search(markdown or ""):
        raise ValidationError(stage, "Missing a 'Sources' section")


def count_markdown_links(markdown: str) -> int:
    return len(_MARKDOWN_LINK_RE.findall(markdown or ""))


def require_min_links(stage: str, markdown: str, min_links: int) -> None:
    links = count_markdown_links(markdown)
    if links < min_links:
        raise ValidationError(stage, f"Expected at least {min_links} markdown link(s), found {links}")
