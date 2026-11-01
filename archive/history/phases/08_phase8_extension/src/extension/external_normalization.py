"""Pure normalization helpers for the Phase 8R6A external-data audit."""
from __future__ import annotations

import hashlib
import re
from typing import Any

ADDRESS_RE = re.compile(r"^(?:0x)?[0-9a-fA-F]{40}$")


def normalize_address(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not ADDRESS_RE.fullmatch(text):
        return None
    return "0x" + text.lower().removeprefix("0x")


def normalize_hex(value: Any) -> str | None:
    if value is None or (isinstance(value, float) and value != value):
        return None
    text = re.sub(r"\s+", "", str(value)).lower().removeprefix("0x")
    return text if text and re.fullmatch(r"[0-9a-f]+", text) else None


def normalize_opcode(value: Any) -> str | None:
    if value is None or (isinstance(value, float) and value != value):
        return None
    text = " ".join(str(value).split())
    return text or None


def digest(value: str | None) -> str | None:
    return hashlib.sha256(value.encode("utf-8")).hexdigest() if value is not None else None


def normalize_label(value: Any) -> int | None:
    if value in (1, "1", True, "ponzi", "Ponzi", "positive"):
        return 1
    if value in (0, "0", False, "benign", "non-ponzi", "negative"):
        return 0
    return None
