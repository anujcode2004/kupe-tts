"""Language resolution utilities — bridge user input → OmniVoice expectations.

OmniVoice's :func:`generate` accepts:

* ``None`` or ``"none"`` → language-agnostic / auto mode
* an ISO-639-3 code (~700 supported, see ``omnivoice.utils.lang_map.LANG_IDS``)
* the canonical English name (e.g. ``"English"``, ``"Hindi"``,
  ``"Gujarati"``, ``"Panjabi"``, ``"Chinese"``, …)

This module gives the server a single normalisation point so we accept
forgiving aliases ("auto", "", whitespace, mixed case, common shortcuts) and
emit exactly what OmniVoice wants.
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger("omnivoice.lang")

# ---------------------------------------------------------------------------
# Pull the canonical maps from OmniVoice itself so we never drift.
# ---------------------------------------------------------------------------
try:
    from omnivoice.utils.lang_map import LANG_IDS, LANG_NAME_TO_ID
except Exception:  # pragma: no cover — fall back if package layout changes
    LANG_IDS = set()
    LANG_NAME_TO_ID = {}

# Auto-detect aliases the server will accept from clients.
_AUTO_TOKENS = {"", "auto", "none", "null", "nil", "any", "*"}

# Friendly shortcuts → canonical ISO code.  Extend as needed.
# (The full ISO 639-3 set lives in OmniVoice's lang_map; these are convenience
# aliases that aren't in the map but users commonly type.)
_SHORTCUT_TO_ID = {
    # Indic
    "pu": "pa",   # common typo for Punjabi (pa)
    "punjabi": "pa",
    # Hindi/Urdu romanisations
    "hin": "hi",
    "guj": "gu",
    "ben": "bn",
    "tam": "ta",
    "tel": "te",
    "mar": "mr",
    "kan": "kn",
    "mal": "ml",
    "ory": "ory",
    # English
    "eng": "en",
    # Chinese
    "chi": "zh",
    "mandarin": "zh",
    # Japanese / Korean
    "jpn": "ja",
    "kor": "ko",
}


def supported_codes_preview(limit: int = 12) -> list[str]:
    """Return a small sample of supported codes for documentation."""
    common = ["en", "hi", "gu", "pa", "bn", "ta", "te", "mr", "zh", "ja", "ko", "ar"]
    return [c for c in common if c in LANG_IDS][:limit] or sorted(LANG_IDS)[:limit]


def resolve_language(value: Optional[str]) -> Optional[str]:
    """Normalise a user-supplied language string.

    Returns:
        - ``None`` for any auto/empty token (server passes ``None`` to OmniVoice)
        - the canonical lowercase ISO 639-3 code if recognised
        - the canonical lowercase English name if matched in ``LANG_NAME_TO_ID``
        - the original string (lowercased + stripped) for OmniVoice to reject
          if it really doesn't know the value — we don't try to be smarter
          than the model.
    """
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    low = s.lower()
    if low in _AUTO_TOKENS:
        return None
    # Direct ISO code match
    if low in LANG_IDS:
        return low
    # Canonical English name match
    if low in LANG_NAME_TO_ID:
        return LANG_NAME_TO_ID[low]
    # Friendly shortcut
    if low in _SHORTCUT_TO_ID:
        return _SHORTCUT_TO_ID[low]
    # Hand off as-is and let OmniVoice raise a clear error
    logger.warning(
        "Unknown language code/name %r — passing through as-is.", value,
    )
    return low
