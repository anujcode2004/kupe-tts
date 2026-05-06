from __future__ import annotations

import re
from typing import List


def split_into_sentence_chunks(text: str) -> List[str]:
    text = text.strip()
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+", text)
    chunks = [p.strip() for p in parts if p.strip()]
    return chunks or [text]
