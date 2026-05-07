"""Text splitting utilities for streaming TTS."""

from __future__ import annotations

from omnivoice.utils.text import add_punctuation, chunk_text_punctuation

from .config import CHUNK_CHARS, FIRST_CHUNK_CHARS


def split_to_chunks(text: str, *, chunk_chars: int = CHUNK_CHARS) -> list[str]:
    """Split *text* into sentence-boundary chunks for streaming TTS.

    Uses OmniVoice's own abbreviation-aware punctuation splitter.
    Returns at least one chunk even for very short inputs.
    Every chunk is ensured to end with punctuation.
    """
    text = text.strip()
    if not text:
        return []

    chunks = (
        chunk_text_punctuation(text=text, chunk_len=chunk_chars, min_chunk_len=3)
        or [text]
    )
    return [add_punctuation(c) for c in chunks]


_MIN_FIRST_CHUNK = 15   # never produce a first chunk shorter than this


def split_first_chunk_early(text: str) -> tuple[str, str]:
    """Split off a short first chunk for low-latency first-audio delivery.

    Returns ``(first_chunk, remainder)``.  The first chunk targets
    ``FIRST_CHUNK_CHARS`` characters, always ends at a word boundary with
    punctuation, and is never shorter than ``_MIN_FIRST_CHUNK`` to ensure
    the model produces meaningful audio.

    Strategy
    ────────
    1. Split full text into normal-sized sentence chunks.
    2. If the first chunk is already short enough → use it directly.
    3. If not → try splitting the first sentence more aggressively by
       finding a comma / semicolon / word boundary near FIRST_CHUNK_CHARS.
    4. Merge tiny leading fragments (e.g. "Hello!") with the next chunk
       so the model always has enough text to synthesise.
    """
    text = text.strip()
    if not text:
        return "", ""

    # Very short text → no split
    if len(text) <= FIRST_CHUNK_CHARS + 20:
        return add_punctuation(text), ""

    # First pass: normal sentence-level split
    sentence_chunks = split_to_chunks(text, chunk_chars=CHUNK_CHARS)
    if len(sentence_chunks) <= 1:
        return add_punctuation(text), ""

    first_sentence = sentence_chunks[0]

    # If the first sentence is too short, merge with next
    if len(first_sentence) < _MIN_FIRST_CHUNK and len(sentence_chunks) > 1:
        merged = first_sentence + " " + sentence_chunks[1]
        consumed = 2
        first_candidate = merged
    else:
        consumed = 1
        first_candidate = first_sentence

    # If the candidate is within target, use it
    if len(first_candidate) <= FIRST_CHUNK_CHARS + 20:
        remainder_parts = sentence_chunks[consumed:]
        remainder = " ".join(remainder_parts).strip() if remainder_parts else ""
        return add_punctuation(first_candidate), remainder

    # Candidate is too long — try to find a sub-sentence break
    # Look for comma, semicolon, colon near the target length
    target = max(FIRST_CHUNK_CHARS, _MIN_FIRST_CHUNK)
    best_cut = -1
    for sep in (",", ";", ":", " — ", " – "):
        idx = first_candidate.find(sep, target // 2)
        if _MIN_FIRST_CHUNK <= idx <= target + 15:
            best_cut = idx + len(sep)
            break

    # Fallback: word boundary
    if best_cut < _MIN_FIRST_CHUNK:
        space = first_candidate.rfind(" ", _MIN_FIRST_CHUNK, target + 10)
        if space > _MIN_FIRST_CHUNK:
            best_cut = space

    if best_cut >= _MIN_FIRST_CHUNK:
        first_part = first_candidate[:best_cut].strip()
        leftover = first_candidate[best_cut:].strip()
        remainder_parts = ([leftover] if leftover else []) + sentence_chunks[consumed:]
        remainder = " ".join(remainder_parts).strip()
        return add_punctuation(first_part), remainder

    # Can't split further — use the whole first sentence
    remainder_parts = sentence_chunks[consumed:]
    remainder = " ".join(remainder_parts).strip() if remainder_parts else ""
    return add_punctuation(first_candidate), remainder
