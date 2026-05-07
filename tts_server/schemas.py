"""Pydantic models for HTTP request / response validation."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class BatchTTSRequest(BaseModel):
    texts:            list[str]                 = Field(..., min_length=1)
    language:         Optional[str]             = Field(
        default=None,
        description=(
            "Language for the synthesised speech.  Accepts:\n"
            "  • 'auto' / 'none' / '' → let the model auto-detect (DEFAULT)\n"
            "  • ISO-639-3 code → 'en', 'hi', 'gu', 'pa', 'bn', 'ta', 'te', "
            "'mr', 'kn', 'ml', 'zh', 'ja', 'ko', 'ar', …\n"
            "  • Canonical English name → 'English', 'Hindi', 'Gujarati', "
            "'Panjabi', 'Chinese', 'Japanese', 'Korean', …\n"
            "If omitted, the server's OMNIVOICE_LANGUAGE default is used."
        ),
    )
    voice:            Optional[str]             = Field(
        default=None,
        description=(
            "Voice profile name to use (e.g. 'ajay', 'soham'). Must be one of "
            "the profiles preloaded at server startup (see OMNIVOICE_VOICE_PROFILES). "
            "If omitted, the server's default voice is used."
        ),
    )
    speed:            Optional[float]           = Field(
        default=None,
        ge=0.25,
        le=3.0,
        description=(
            "Speaking-speed multiplier.  1.0 = normal pace, <1.0 slower, "
            ">1.0 faster.  Allowed range: 0.25 to 3.0.  If omitted, the "
            "server's OMNIVOICE_DEFAULT_SPEED is used (None → model default)."
        ),
    )
    use_high_quality: bool = False


class BatchTTSItem(BaseModel):
    id:           int
    audio_base64: str
    audio_ms:     int
    sample_rate:  int


class BatchTTSResponse(BaseModel):
    results:               list[BatchTTSItem]
    total_gen_ms:          float
    batch_size:            int
    server_batches_formed: int
    language:              str
    voice:                 str
    speed:                 Optional[float] = None
