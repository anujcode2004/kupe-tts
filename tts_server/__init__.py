"""OmniVoice Production TTS Server — modular package.

Package layout
──────────────
  config.py        Environment-driven configuration knobs
  worker.py        GPU worker process (SageAttn, torch.compile, model load)
  batcher.py       DynamicBatcher — groups concurrent requests into GPU batches
  audio_utils.py   Crossfade, WAV encode / decode, numpy helpers
  text_utils.py    Sentence-boundary text splitting
  schemas.py       Pydantic request / response models
  app.py           FastAPI app factory, lifespan
  routes/          Endpoint routers (batch, streaming, health)
"""

__version__ = "2.1.0"
