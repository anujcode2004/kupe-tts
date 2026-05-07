#!/usr/bin/env python3
"""OmniVoice Production TTS Server — entry point.

All logic lives in the ``tts_server`` package.  This file is the thin
``uvicorn`` launcher.

    python server.py                        # default
    OMNIVOICE_SAGE_ATTN=1 python server.py  # SageAttention (default ON)
    OMNIVOICE_COMPILE=1 python server.py    # + torch.compile
"""

from tts_server.app import create_app

app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        log_level="info",
        ws_ping_interval=20,
        ws_ping_timeout=30,
    )
