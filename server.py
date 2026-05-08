#!/usr/bin/env python3
"""OmniVoice Production TTS Server — entry point.

All logic lives in the ``tts_server`` package.  This file is the thin
``uvicorn`` launcher.

    python server.py                        # default
    OMNIVOICE_SAGE_ATTN=1 python server.py  # SageAttention (default ON)
    OMNIVOICE_COMPILE=1 python server.py    # + torch.compile
"""

import atexit
import os
from typing import Optional

from tts_server.app import create_app

app = create_app()


def _start_ngrok_tunnel(port: int) -> Optional[str]:
    auth_token = os.getenv("NGROK_AUTH_TOKEN", "2udz3fP5K4xTUfeU5cVk6rwVKyL_67Zo7tAbUBvCRYjKYtSVd")
    if not auth_token:
        print("NGROK_AUTH_TOKEN not set; skipping ngrok tunnel.")
        return None

    try:
        from pyngrok import conf, ngrok
    except Exception as exc:
        print(f"pyngrok not available; skipping ngrok tunnel: {exc}")
        return None

    conf.get_default().auth_token = auth_token
    tunnel = ngrok.connect(addr=str(port), proto="http")
    public_url = tunnel.public_url

    def _shutdown_ngrok() -> None:
        try:
            ngrok.disconnect(public_url)
        except Exception:
            pass
        try:
            ngrok.kill()
        except Exception:
            pass

    atexit.register(_shutdown_ngrok)
    return public_url


if __name__ == "__main__":
    import uvicorn

    server_port = int(os.getenv("PORT", "8000"))
    ngrok_url = _start_ngrok_tunnel(server_port)
    if ngrok_url:
        print(f"Ngrok tunnel URL: {ngrok_url}")

    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=server_port,
        log_level="info",
        ws_ping_interval=20,
        ws_ping_timeout=30,
    )
