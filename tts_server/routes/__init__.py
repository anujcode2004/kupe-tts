"""Route sub-package — aggregate all endpoint routers."""

from .batch import router as batch_router
from .health import router as health_router
from .streaming import router as streaming_router

__all__ = ["batch_router", "health_router", "streaming_router"]
