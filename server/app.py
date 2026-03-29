"""OpenEnv validator entry — exposes the same FastAPI app as :mod:`api.app`."""

from api.app import app

__all__ = ["app"]
