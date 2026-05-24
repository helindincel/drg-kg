"""
DRG API Module - FastAPI Server

Provides REST API endpoints for:
- Graph visualization
- Community reports
- Query provenance
- Knowledge graph exploration
"""

from .server import DRGAPIServer, create_app

__all__ = ["DRGAPIServer", "create_app"]
