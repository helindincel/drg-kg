"""Chunking module for dataset-agnostic text segmentation."""

from .strategies import (
    CHUNKING_PRESETS,
    Chunk,
    ChunkingStrategy,
    SentenceBasedChunker,
    TokenBasedChunker,
    create_chunker,
)
from .validators import ChunkValidator, validate_chunks

__all__ = [
    "CHUNKING_PRESETS",
    "Chunk",
    "ChunkValidator",
    "ChunkingStrategy",
    "SentenceBasedChunker",
    "TokenBasedChunker",
    "create_chunker",
    "validate_chunks",
]
