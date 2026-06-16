"""Chunking strategies for text segmentation."""

import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class Chunk:
    """Represents a text chunk with metadata."""

    chunk_id: str
    sequence_index: int
    text: str
    token_count: int
    char_count: int
    origin_dataset: str
    origin_file: str
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Convert chunk to dictionary."""
        return {
            "chunk_id": self.chunk_id,
            "sequence_index": self.sequence_index,
            "origin_dataset": self.origin_dataset,
            "origin_file": self.origin_file,
            "token_count": self.token_count,
            "char_count": self.char_count,
            "chunk_text": self.text,
            **self.metadata,
        }


class Tokenizer(ABC):
    """Abstract tokenizer interface."""

    @abstractmethod
    def encode(self, text: str) -> list[int]:
        """Encode text to token IDs."""
        pass

    @abstractmethod
    def decode(self, token_ids: list[int]) -> str:
        """Decode token IDs to text."""
        pass

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        pass


class TiktokenTokenizer(Tokenizer):
    """Tiktoken-based tokenizer (OpenAI compatible)."""

    def __init__(self, model: str = "cl100k_base"):
        """Initialize tiktoken tokenizer.

        Args:
            model: Tiktoken model name (default: cl100k_base for GPT-4)
        """
        try:
            import tiktoken

            self.encoding = tiktoken.get_encoding(model)
        except ImportError as err:
            raise ImportError(
                "tiktoken is required for TiktokenTokenizer. Install with: pip install tiktoken"
            ) from err

    def encode(self, text: str) -> list[int]:
        """Encode text to token IDs."""
        return self.encoding.encode(text)

    def decode(self, token_ids: list[int]) -> str:
        """Decode token IDs to text."""
        return self.encoding.decode(token_ids)

    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.encoding.encode(text))


class ChunkingStrategy(ABC):
    """Abstract chunking strategy."""

    @abstractmethod
    def chunk(self, text: str, origin_dataset: str, origin_file: str, **kwargs) -> list[Chunk]:
        """Chunk text into segments.

        Args:
            text: Input text to chunk
            origin_dataset: Name of the dataset
            origin_file: Path to source file
            **kwargs: Additional strategy-specific parameters

        Returns:
            List of Chunk objects
        """
        pass


def _generate_chunk_id(dataset: str, doc_id: str, chunk_index: int) -> str:
    """Generate deterministic chunk ID.

    Args:
        dataset: Dataset name
        doc_id: Document identifier
        chunk_index: Chunk index

    Returns:
        Unique chunk ID
    """
    # Use file basename as doc_id if not provided
    if not doc_id:
        doc_id = "doc_001"

    return f"{dataset}_{doc_id}_chunk_{chunk_index:03d}"


def _find_sentence_boundaries(text: str) -> list[int]:
    """Find sentence boundary positions in text.

    Args:
        text: Input text

    Returns:
        List of character positions where sentences end
    """
    # Simple sentence boundary detection
    # Match sentence endings: . ! ? followed by space or newline
    pattern = r"[.!?]\s+"
    boundaries = []
    for match in re.finditer(pattern, text):
        boundaries.append(match.end())
    return boundaries


def _find_paragraph_boundaries(text: str) -> list[int]:
    """Find paragraph boundary positions in text.

    Args:
        text: Input text

    Returns:
        List of character positions where paragraphs end
    """
    # Paragraph boundaries: double newline or newline followed by whitespace
    pattern = r"\n\s*\n"
    boundaries = []
    for match in re.finditer(pattern, text):
        boundaries.append(match.end())
    return boundaries


class TokenBasedChunker(ChunkingStrategy):
    """Token-based chunking strategy with boundary awareness."""

    def __init__(
        self,
        tokenizer: Tokenizer,
        chunk_size: int = 768,
        overlap_ratio: float = 0.15,
        respect_sentence_boundaries: bool = True,
        respect_paragraph_boundaries: bool = False,
    ):
        """Initialize token-based chunker.

        Args:
            tokenizer: Tokenizer instance
            chunk_size: Target chunk size in tokens (default: 768)
            overlap_ratio: Overlap ratio between chunks (0.0-1.0, default: 0.15)
            respect_sentence_boundaries: Whether to respect sentence boundaries
            respect_paragraph_boundaries: Whether to respect paragraph boundaries
        """
        if chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        if not 0 <= overlap_ratio <= 1:
            raise ValueError("overlap_ratio must be between 0 and 1")

        self.tokenizer = tokenizer
        self.chunk_size = chunk_size
        self.overlap_ratio = overlap_ratio
        self.respect_sentence_boundaries = respect_sentence_boundaries
        self.respect_paragraph_boundaries = respect_paragraph_boundaries
        self.overlap_tokens = int(chunk_size * overlap_ratio)

    def chunk(
        self,
        text: str,
        origin_dataset: str,
        origin_file: str,
        doc_id: str | None = None,
    ) -> list[Chunk]:
        """Chunk text using token-based strategy.

        Args:
            text: Input text to chunk
            origin_dataset: Name of the dataset
            origin_file: Path to source file
            doc_id: Optional document identifier

        Returns:
            List of Chunk objects
        """
        if not text.strip():
            return []

        # Generate doc_id from file path if not provided
        if not doc_id:
            doc_id = os.path.splitext(os.path.basename(origin_file))[0] or "doc_001"

        # Tokenize text
        token_ids = self.tokenizer.encode(text)
        total_tokens = len(token_ids)

        # Short documents: single chunk, no overlap
        if total_tokens <= self.chunk_size:
            chunk_text = text
            chunk = Chunk(
                chunk_id=_generate_chunk_id(origin_dataset, doc_id, 0),
                sequence_index=0,
                text=chunk_text,
                token_count=total_tokens,
                char_count=len(chunk_text),
                origin_dataset=origin_dataset,
                origin_file=origin_file,
                metadata={
                    "chunking_strategy": f"token_based_{self.chunk_size}_{int(self.overlap_ratio * 100)}pct",
                    "boundary_info": {
                        "starts_at_sentence": True,
                        "ends_at_sentence": True,
                        "starts_at_paragraph": True,
                        "ends_at_paragraph": True,
                    },
                },
            )
            return [chunk]

        # Find boundaries if needed
        sentence_boundaries = None
        paragraph_boundaries = None

        if self.respect_sentence_boundaries:
            sentence_boundaries = _find_sentence_boundaries(text)

        if self.respect_paragraph_boundaries:
            paragraph_boundaries = _find_paragraph_boundaries(text)

        chunks = []
        start_token = 0
        chunk_index = 0

        while start_token < total_tokens:
            # Calculate end token
            end_token = min(start_token + self.chunk_size, total_tokens)

            # Adjust boundaries if needed
            if sentence_boundaries and chunk_index > 0:
                # Try to align with sentence boundary
                end_token = self._align_to_boundary(
                    text, token_ids, end_token, sentence_boundaries, is_sentence=True
                )

            if paragraph_boundaries and chunk_index > 0:
                # Try to align with paragraph boundary
                end_token = self._align_to_boundary(
                    text, token_ids, end_token, paragraph_boundaries, is_sentence=False
                )

            # Extract chunk text
            chunk_token_ids = token_ids[start_token:end_token]
            chunk_text = self.tokenizer.decode(chunk_token_ids)

            # Create chunk
            chunk = Chunk(
                chunk_id=_generate_chunk_id(origin_dataset, doc_id, chunk_index),
                sequence_index=chunk_index,
                text=chunk_text,
                token_count=len(chunk_token_ids),
                char_count=len(chunk_text),
                origin_dataset=origin_dataset,
                origin_file=origin_file,
                metadata={
                    "chunking_strategy": f"token_based_{self.chunk_size}_{int(self.overlap_ratio * 100)}pct",
                    "boundary_info": self._get_boundary_info(chunk_text, text),
                },
            )
            chunks.append(chunk)

            # Calculate next start position with overlap
            if end_token >= total_tokens:
                break

            start_token = end_token - self.overlap_tokens
            if start_token < 0:
                start_token = 0

            chunk_index += 1

        return chunks

    def _align_to_boundary(
        self,
        text: str,
        token_ids: list[int],
        target_token: int,
        boundaries: list[int],
        is_sentence: bool = True,
    ) -> int:
        """Align chunk boundary to sentence/paragraph boundary.

        Args:
            text: Original text
            token_ids: Token IDs
            target_token: Target token position
            boundaries: List of boundary character positions
            is_sentence: Whether aligning to sentence boundaries

        Returns:
            Adjusted token position
        """
        if not boundaries:
            return target_token

        # Convert token position to character position (approximate)
        # This is a simplified approach - for production, use proper token-to-char mapping
        target_char = len(self.tokenizer.decode(token_ids[:target_token]))

        # Find nearest boundary
        nearest_boundary = None
        min_distance = float("inf")

        for boundary in boundaries:
            distance = abs(boundary - target_char)
            if distance < min_distance and distance < len(text) * 0.1:  # Within 10% of text
                min_distance = distance
                nearest_boundary = boundary

        if nearest_boundary is None:
            return target_token

        # Convert back to token position (approximate)
        boundary_text = text[:nearest_boundary]
        boundary_tokens = self.tokenizer.count_tokens(boundary_text)

        return min(boundary_tokens, len(token_ids))

    def _get_boundary_info(self, chunk_text: str, full_text: str) -> dict[str, bool]:
        """Get boundary information for chunk.

        Args:
            chunk_text: Chunk text
            full_text: Full original text

        Returns:
            Dictionary with boundary information
        """
        # Simple heuristics for boundary detection
        starts_at_sentence = chunk_text[0].isupper() if chunk_text else False
        ends_at_sentence = chunk_text.rstrip().endswith((".", "!", "?")) if chunk_text else False

        starts_at_paragraph = (
            chunk_text.startswith("\n") or chunk_text[0].isupper() if chunk_text else False
        )
        ends_at_paragraph = chunk_text.rstrip().endswith("\n\n") if chunk_text else False

        return {
            "starts_at_sentence": starts_at_sentence,
            "ends_at_sentence": ends_at_sentence,
            "starts_at_paragraph": starts_at_paragraph,
            "ends_at_paragraph": ends_at_paragraph,
        }


class SentenceBasedChunker(ChunkingStrategy):
    """Sentence-based chunking strategy."""

    def __init__(
        self,
        tokenizer: Tokenizer,
        target_chunk_size: int = 768,
        overlap_sentences: int = 2,
    ):
        """Initialize sentence-based chunker.

        Args:
            tokenizer: Tokenizer instance
            target_chunk_size: Target chunk size in tokens (default: 768)
            overlap_sentences: Number of sentences to overlap (default: 2)
        """
        self.tokenizer = tokenizer
        self.target_chunk_size = target_chunk_size
        self.overlap_sentences = overlap_sentences

    def chunk(
        self,
        text: str,
        origin_dataset: str,
        origin_file: str,
        doc_id: str | None = None,
    ) -> list[Chunk]:
        """Chunk text using sentence-based strategy.

        Args:
            text: Input text to chunk
            origin_dataset: Name of the dataset
            origin_file: Path to source file
            doc_id: Optional document identifier

        Returns:
            List of Chunk objects
        """
        if not text.strip():
            return []

        if not doc_id:
            doc_id = os.path.splitext(os.path.basename(origin_file))[0] or "doc_001"

        # Split into sentences
        sentences = re.split(r"([.!?]\s+)", text)
        # Recombine sentences with their punctuation
        sentences = [
            sentences[i] + (sentences[i + 1] if i + 1 < len(sentences) else "")
            for i in range(0, len(sentences), 2)
            if sentences[i].strip()
        ]

        chunks = []
        current_chunk_sentences: list[str] = []
        current_token_count = 0
        chunk_index = 0

        for sentence in sentences:
            sentence_tokens = self.tokenizer.count_tokens(sentence)

            # If adding this sentence would exceed target, create a chunk
            if (
                current_token_count + sentence_tokens > self.target_chunk_size
                and current_chunk_sentences
            ):
                chunk_text = "".join(current_chunk_sentences)
                chunk = Chunk(
                    chunk_id=_generate_chunk_id(origin_dataset, doc_id, chunk_index),
                    sequence_index=chunk_index,
                    text=chunk_text,
                    token_count=current_token_count,
                    char_count=len(chunk_text),
                    origin_dataset=origin_dataset,
                    origin_file=origin_file,
                    metadata={
                        "chunking_strategy": f"sentence_based_{self.target_chunk_size}",
                        "boundary_info": {
                            "starts_at_sentence": True,
                            "ends_at_sentence": True,
                            "starts_at_paragraph": False,
                            "ends_at_paragraph": False,
                        },
                    },
                )
                chunks.append(chunk)

                # Start new chunk with overlap
                overlap_start = max(0, len(current_chunk_sentences) - self.overlap_sentences)
                current_chunk_sentences = current_chunk_sentences[overlap_start:]
                current_token_count = sum(
                    self.tokenizer.count_tokens(s) for s in current_chunk_sentences
                )
                chunk_index += 1

            current_chunk_sentences.append(sentence)
            current_token_count += sentence_tokens

        # Add final chunk
        if current_chunk_sentences:
            chunk_text = "".join(current_chunk_sentences)
            chunk = Chunk(
                chunk_id=_generate_chunk_id(origin_dataset, doc_id, chunk_index),
                sequence_index=chunk_index,
                text=chunk_text,
                token_count=current_token_count,
                char_count=len(chunk_text),
                origin_dataset=origin_dataset,
                origin_file=origin_file,
                metadata={
                    "chunking_strategy": f"sentence_based_{self.target_chunk_size}",
                    "boundary_info": {
                        "starts_at_sentence": True,
                        "ends_at_sentence": True,
                        "starts_at_paragraph": False,
                        "ends_at_paragraph": False,
                    },
                },
            )
            chunks.append(chunk)

        return chunks


# Chunking presets for declarative usage
CHUNKING_PRESETS = {
    "small": {
        "strategy": "token_based",
        "chunk_size": 256,
        "overlap_ratio": 0.15,
    },
    "medium": {
        "strategy": "token_based",
        "chunk_size": 512,
        "overlap_ratio": 0.15,
    },
    "large": {
        "strategy": "token_based",
        "chunk_size": 768,
        "overlap_ratio": 0.15,
    },
    "xlarge": {
        "strategy": "token_based",
        "chunk_size": 1024,
        "overlap_ratio": 0.15,
    },
    "sentence": {
        "strategy": "sentence_based",
        "chunk_size": 512,
        "overlap_sentences": 2,
    },
}


def create_chunker(
    strategy: str = "token_based",
    tokenizer: Tokenizer | None = None,
    chunk_size: int | None = None,
    overlap_ratio: float | None = None,
    preset: str | None = None,
    **kwargs,
) -> ChunkingStrategy:
    """Factory function to create chunker - declarative interface.

    Can be used in two ways:
    1. Preset-based (declarative): create_chunker(preset="small")
    2. Parameter-based (low-level): create_chunker(strategy="token_based", chunk_size=200)

    Args:
        preset: Chunking preset name ("small", "medium", "large", "xlarge", "sentence")
               If preset is provided, it overrides strategy/chunk_size/overlap_ratio
        strategy: Chunking strategy ("token_based" or "sentence_based")
                 Only used if preset is not provided
        tokenizer: Tokenizer instance (default: TiktokenTokenizer)
        chunk_size: Target chunk size in tokens (default: 768)
                   Only used if preset is not provided
        overlap_ratio: Overlap ratio (for token_based, default: 0.15)
                      Only used if preset is not provided
        **kwargs: Additional strategy-specific parameters

    Returns:
        ChunkingStrategy instance

    Examples:
        # Declarative (preset-based)
        chunker = create_chunker(preset="small")

        # Low-level (parameter-based)
        chunker = create_chunker(strategy="token_based", chunk_size=200, overlap_ratio=0.15)
    """
    # If preset is provided, use preset configuration
    if preset:
        if preset not in CHUNKING_PRESETS:
            raise ValueError(
                f"Unknown preset: {preset}. Available presets: {list(CHUNKING_PRESETS.keys())}"
            )
        preset_config = CHUNKING_PRESETS[preset].copy()
        strategy = preset_config.pop("strategy")
        chunk_size = preset_config.pop("chunk_size", chunk_size)
        overlap_ratio = preset_config.pop("overlap_ratio", overlap_ratio)
        kwargs.update(preset_config)  # Add any remaining preset params

    # Set defaults if not provided
    if tokenizer is None:
        tokenizer = TiktokenTokenizer()

    if chunk_size is None:
        chunk_size = 768
    if overlap_ratio is None:
        overlap_ratio = 0.15

    if strategy == "token_based":
        return TokenBasedChunker(
            tokenizer=tokenizer, chunk_size=chunk_size, overlap_ratio=overlap_ratio, **kwargs
        )
    elif strategy == "sentence_based":
        overlap_sentences = kwargs.get("overlap_sentences", 2)
        return SentenceBasedChunker(
            tokenizer=tokenizer,
            target_chunk_size=chunk_size,
            overlap_sentences=overlap_sentences,
        )
    else:
        raise ValueError(f"Unknown chunking strategy: {strategy}")
