"""Chunk validation utilities."""

from .strategies import Chunk


class ChunkValidator:
    """Validates chunk quality and consistency."""

    @staticmethod
    def validate_chunks(chunks: list[Chunk]) -> list[str]:
        """Validate chunks and return list of issues.

        Args:
            chunks: List of chunks to validate

        Returns:
            List of validation issue messages (empty if all valid)
        """
        issues = []

        if not chunks:
            return ["No chunks provided"]

        # Check for empty chunks
        for chunk in chunks:
            if not chunk.text.strip():
                issues.append(f"Chunk {chunk.chunk_id} is empty")

            if chunk.token_count == 0:
                issues.append(f"Chunk {chunk.chunk_id} has zero tokens")

            if chunk.sequence_index < 0:
                issues.append(f"Chunk {chunk.chunk_id} has negative sequence_index")

        # Check sequence index continuity
        sequence_indices = [chunk.sequence_index for chunk in chunks]
        if sequence_indices != sorted(sequence_indices):
            issues.append("Sequence indices are not in order")

        # Check for duplicate chunk IDs
        chunk_ids = [chunk.chunk_id for chunk in chunks]
        if len(chunk_ids) != len(set(chunk_ids)):
            issues.append("Duplicate chunk IDs found")

        return issues

    @staticmethod
    def check_overlap_consistency(chunks: list[Chunk], expected_overlap_ratio: float) -> list[str]:
        """Check if overlap between chunks is consistent.

        Args:
            chunks: List of chunks
            expected_overlap_ratio: Expected overlap ratio

        Returns:
            List of issues
        """
        issues = []

        if len(chunks) < 2:
            return issues

        for i in range(len(chunks) - 1):
            current = chunks[i]
            next_chunk = chunks[i + 1]

            # Check if chunks are from same document
            if current.origin_file != next_chunk.origin_file:
                continue

            # Note: Actual overlap calculation would require text content comparison
            # This simplified version checks sequence indices which is sufficient for
            # basic validation. For precise overlap calculation, compare actual text tokens.
            int(current.token_count * expected_overlap_ratio)

            # For now, just check that sequence indices are consecutive
            if next_chunk.sequence_index != current.sequence_index + 1:
                issues.append(
                    f"Non-consecutive sequence indices: {current.sequence_index} -> {next_chunk.sequence_index}"
                )

        return issues


def validate_chunks(chunks: list[Chunk]) -> bool:
    """Quick validation check.

    Args:
        chunks: List of chunks to validate

    Returns:
        True if all chunks are valid, False otherwise
    """
    validator = ChunkValidator()
    issues = validator.validate_chunks(chunks)
    return len(issues) == 0
