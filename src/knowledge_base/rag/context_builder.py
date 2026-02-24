"""Context builder for RAG queries.

This module builds context from retrieved documents by:
1. Deduplicating retrieved content
2. Ordering by relevance
3. Adding metadata and source information
4. Staying within token limits
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from knowledge_base.storage.portable.hybrid_search import HybridSearchResult

logger = logging.getLogger(__name__)


@dataclass
class ContextWindow:
    """A window of context for RAG."""

    chunks: list[HybridSearchResult] = field(default_factory=list)
    total_chars: int = 0
    total_tokens: int = 0
    sources: list[dict[str, Any]] = field(default_factory=list)

    def to_prompt_text(self, include_metadata: bool = True) -> str:
        """Convert context to text for prompt.

        Args:
            include_metadata: Whether to include metadata in context.

        Returns:
            Formatted context text.
        """
        parts = []

        for i, chunk in enumerate(self.chunks, 1):
            if include_metadata:
                source_info = f"[Source {i}"
                if chunk.document_name:
                    source_info += f" - {chunk.document_name}"
                source_info += "]"
                parts.append(f"{source_info}\n{chunk.text}\n")
            else:
                parts.append(chunk.text)

        return "\n---\n".join(parts)


class ContextBuilder:
    """Builds context for RAG queries.

    This class takes retrieved chunks and builds a coherent context
    window suitable for LLM prompts.

    Example:
        >>> builder = ContextBuilder(max_tokens=4000)
        >>> context = builder.build(search_results)
        >>> prompt_context = context.to_prompt_text()
    """

    def __init__(
        self,
        max_tokens: int = 4000,
        max_chunks: int = 20,
        chars_per_token: int = 4,
        include_metadata: bool = True,
        deduplicate: bool = True,
    ) -> None:
        """Initialize context builder.

        Args:
            max_tokens: Maximum tokens in context window.
            max_chunks: Maximum number of chunks to include.
            chars_per_token: Approximate characters per token.
            include_metadata: Whether to include metadata.
            deduplicate: Whether to deduplicate similar chunks.
        """
        self._max_tokens = max_tokens
        self._max_chunks = max_chunks
        self._chars_per_token = chars_per_token
        self._include_metadata = include_metadata
        self._deduplicate = deduplicate

    def build(
        self,
        results: list[HybridSearchResult],
        additional_context: Optional[str] = None,
    ) -> ContextWindow:
        """Build context window from search results.

        Args:
            results: Search results from hybrid search.
            additional_context: Optional additional context to prepend.

        Returns:
            ContextWindow with selected chunks.
        """
        # Deduplicate if enabled
        if self._deduplicate:
            results = self._deduplicate_results(results)

        # Limit number of chunks
        results = results[:self._max_chunks]

        # Build context window
        context = ContextWindow()
        current_tokens = 0

        # Add additional context if provided
        if additional_context:
            additional_tokens = len(additional_context) // self._chars_per_token
            current_tokens += additional_tokens

        # Add chunks until we hit the limit
        for result in results:
            chunk_tokens = len(result.text) // self._chars_per_token

            if current_tokens + chunk_tokens > self._max_tokens:
                # Check if we can fit a truncated version
                remaining_tokens = self._max_tokens - current_tokens
                if remaining_tokens > 100:  # Only add if we have room for meaningful content
                    truncated_text = result.text[:remaining_tokens * self._chars_per_token]
                    result.text = truncated_text + "..."
                    context.chunks.append(result)
                    context.total_chars += len(truncated_text)
                    context.total_tokens += remaining_tokens
                break

            context.chunks.append(result)
            context.total_chars += len(result.text)
            context.total_tokens += chunk_tokens
            current_tokens += chunk_tokens

            # Add source info
            source_info = {
                "chunk_id": result.chunk_id,
                "document_id": result.document_id,
                "document_name": result.document_name,
                "score": result.score,
            }
            if source_info not in context.sources:
                context.sources.append(source_info)

        return context

    def _deduplicate_results(
        self,
        results: list[HybridSearchResult],
        similarity_threshold: float = 0.9,
    ) -> list[HybridSearchResult]:
        """Remove highly similar chunks.

        Args:
            results: Search results.
            similarity_threshold: Jaccard similarity threshold.

        Returns:
            Deduplicated results.
        """
        if len(results) <= 1:
            return results

        deduplicated = [results[0]]

        for result in results[1:]:
            # Check similarity with already included results
            is_duplicate = False
            result_words = set(result.text.lower().split())

            for included in deduplicated:
                included_words = set(included.text.lower().split())

                # Jaccard similarity
                if result_words and included_words:
                    intersection = len(result_words & included_words)
                    union = len(result_words | included_words)
                    similarity = intersection / union if union > 0 else 0

                    if similarity > similarity_threshold:
                        is_duplicate = True
                        break

            if not is_duplicate:
                deduplicated.append(result)

        return deduplicated

    def build_with_entities(
        self,
        results: list[HybridSearchResult],
        entities: list[dict[str, Any]],
        max_tokens: int = 4000,
    ) -> ContextWindow:
        """Build context with entity information.

        Args:
            results: Search results.
            entities: List of entities with their information.
            max_tokens: Maximum tokens.

        Returns:
            ContextWindow with entity context.
        """
        # Build entity context
        entity_context = self._build_entity_context(entities)

        # Build document context with remaining tokens
        doc_tokens = max_tokens - len(entity_context) // self._chars_per_token

        # Temporarily adjust max tokens
        original_max = self._max_tokens
        self._max_tokens = doc_tokens

        context = self.build(results)

        # Restore original max
        self._max_tokens = original_max

        # Prepend entity context
        if entity_context and context.chunks:
            entity_chunk = HybridSearchResult(
                chunk_id="entities",
                document_id="entities",
                text=entity_context,
                score=1.0,
                document_name="Entity Context",
            )
            context.chunks.insert(0, entity_chunk)

        return context

    def _build_entity_context(self, entities: list[dict[str, Any]]) -> str:
        """Build context from entity information.

        Args:
            entities: List of entities.

        Returns:
            Entity context string.
        """
        if not entities:
            return ""

        lines = ["## Relevant Entities\n"]

        for entity in entities[:10]:  # Limit to 10 entities
            name = entity.get("name", "Unknown")
            entity_type = entity.get("entity_type", "entity")
            description = entity.get("description", "")

            lines.append(f"- **{name}** ({entity_type})")
            if description:
                lines.append(f"  {description}")

        return "\n".join(lines) + "\n\n"

    def build_with_graph_context(
        self,
        results: list[HybridSearchResult],
        graph_context: dict[str, Any],
    ) -> ContextWindow:
        """Build context with graph traversal results.

        Args:
            results: Search results.
            graph_context: Graph traversal context (nodes, edges).

        Returns:
            ContextWindow with graph context.
        """
        # Build graph context text
        graph_text = self._build_graph_context(graph_context)

        # Build document context
        doc_context = self.build(results)

        # Prepend graph context
        if graph_text and doc_context.chunks:
            graph_chunk = HybridSearchResult(
                chunk_id="graph",
                document_id="graph",
                text=graph_text,
                score=1.0,
                document_name="Knowledge Graph Context",
            )
            doc_context.chunks.insert(0, graph_chunk)
            doc_context.total_chars += len(graph_text)
            doc_context.total_tokens += len(graph_text) // self._chars_per_token

        return doc_context

    def _build_graph_context(self, graph_context: dict[str, Any]) -> str:
        """Build context from graph traversal.

        Args:
            graph_context: Graph data with nodes and edges.

        Returns:
            Graph context string.
        """
        nodes = graph_context.get("nodes", [])
        edges = graph_context.get("edges", [])

        if not nodes:
            return ""

        lines = ["## Related Knowledge\n"]

        # List entities
        lines.append("### Entities")
        for node in nodes[:15]:
            name = node.get("name", node.get("id", "Unknown"))
            node_type = node.get("type", "entity")
            lines.append(f"- {name} ({node_type})")

        # List relationships
        if edges:
            lines.append("\n### Relationships")
            for edge in edges[:10]:
                source = edge.get("source", "?")
                target = edge.get("target", "?")
                relation = edge.get("relation_type", "relates to")
                lines.append(f"- {source} → {relation} → {target}")

        return "\n".join(lines) + "\n\n"
